# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Prompt DNA — decompose/compose system prompts as modular blocks.

The production system_prompt.py uses Layer A/B/C. This module provides:
1. Decomposition — split the production prompt into named blocks
2. Composition — assemble blocks back into a full prompt
3. Block-level operations — replace, reorder, remove individual blocks

No changes to the production code — blocks are a read-only view for evolution.
"""
import re
from typing import Dict, List, Optional

from flyto_ai.evolution.models import PromptBlock, PromptCandidate


def get_baseline_candidate(mode: str = "execute") -> PromptCandidate:
    """Create a PromptCandidate from the current production prompt.

    This is the 'generation 0' baseline that evolution starts from.
    """
    from flyto_ai.prompt.system_prompt import (
        LAYER_A_POLICY, LAYER_B_EXECUTE, LAYER_B_YAML,
        LAYER_B_TOOLLESS, LAYER_C_GATES,
    )

    layer_b_map = {
        "execute": LAYER_B_EXECUTE,
        "yaml": LAYER_B_YAML,
        "toolless": LAYER_B_TOOLLESS,
    }
    layer_b = layer_b_map.get(mode, LAYER_B_EXECUTE)

    blocks = _decompose_layer(LAYER_A_POLICY, "policy", priority_start=10)
    blocks += _decompose_layer(layer_b, "behavior", priority_start=30)
    blocks += _decompose_layer(LAYER_C_GATES, "gate", priority_start=50)

    return PromptCandidate(
        id="baseline",
        mutation_type="baseline",
        mutation_desc="Current production prompt",
        blocks=blocks,
        generation=0,
    )


def _decompose_layer(text: str, category: str, priority_start: int) -> List[PromptBlock]:
    """Split a layer into blocks by ## headings."""
    sections = _split_by_headings(text)
    blocks = []
    for i, (heading, content) in enumerate(sections):
        name = _heading_to_name(heading, category, i)
        full_content = content.strip()
        if heading:
            full_content = "## {}\n{}".format(heading, full_content)
        blocks.append(PromptBlock(
            name=name,
            content=full_content,
            category=category,
            priority=priority_start + i,
        ))
    return blocks


def _split_by_headings(text: str) -> List[tuple]:
    """Split text into (heading, body) tuples by ## markers.

    Text before the first ## heading gets heading=''.
    """
    lines = text.strip().split("\n")
    sections = []
    current_heading = ""
    current_lines = []

    for line in lines:
        m = re.match(r'^##\s+(.+)', line)
        if m:
            if current_lines or current_heading:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    if current_lines or current_heading:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections


def _heading_to_name(heading: str, category: str, index: int) -> str:
    """Convert a heading to a block name like 'policy_output_contract'."""
    if not heading:
        return "{}_preamble_{}".format(category, index)
    # Normalize: lowercase, replace spaces/special with underscore
    slug = re.sub(r'[^a-z0-9]+', '_', heading.lower()).strip('_')
    slug = slug[:40]
    return "{}_{}".format(category, slug)


def compose_prompt(blocks: List[PromptBlock], **kwargs) -> str:
    """Compose a full system prompt from ordered blocks."""
    sorted_blocks = sorted(blocks, key=lambda b: b.priority)
    parts = [b.content for b in sorted_blocks]
    prompt = "\n\n".join(parts)
    for k, v in kwargs.items():
        prompt = prompt.replace("{{" + k + "}}", str(v))
    return prompt


def replace_block(
    candidate: PromptCandidate,
    block_name: str,
    new_content: str,
    new_id: Optional[str] = None,
    mutation_desc: str = "",
) -> PromptCandidate:
    """Create a new candidate with one block replaced."""
    new_blocks = []
    found = False
    for b in candidate.blocks:
        if b.name == block_name:
            new_blocks.append(PromptBlock(
                name=b.name,
                content=new_content,
                category=b.category,
                priority=b.priority,
            ))
            found = True
        else:
            new_blocks.append(b.model_copy())

    if not found:
        raise ValueError("Block not found: {}".format(block_name))

    cid = new_id or "{}_mut{}".format(candidate.id, candidate.generation + 1)
    return PromptCandidate(
        id=cid,
        parent_id=candidate.id,
        mutation_type="mutation",
        mutation_desc=mutation_desc or "Replace block: {}".format(block_name),
        blocks=new_blocks,
        generation=candidate.generation + 1,
    )


def reorder_blocks(
    candidate: PromptCandidate,
    new_order: Dict[str, int],
    new_id: Optional[str] = None,
) -> PromptCandidate:
    """Create a new candidate with blocks reordered by new priorities."""
    new_blocks = []
    for b in candidate.blocks:
        priority = new_order.get(b.name, b.priority)
        new_blocks.append(PromptBlock(
            name=b.name,
            content=b.content,
            category=b.category,
            priority=priority,
        ))

    cid = new_id or "{}_reord".format(candidate.id)
    return PromptCandidate(
        id=cid,
        parent_id=candidate.id,
        mutation_type="reorder",
        mutation_desc="Reorder: {}".format(new_order),
        blocks=new_blocks,
        generation=candidate.generation + 1,
    )


def remove_block(
    candidate: PromptCandidate,
    block_name: str,
    new_id: Optional[str] = None,
) -> PromptCandidate:
    """Create a new candidate with one block removed."""
    new_blocks = [b.model_copy() for b in candidate.blocks if b.name != block_name]
    if len(new_blocks) == len(candidate.blocks):
        raise ValueError("Block not found: {}".format(block_name))

    cid = new_id or "{}_del".format(candidate.id)
    return PromptCandidate(
        id=cid,
        parent_id=candidate.id,
        mutation_type="simplify",
        mutation_desc="Remove block: {}".format(block_name),
        blocks=new_blocks,
        generation=candidate.generation + 1,
    )


def crossover(
    parent_a: PromptCandidate,
    parent_b: PromptCandidate,
    blocks_from_b: List[str],
    new_id: Optional[str] = None,
) -> PromptCandidate:
    """Create a child candidate by taking some blocks from parent B."""
    b_blocks = {b.name: b for b in parent_b.blocks}
    new_blocks = []
    for b in parent_a.blocks:
        if b.name in blocks_from_b and b.name in b_blocks:
            new_blocks.append(b_blocks[b.name].model_copy())
        else:
            new_blocks.append(b.model_copy())

    cid = new_id or "cross_{}_{}".format(parent_a.id[:6], parent_b.id[:6])
    return PromptCandidate(
        id=cid,
        parent_id=parent_a.id,
        mutation_type="crossover",
        mutation_desc="Crossover: {} blocks from {}".format(
            len(blocks_from_b), parent_b.id,
        ),
        blocks=new_blocks,
        generation=max(parent_a.generation, parent_b.generation) + 1,
    )
