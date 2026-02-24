# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Prompt mutation engine — generate prompt variants for evolution.

Mutation strategies:
  A. Mutation — small changes to individual blocks
  B. Crossover — combine best parts of two candidates
  C. Simplify — remove redundancy, tighten rules
  D. Reorder — change block priorities
  E. Strengthen — add emphasis to weak rules
"""
import hashlib
import logging
import random
import re
from typing import List, Optional

from flyto_ai.evolution.blocks import (
    crossover, get_baseline_candidate, remove_block, reorder_blocks,
    replace_block,
)
from flyto_ai.evolution.models import EvolutionConfig, PromptBlock, PromptCandidate

logger = logging.getLogger(__name__)


class PromptMutator:
    """Generate prompt variants using rule-based mutation strategies.

    For LLM-assisted mutation, see mutate_with_llm().
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self._config = config or EvolutionConfig()
        self._rng = random.Random(42)

    def generate_candidates(
        self,
        parents: List[PromptCandidate],
        generation: int,
        count: int = 5,
    ) -> List[PromptCandidate]:
        """Generate N candidate prompts from parent pool.

        Strategies used:
        - 1 elite (unchanged best parent)
        - crossover between top 2 parents (if available)
        - mutations of random parents
        - 1 simplification attempt
        """
        if not parents:
            parents = [get_baseline_candidate(self._config.mode)]

        candidates = []

        # Elite: keep best parent unchanged
        elite = parents[0].model_copy(deep=True)
        elite.id = "gen{}_elite".format(generation)
        elite.generation = generation
        candidates.append(elite)

        remaining = count - 1

        # Crossover (if 2+ parents)
        if len(parents) >= 2 and remaining > 0:
            child = self._try_crossover(parents[0], parents[1], generation)
            if child:
                candidates.append(child)
                remaining -= 1

        # Mutations
        strategies = [
            self._mutate_strengthen,
            self._mutate_reword,
            self._mutate_reorder,
            self._mutate_simplify,
            self._mutate_add_example,
        ]

        for i in range(remaining):
            parent = parents[i % len(parents)]
            strategy = strategies[i % len(strategies)]
            try:
                child = strategy(parent, generation, i)
                if child:
                    candidates.append(child)
            except Exception as e:
                logger.debug("Mutation failed: %s", e)
                # Fallback: add parent copy with minor change
                fallback = parent.model_copy(deep=True)
                fallback.id = "gen{}_{}_fallback".format(generation, i)
                fallback.generation = generation
                candidates.append(fallback)

        return candidates

    def _try_crossover(
        self,
        parent_a: PromptCandidate,
        parent_b: PromptCandidate,
        generation: int,
    ) -> Optional[PromptCandidate]:
        """Crossover: take behavior blocks from parent B, rest from A."""
        b_block_names = [
            b.name for b in parent_b.blocks if b.category == "behavior"
        ]
        if not b_block_names:
            return None
        child = crossover(parent_a, parent_b, b_block_names)
        child.id = "gen{}_cross".format(generation)
        child.generation = generation
        return child

    def _mutate_strengthen(
        self, parent: PromptCandidate, generation: int, index: int,
    ) -> Optional[PromptCandidate]:
        """Add emphasis markers (⛔, HARD, NEVER) to a rule block."""
        # Find a policy/gate block to strengthen
        candidates = [b for b in parent.blocks if b.category in ("policy", "gate")]
        if not candidates:
            return None

        block = self._rng.choice(candidates)
        content = block.content

        # Strengthen patterns
        replacements = [
            (r'(?i)\bshould\b', 'MUST'),
            (r'(?i)\bavoid\b', 'NEVER'),
            (r'(?i)\btry to\b', 'ALWAYS'),
            (r'(?i)\bprefer\b', 'MUST'),
        ]
        changed = False
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, content, count=1)
            if new_content != content:
                content = new_content
                changed = True
                break

        if not changed:
            # Add emphasis prefix if no pattern matched
            if "⛔" not in content and "HARD" not in content:
                lines = content.split("\n")
                if lines:
                    lines[0] = lines[0] + " (HARD)"
                    content = "\n".join(lines)

        return replace_block(
            parent, block.name, content,
            new_id="gen{}_{}_str".format(generation, index),
            mutation_desc="Strengthen: {}".format(block.name),
        )

    def _mutate_reword(
        self, parent: PromptCandidate, generation: int, index: int,
    ) -> Optional[PromptCandidate]:
        """Reword a block: convert bullets to if/then rules."""
        candidates = [b for b in parent.blocks if b.category == "behavior"]
        if not candidates:
            return None

        block = self._rng.choice(candidates)
        content = block.content

        # Convert "- Do X" patterns to "- If user asks → do X"
        lines = content.split("\n")
        new_lines = []
        changed = False
        for line in lines:
            m = re.match(r'^(\s*-\s+)(Do NOT|NEVER|Don\'t)\s+(.+)', line)
            if m and not changed:
                prefix, neg, rest = m.groups()
                new_lines.append("{}If tempted to {} → STOP. This is forbidden.".format(
                    prefix, rest[:60],
                ))
                changed = True
            else:
                new_lines.append(line)

        if not changed:
            return None

        return replace_block(
            parent, block.name, "\n".join(new_lines),
            new_id="gen{}_{}_rew".format(generation, index),
            mutation_desc="Reword: {} (if/then rules)".format(block.name),
        )

    def _mutate_reorder(
        self, parent: PromptCandidate, generation: int, index: int,
    ) -> Optional[PromptCandidate]:
        """Move failure-handling rules to higher priority."""
        failure_blocks = [
            b for b in parent.blocks
            if "fail" in b.name.lower() or "error" in b.name.lower()
        ]
        if not failure_blocks:
            return None

        new_order = {}
        for b in failure_blocks:
            new_order[b.name] = max(1, b.priority - 10)

        child = reorder_blocks(parent, new_order)
        child.id = "gen{}_{}_reord".format(generation, index)
        child.generation = generation
        return child

    def _mutate_simplify(
        self, parent: PromptCandidate, generation: int, index: int,
    ) -> Optional[PromptCandidate]:
        """Remove the longest non-essential block to reduce prompt size."""
        removable = [
            b for b in parent.blocks
            if b.category == "gate" and len(b.content) > 100
        ]
        if not removable:
            return None

        # Remove the longest gate block
        longest = max(removable, key=lambda b: len(b.content))
        try:
            child = remove_block(parent, longest.name)
            child.id = "gen{}_{}_simp".format(generation, index)
            child.generation = generation
            return child
        except ValueError:
            return None

    def _mutate_add_example(
        self, parent: PromptCandidate, generation: int, index: int,
    ) -> Optional[PromptCandidate]:
        """Add a concrete example to a behavior block."""
        behavior_blocks = [b for b in parent.blocks if b.category == "behavior"]
        if not behavior_blocks:
            return None

        block = self._rng.choice(behavior_blocks)
        content = block.content

        # Add example for failure handling if not already present
        if "Example:" not in content and "例" not in content:
            example = (
                "\n\nExample (failure case):\n"
                "- User: search the web for X\n"
                "- browser.launch → ok=false, error='chromium not installed'\n"
                "- Correct response: '瀏覽器啟動失敗：chromium 未安裝。請執行 playwright install chromium'\n"
                "- Wrong response: '我已經幫您搜尋了X，以下是結果...' ← NEVER do this"
            )
            content = content + example

        return replace_block(
            parent, block.name, content,
            new_id="gen{}_{}_ex".format(generation, index),
            mutation_desc="Add example: {}".format(block.name),
        )


async def mutate_with_llm(
    candidate: PromptCandidate,
    block_name: str,
    instruction: str,
    provider,
    generation: int = 0,
) -> Optional[PromptCandidate]:
    """Use an LLM to rewrite a specific block.

    This is more creative but also less predictable than rule-based mutation.
    """
    block = None
    for b in candidate.blocks:
        if b.name == block_name:
            block = b
            break
    if not block:
        return None

    messages = [{"role": "user", "content": (
        "Rewrite the following system prompt block.\n\n"
        "## Instruction\n{}\n\n"
        "## Current Block (name: {})\n```\n{}\n```\n\n"
        "Output ONLY the rewritten block text, no explanation."
    ).format(instruction, block_name, block.content)}]

    system = (
        "You are a prompt engineering expert. Rewrite the block as instructed. "
        "Keep the same purpose but improve clarity and effectiveness. "
        "Output only the rewritten text."
    )

    try:
        async def _noop(n, a):
            return {"ok": False}

        content, _, _, _ = await provider.chat(
            messages, system, tools=[], dispatch_fn=_noop, max_rounds=1,
        )
        if content and len(content.strip()) > 10:
            # Strip code fences if present
            text = re.sub(r'^```\w*\n?', '', content.strip())
            text = re.sub(r'\n?```$', '', text)
            return replace_block(
                candidate, block_name, text.strip(),
                new_id="gen{}_llm_{}".format(generation, block_name[:10]),
                mutation_desc="LLM rewrite: {}".format(instruction[:50]),
            )
    except Exception as e:
        logger.warning("LLM mutation failed: %s", e)

    return None
