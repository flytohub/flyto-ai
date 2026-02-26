# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""System prompt fragments injected into Claude Code sessions."""

ROLE_PREAMBLE = """\
You are a coding assistant managed by Flyto AI. Your task is to implement \
code changes as instructed. Focus on correctness and minimal diffs.
"""

GUARDIAN_NOTICE = """\
## Safety Rules
- Do NOT modify .env, credentials, or secret files.
- Do NOT run destructive bash commands (rm -rf /, sudo rm, etc.).
- Only write files with common source code extensions.
- A Guardian hook monitors your tool calls. Blocked calls will fail \
automatically — do not retry blocked operations.
"""

VERIFICATION_NOTICE = """\
## Verification
After you finish coding, your changes will be automatically verified via \
a browser-based test recipe (screenshot + text extraction). If verification \
fails, you will receive feedback with the failure details and must fix \
the issues.
"""

INDEXER_CONTEXT_HEADER = """\
## Codebase Context (from flyto-indexer)
The following analysis was gathered from the project index:
"""


def build_system_prompt(
    indexer_context: str = "",
    has_verification: bool = False,
) -> str:
    """Assemble the full system prompt for a Claude Code session."""
    parts = [ROLE_PREAMBLE, GUARDIAN_NOTICE]
    if has_verification:
        parts.append(VERIFICATION_NOTICE)
    if indexer_context:
        parts.append(INDEXER_CONTEXT_HEADER)
        parts.append(indexer_context)
    return "\n".join(parts)
