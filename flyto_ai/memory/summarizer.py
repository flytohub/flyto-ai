# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Conversation summarizer — compresses old messages via LLM."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
Summarize the following conversation history into a concise paragraph.
Preserve key facts, decisions, user preferences, and any important context.
Write in the same language as the conversation.

Conversation:
{conversation}

Summary:"""


class ConversationSummarizer:
    """Summarizes old messages when a session exceeds threshold."""

    def __init__(self, provider, threshold: int = 20, keep_recent: int = 10) -> None:
        self._provider = provider
        self.threshold = threshold
        self.keep_recent = keep_recent

    async def maybe_summarize(self, session_id: str, store) -> Optional[str]:
        """If message count > threshold, summarize old messages and replace them.

        Returns the summary text, or None if no summarization was needed.
        """
        count = await store.get_message_count(session_id)
        if count <= self.threshold:
            return None

        messages = await store.get_messages(session_id)
        if len(messages) <= self.threshold:
            return None

        old_messages = messages[:-self.keep_recent]
        summary = await self._call_llm_summarize(old_messages)
        if summary:
            await store.replace_old_with_summary(
                session_id, summary, keep_recent=self.keep_recent
            )
            logger.info(
                "Summarized %d messages for session %s", len(old_messages), session_id
            )
        return summary

    async def _call_llm_summarize(self, messages: list) -> Optional[str]:
        """Call LLM to generate a summary of the given messages."""
        conversation_text = self._format_messages(messages)
        prompt = _SUMMARIZE_PROMPT.format(conversation=conversation_text)

        try:
            # Use the provider's chat method with minimal settings
            content, _, _, _ = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a concise summarizer. Output only the summary.",
                tools=[],
                dispatch_fn=lambda n, a: {"ok": False, "error": "no tools"},
                max_rounds=1,
            )
            return content
        except Exception as e:
            logger.warning("Summarization LLM call failed: %s", e)
            return None

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format messages list into readable text."""
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            lines.append("{}: {}".format(role, content))
        return "\n".join(lines)
