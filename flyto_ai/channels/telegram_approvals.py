# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Telegram approval notifications — push approval requests to chat,
handle approve/reject via inline keyboard callbacks.

Usage:
    notifier = TelegramApprovalNotifier(bot_token, engine_url)
    await notifier.send_approval(chat_id, approval)
    # When callback received with data "approve:<id>" or "reject:<id>":
    await notifier.handle_callback(callback_data, user_id)
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENGINE_URL = os.getenv("FLYTO_ENGINE_URL", "http://localhost:8080")


class TelegramApprovalNotifier:
    """Sends approval notifications with inline keyboard and handles decisions."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        engine_url: Optional[str] = None,
        firebase_token: Optional[str] = None,
    ) -> None:
        self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._engine_url = engine_url or ENGINE_URL
        self._firebase_token = firebase_token or os.getenv("FLYTO_SERVICE_TOKEN", "")

    async def send_approval(self, chat_id: str, approval: Dict[str, Any]) -> bool:
        """Send an approval request to Telegram with approve/reject buttons."""
        if not self._bot_token:
            logger.warning("Telegram: no bot token configured")
            return False

        approval_id = approval.get("id", "")
        title = approval.get("title", "待審批項目")
        approval_type = approval.get("type", "generic")
        requested_by = approval.get("requestedBy", "system")

        text = (
            f"📋 *新的審批請求*\n\n"
            f"*類型：* {approval_type}\n"
            f"*標題：* {title}\n"
            f"*請求者：* {requested_by}\n\n"
            f"請選擇操作："
        )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "✅ 核准", "callback_data": f"approve:{approval_id}"},
                    {"text": "❌ 拒絕", "callback_data": f"reject:{approval_id}"},
                ]
            ]
        }

        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "reply_markup": keyboard,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram approval send failed: %s %s", resp.status, body[:200])
                        return False
            return True
        except Exception as e:
            logger.warning("Telegram approval send error: %s", e)
            return False

    async def handle_callback(self, callback_data: str, user_id: str) -> Optional[str]:
        """Handle approve/reject callback from inline keyboard.

        callback_data format: "approve:<approval_id>" or "reject:<approval_id>"
        Returns a response message string or None on error.
        """
        parts = callback_data.split(":", 1)
        if len(parts) != 2 or parts[0] not in ("approve", "reject"):
            return None

        action, approval_id = parts
        endpoint = f"{self._engine_url}/api/v1/approvals/{approval_id}/{action}"

        try:
            import aiohttp

            headers = {
                "Content-Type": "application/json",
            }
            if self._firebase_token:
                headers["Authorization"] = f"Bearer {self._firebase_token}"

            body = {"note": f"已透過 Telegram 由使用者 {user_id} {action}"}

            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=body, headers=headers) as resp:
                    if resp.status == 200:
                        action_zh = "核准" if action == "approve" else "拒絕"
                        return f"✅ 已{action_zh}審批請求"
                    else:
                        resp_body = await resp.text()
                        logger.warning("Engine approval %s failed: %s %s", action, resp.status, resp_body[:200])
                        return f"❌ 操作失敗：{resp.status}"
        except Exception as e:
            logger.warning("Telegram approval callback error: %s", e)
            return f"❌ 錯誤：{e}"

    async def answer_callback_query(self, callback_query_id: str, text: str) -> bool:
        """Answer the callback query to remove the loading spinner."""
        if not self._bot_token:
            return False
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._bot_token}/answerCallbackQuery"
            payload = {"callback_query_id": callback_query_id, "text": text}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception:
            return False
