# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Telegram message sender — enhanced send with inline keyboard, edit, delete."""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096


class TelegramSender:
    """Enhanced Telegram Bot API sender.

    Supports plain messages, inline keyboards, message editing/deletion,
    and callback query acknowledgement. Handles Markdown retry automatically.
    """

    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token

    async def _request(self, method: str, payload: Dict[str, Any]) -> Optional[Dict]:
        """Send a request to the Telegram Bot API. Returns response JSON or None."""
        import aiohttp

        url = "https://api.telegram.org/bot{}/{}".format(self._bot_token, method)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    body = await resp.text()
                    logger.warning("TG API %s failed: %s %s", method, resp.status, body[:200])
                    return None
        except Exception as e:
            logger.warning("TG API %s error: %s", method, e)
            return None

    async def send(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "Markdown",
    ) -> Optional[int]:
        """Send a text message. Returns message_id on success, None on failure.

        Falls back to plain text if Markdown parsing fails.
        """
        if len(text) > _MAX_MESSAGE_LENGTH:
            text = text[:_MAX_MESSAGE_LENGTH - 20] + "\n\n... (truncated)"

        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        result = await self._request("sendMessage", payload)

        # Retry without parse_mode if Markdown caused a failure
        if result is None and parse_mode:
            payload.pop("parse_mode", None)
            result = await self._request("sendMessage", payload)

        if result and result.get("ok"):
            return result["result"]["message_id"]
        return None

    async def send_with_keyboard(
        self,
        chat_id: int,
        text: str,
        buttons: List[List[Dict[str, str]]],
        parse_mode: str = "Markdown",
    ) -> Optional[int]:
        """Send a message with an inline keyboard.

        Parameters
        ----------
        buttons : list of list of dict
            Each inner list is a row. Each dict has ``text`` and ``callback_data`` keys.
            Example: ``[[{"text": "Yes", "callback_data": "approve:123"}]]``
        """
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": {"inline_keyboard": buttons},
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        result = await self._request("sendMessage", payload)

        if result is None and parse_mode:
            payload.pop("parse_mode", None)
            result = await self._request("sendMessage", payload)

        if result and result.get("ok"):
            return result["result"]["message_id"]
        return None

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str = "Markdown",
    ) -> bool:
        """Edit an existing message. Returns True on success."""
        if len(text) > _MAX_MESSAGE_LENGTH:
            text = text[:_MAX_MESSAGE_LENGTH - 20] + "\n\n... (truncated)"

        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        result = await self._request("editMessageText", payload)

        if result is None and parse_mode:
            payload.pop("parse_mode", None)
            result = await self._request("editMessageText", payload)

        return result is not None and result.get("ok", False)

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Delete a message. Returns True on success."""
        result = await self._request("deleteMessage", {
            "chat_id": chat_id,
            "message_id": message_id,
        })
        return result is not None and result.get("ok", False)

    async def answer_callback(
        self,
        callback_query_id: str,
        text: str = "",
    ) -> None:
        """Acknowledge a callback query (removes the loading indicator)."""
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        await self._request("answerCallbackQuery", payload)

    async def download_file(self, file_id: str, dest_path: str) -> bool:
        """Download a Telegram file to a local path. Returns True on success."""
        import aiohttp

        result = await self._request("getFile", {"file_id": file_id})
        if not result or not result.get("ok"):
            logger.warning("TG getFile failed for %s", file_id)
            return False

        tg_path = result["result"]["file_path"]
        url = "https://api.telegram.org/file/bot{}/{}".format(self._bot_token, tg_path)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning("TG file download failed: %s", resp.status)
                        return False
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
            return True
        except Exception as e:
            logger.warning("TG file download error: %s", e)
            return False

    async def send_long(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "Markdown",
    ) -> List[int]:
        """Send a long message in chunks. Returns list of message_ids."""
        if len(text) <= 4000:
            msg_id = await self.send(chat_id, text, parse_mode)
            return [msg_id] if msg_id else []

        msg_ids: List[int] = []
        for i in range(0, len(text), 4000):
            chunk = text[i:i + 4000]
            msg_id = await self.send(chat_id, chunk, parse_mode)
            if msg_id:
                msg_ids.append(msg_id)
        return msg_ids

    async def send_document(
        self,
        chat_id: int,
        file_path: str,
        caption: str = "",
    ) -> bool:
        """Send a file as a Telegram document. Returns True on success."""
        import aiohttp

        url = "https://api.telegram.org/bot{}/sendDocument".format(self._bot_token)
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        if caption:
            data.add_field("caption", caption[:1024])
        data.add_field(
            "document",
            open(file_path, "rb"),
            filename=os.path.basename(file_path),
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.warning("TG sendDocument failed: %s %s", resp.status, body[:200])
                    return False
        except Exception as e:
            logger.warning("TG sendDocument error: %s", e)
            return False
