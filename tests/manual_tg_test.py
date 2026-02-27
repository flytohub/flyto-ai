#!/usr/bin/env python3
"""Manual integration test: starts a fake Telegram API + flyto-ai server,
sends curl-like requests and prints the actual replies that would go to TG."""
import asyncio
import json
import sys
from aiohttp import web

# --- Collect replies sent to "Telegram" ---
replies = []

async def fake_tg_api(request):
    """Pretend to be api.telegram.org/bot.../sendMessage"""
    body = await request.json()
    text = body.get("text", "")
    chat_id = body.get("chat_id")
    replies.append({"chat_id": chat_id, "text": text})
    print("  [TG reply] -> chat {} : {}".format(chat_id, text[:120]))
    return web.json_response({"ok": True})


async def main():
    # 1) Start fake TG API on port 19443
    tg_app = web.Application()
    tg_app.router.add_post("/bottest-token/sendMessage", fake_tg_api)
    tg_runner = web.AppRunner(tg_app)
    await tg_runner.setup()
    tg_site = web.TCPSite(tg_runner, "127.0.0.1", 19443)
    await tg_site.start()
    print("Fake TG API listening on http://127.0.0.1:19443\n")

    # 2) Monkey-patch _tg_send to use our fake server
    import flyto_ai.cli as cli
    import aiohttp as _aio

    original_send = cli._tg_send

    async def patched_tg_send(token, chat_id, text):
        url = "http://127.0.0.1:19443/bot{}/sendMessage".format(token)
        payload = {"chat_id": chat_id, "text": text}
        async with _aio.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                await resp.json()

    cli._tg_send = patched_tg_send
    cli._TG_TOKEN = "test-token"
    cli._TG_ALLOWED_CHATS = frozenset({5608426436})

    # 3) Build the flyto-ai app (capture from _cmd_serve_aiohttp)
    captured = {}
    original_run = web.run_app

    def fake_run(app, **kw):
        captured["app"] = app

    web.run_app = fake_run

    import argparse
    args = argparse.Namespace(
        host="127.0.0.1", port=0,
        provider=None, model=None, api_key=None,
        dir="/Library/其他專案/flytohub",
    )
    cli._cmd_serve_aiohttp(args)
    web.run_app = original_run

    app = captured["app"]
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 7422)
    await site.start()
    print("flyto-ai server listening on http://127.0.0.1:7422\n")

    # 4) Send test commands
    test_cases = [
        ("/help", "Should show command list"),
        ("/yaml", "Should list blueprints"),
        ("/blueprint", "Same as /yaml"),
        ("/claude", "Should show usage"),
        ("what is 2+2", "Should go to agent.chat"),
    ]

    async with _aio.ClientSession() as session:
        for text, desc in test_cases:
            replies.clear()
            print("--- {} ---  ({})".format(text, desc))

            payload = {"message": {"chat": {"id": 5608426436}, "text": text}}
            async with session.post("http://127.0.0.1:7422/telegram", json=payload) as resp:
                status = resp.status
                body = await resp.json()
                print("  [webhook] status={} body={}".format(status, body))

            # Wait for background task to send replies
            await asyncio.sleep(2)

            if not replies:
                print("  [TG reply] (none)")
            print()

    # 5) Test unauthorized
    print("--- /help from unauthorized chat ---")
    replies.clear()
    async with _aio.ClientSession() as session:
        payload = {"message": {"chat": {"id": 999}, "text": "/help"}}
        async with session.post("http://127.0.0.1:7422/telegram", json=payload) as resp:
            print("  [webhook] status={} body={}".format(resp.status, await resp.json()))
    await asyncio.sleep(1)
    if not replies:
        print("  [TG reply] (none — correctly blocked)")
    print()

    # Cleanup
    await runner.cleanup()
    await tg_runner.cleanup()
    print("All tests done.")


if __name__ == "__main__":
    asyncio.run(main())
