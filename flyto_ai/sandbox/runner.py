#!/usr/bin/env python3
# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Container entry point — reads JSON from stdin, executes module, writes JSON to stdout."""
import asyncio
import json
import sys


async def run():
    payload = json.loads(sys.stdin.read())
    module_id = payload.get("module_id", "")
    params = payload.get("params", {})
    context = payload.get("context")

    try:
        from core.mcp_handler import execute_module
        result = await execute_module(
            module_id=module_id,
            params=params,
            context=context,
            browser_sessions={},
        )
        print(json.dumps(result, ensure_ascii=False, default=str))
    except ImportError:
        print(json.dumps({"ok": False, "error": "flyto-core not installed in container"}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    asyncio.run(run())
