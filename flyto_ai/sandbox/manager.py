# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Sandbox manager — routes dangerous modules to Docker containers."""
import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module categories that run directly in main process (safe)
DIRECT_CATEGORIES = {"string", "math", "json", "array", "datetime", "crypto", "regex"}

# Module categories that require sandbox isolation
SANDBOXED_CATEGORIES = {"browser", "shell", "file", "http", "database", "notification"}


class SandboxManager:
    """Execute flyto-core modules inside Docker containers for isolation."""

    def __init__(
        self,
        image: str = "flyto-sandbox:latest",
        timeout: int = 60,
    ) -> None:
        self._image = image
        self._timeout = timeout

    def needs_sandbox(self, module_id: str) -> bool:
        """Check if a module should run in a sandbox."""
        category = module_id.split(".")[0] if "." in module_id else module_id
        return category in SANDBOXED_CATEGORIES

    async def execute(
        self,
        module_id: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a module inside a Docker container.

        Protocol: stdin → JSON payload, stdout → JSON result.
        """
        payload = json.dumps({
            "module_id": module_id,
            "params": params,
            "context": context,
        })

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "-i",
                "--network=none",
                "--memory=512m",
                "--cpus=1",
                self._image,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload.encode()),
                timeout=self._timeout,
            )

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else "Container exited with code {}".format(proc.returncode)
                return {"ok": False, "error": "Sandbox error: {}".format(err[:500])}

            result = json.loads(stdout.decode())
            return result

        except asyncio.TimeoutError:
            logger.warning("Sandbox timeout for %s after %ds", module_id, self._timeout)
            # Try to kill the container
            try:
                proc.kill()
            except Exception:
                pass
            return {"ok": False, "error": "Sandbox timeout ({}s) for {}".format(self._timeout, module_id)}

        except json.JSONDecodeError as e:
            return {"ok": False, "error": "Sandbox returned invalid JSON: {}".format(e)}

        except FileNotFoundError:
            return {"ok": False, "error": "Docker not found. Install Docker to use sandbox mode."}

        except Exception as e:
            return {"ok": False, "error": "Sandbox error: {}".format(e)}
