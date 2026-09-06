"""Bounded official CLI subprocesses with no inherited action authority."""

import asyncio
import base64
import inspect
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from .codex import CodexProtocol, codex_argv
from .contracts import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    MAX_OUTPUT_BYTES,
    CliRuntimeConfig,
    CliRuntimeError,
    cli_environment,
    encode_json,
)
from .events import EventReader, failure_code

_CLAUDE_FLAGS = (
    "--tools", "--strict-mcp-config", "--restricted", "--safe-mode",
    "--no-session-persistence", "--json-schema", "--input-format",
    "--disable-slash-commands", "--setting-sources", "--no-chrome",
)


def resolve_cli_executable(source: str, command: str | None = None) -> str | None:
    """Resolve trusted host selection; never search arbitrary app directories."""
    if source not in {"codex_cli", "claude_cli"}:
        raise ValueError("Unsupported local CLI source")
    selected = shutil.which(command or ("codex" if source == "codex_cli" else "claude"))
    if selected or command is not None:
        return selected
    if source == "codex_cli" and platform.system() == "Darwin":
        bundle = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        if bundle.is_file() and os.access(bundle, os.X_OK):
            return str(bundle)
    return None


def required_cli_flags(source: str) -> tuple[str, ...]:
    if source == "claude_cli":
        return _CLAUDE_FLAGS
    if source == "codex_cli":
        return ("--stdio", "--strict-config", "--listen")
    raise ValueError("Unsupported local CLI source")


async def _signal_group(pid, action):
    try:
        os.killpg(pid, action)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can retain dead orphan groups until init reaps their zombies.
        # The same check is needed on the first signal, not only the final one.
        probe = await asyncio.create_subprocess_exec(
            "ps", "-axo", "pgid=,stat=", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await asyncio.wait_for(probe.communicate(), 2)
        if probe.returncode or any(
            len(parts := line.split()) == 2 and parts[0] == str(pid).encode()
            and not parts[1].startswith(b"Z") for line in output.splitlines()
        ):
            raise CliRuntimeError("cli_cleanup_failed")


async def _stop(process):
    # A CLI may exit while its children remain alive. The process group belongs
    # to this one invocation, so terminal success must release it as well.
    if os.name != "nt":
        await _signal_group(process.pid, signal.SIGKILL if process.returncode is not None else signal.SIGTERM)
    elif process.returncode is None:
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(killer.wait(), 3)
    try:
        await asyncio.wait_for(process.wait(), 0.75)
    except TimeoutError:
        if os.name != "nt":
            await _signal_group(process.pid, signal.SIGKILL)
        else:
            process.kill()
        await asyncio.wait_for(process.wait(), 2)
    if os.name != "nt":
        await _signal_group(process.pid, signal.SIGKILL)


async def _finish_cleanup(process):
    """A second cancellation cannot abandon a child after its first cancel."""
    cleanup = asyncio.create_task(_stop(process))
    interrupted = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            interrupted = True
    cleanup.result()
    if interrupted:
        raise asyncio.CancelledError


async def _read(stream, *, consume=None):
    total = 0
    chunks = []
    while True:
        try:
            part = await (stream.readline() if consume else stream.read(65536))
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise CliRuntimeError("cli_output_too_large") from exc
        if not part:
            return b"".join(chunks)
        total += len(part)
        if total > MAX_OUTPUT_BYTES:
            raise CliRuntimeError("cli_output_too_large")
        if consume:
            if part.strip():
                consumed = consume(part)
                if inspect.isawaitable(consumed):
                    await consumed
        else:
            chunks.append(part)


class ProcessRunner:
    """One selected executable; a closed runner can never launch again."""

    def __init__(self, cli: CliRuntimeConfig):
        self.cli = cli
        self.executable = resolve_cli_executable(cli.source, cli.command)
        self._processes = set()
        self._closed = False
        self._supported = False
        self.last_model = ""

    async def _run(self, argv, *, cwd, stdin=b"", timeout=5, reader=None, protocol=None):
        if self._closed:
            raise CliRuntimeError("cli_closed")
        if not self.executable:
            raise CliRuntimeError("cli_not_found")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = await asyncio.create_subprocess_exec(
            self.executable, *argv, cwd=cwd, env=cli_environment(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=512_001, **options,
        )
        self._processes.add(process)
        if protocol:
            protocol.writer = process.stdin
            stdin, reader = protocol.initial, protocol.read
        readers = []
        try:
            if self._closed:
                raise CliRuntimeError("cli_closed")
            async def send():
                try:
                    process.stdin.write(stdin)
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if protocol is None:
                        process.stdin.close()
            readers = [asyncio.create_task(_read(process.stdout, consume=reader)),
                       asyncio.create_task(_read(process.stderr)), asyncio.create_task(send()),
                       asyncio.create_task(process.wait())]
            output, errors, _, code = await asyncio.wait_for(asyncio.gather(*readers), timeout)
            return code, output, errors
        except TimeoutError as exc:
            raise CliRuntimeError("cli_timeout") from exc
        finally:
            for pending in readers:
                if not pending.done():
                    pending.cancel()
            try:
                await _finish_cleanup(process)
            finally:
                self._processes.discard(process)
                await asyncio.gather(*readers, return_exceptions=True)

    async def inspect(self):
        result = {"source": self.cli.source, "installed": bool(self.executable), "version": "", "supported": False, "reason_code": "cli_not_found"}
        if not self.executable:
            return result
        if os.name == "nt":
            result['reason_code'] = 'cli_process_isolation_unavailable'
            return result  # Windows requires a job-object implementation first.
        with tempfile.TemporaryDirectory(prefix="flyto-cli-probe-") as cwd:
            try:
                code, output, _ = await self._run(["--version"], cwd=cwd)
                match = re.search(rb"\b\d+\.\d+\.\d+(?:[-.][A-Za-z0-9]+)*\b", output[:512])
                if code or not match:
                    raise CliRuntimeError("cli_probe_failed")
                result["version"] = match.group().decode()
                args = ["app-server", "--help"] if self.cli.source == "codex_cli" else ["--help"]
                code, output, _ = await self._run(args, cwd=cwd)
                if code or any(flag.encode() not in output for flag in required_cli_flags(self.cli.source)):
                    raise CliRuntimeError("cli_required_flags_unavailable")
                if self.cli.source == "codex_cli":
                    if tuple(map(int, match.group().decode().split('.')[:3])) < (0, 153, 4):
                        raise CliRuntimeError("cli_required_flags_unavailable")
                    protocol = CodexProtocol(self.cli, cwd)
                    await self._run(codex_argv(), cwd=cwd, protocol=protocol, timeout=10)
                    protocol.result()
                self._supported = True
                result.update(supported=True, reason_code="ready")
            except (OSError, CliRuntimeError) as exc:
                result["reason_code"] = exc.code if isinstance(exc, CliRuntimeError) else "cli_probe_failed"
        return result

    async def infer(self, prompt, schema, images=()):
        if self._closed:
            raise CliRuntimeError("cli_closed")
        if not self._supported:
            status = await self.inspect()
            if not status["supported"]:
                raise CliRuntimeError(status["reason_code"])
        encode_json(schema, limit=128_000)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise CliRuntimeError("cli_invalid_schema")
        with tempfile.TemporaryDirectory(prefix="flyto-cli-inference-") as cwd:
            self._check_images(images)
            if self.cli.source == "codex_cli":
                protocol = CodexProtocol(self.cli, cwd, prompt, schema, images)
                await self._run(codex_argv(), cwd=cwd, protocol=protocol,
                                timeout=self.cli.timeout_seconds)
                self.last_model = protocol.model
                return protocol.result()
            argv, stdin = self._request(prompt, schema, images, Path(cwd))
            reader = EventReader(self.cli.source)
            code, _, errors = await self._run(
                argv, cwd=cwd, stdin=stdin, timeout=self.cli.timeout_seconds, reader=reader.read,
            )
            if code:
                raise CliRuntimeError(failure_code(errors.decode(errors="replace")))
            self.last_model = reader.model
            return reader.result()

    @staticmethod
    def _check_images(images):
        if len(images) > MAX_IMAGES:
            raise CliRuntimeError("cli_image_limit")
        for image in images:
            media = image.get("media_type")
            if media not in {"image/png", "image/jpeg", "image/webp"}:
                raise CliRuntimeError("cli_invalid_image")
            try:
                data = base64.b64decode(image["base64"], validate=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise CliRuntimeError("cli_invalid_image") from exc
            if not data or len(data) > MAX_IMAGE_BYTES:
                raise CliRuntimeError("cli_image_limit")

    def _request(self, prompt, schema, images, cwd):
        content = [{"type": "text", "text": prompt}]
        for image in images:
            media = image['media_type']
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": image["base64"]}})
        payload = {"type": "user", "message": {"role": "user", "content": content}}
        # No prompt, model reply, picture or sign-in material is written to disk.
        stdin = (encode_json(payload, limit=MAX_IMAGES * MAX_IMAGE_BYTES * 2 + 1_000_000) + "\n").encode()
        argv = ["--print", "--output-format", "stream-json", "--input-format", "stream-json",
                "--verbose", "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--safe-mode", "--restricted", "--setting-sources", "", "--no-session-persistence",
                "--disable-slash-commands", "--no-chrome", "--permission-mode", "dontAsk",
                "--json-schema", encode_json(schema, limit=128_000)]
        if self.cli.model:
            argv.extend(("--model", self.cli.model))
        return argv, stdin

    async def close(self):
        self._closed = True
        for process in tuple(self._processes):
            await _finish_cleanup(process)


async def inspect_cli_runtime(cli: CliRuntimeConfig) -> dict:
    runner = ProcessRunner(cli)
    try:
        return await runner.inspect()
    finally:
        await runner.close()
