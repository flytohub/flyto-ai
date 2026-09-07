"""Read official model metadata without starting a model turn or a thread."""

import tempfile
from dataclasses import replace

from .codex import CodexProtocol, codex_argv
from .contracts import CliRuntimeError, valid_model_id
from .process import ProcessRunner


class ModelCatalogProtocol(CodexProtocol):
    """The official model/list RPC is metadata, never an inference request."""

    def __init__(self, cli, cwd):
        super().__init__(cli, cwd)
        self.rows = []
        self.cursors = set()

    async def _response(self, identity, result):
        if not isinstance(result, dict):
            raise CliRuntimeError("cli_invalid_catalog")
        if identity == 1:
            await super()._response(identity, result)
        elif identity == 2:
            self._check_config(result.get("config") or {})
            await self._send(6, "model/list", {"limit": 100, "includeHidden": False})
        else:
            rows = result.get("data")
            if not isinstance(rows, list) or len(self.rows) + len(rows) > 200:
                raise CliRuntimeError("cli_invalid_catalog")
            for row in rows:
                if not isinstance(row, dict) or not valid_model_id(row.get("model"), allow_empty=False):
                    raise CliRuntimeError("cli_invalid_catalog")
                if row.get("hidden") is True:
                    continue
                label = row.get("displayName")
                description = row.get("description", "")
                if not isinstance(label, str) or len(label) > 256 or not isinstance(description, str) or len(description) > 2000:
                    raise CliRuntimeError("cli_invalid_catalog")
                item = {"id": row["model"], "label": label, "description": description,
                        "is_default": row.get("isDefault") is True}
                modalities = row.get("inputModalities")
                if isinstance(modalities, list) and all(value in {"text", "image"} for value in modalities):
                    item["input_modalities"] = modalities
                self.rows.append(item)
            cursor = result.get("nextCursor")
            if cursor is not None:
                if not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096 or cursor in self.cursors or len(self.cursors) >= 4:
                    raise CliRuntimeError("cli_invalid_catalog")
                self.cursors.add(cursor)
                await self._send(identity + 1, "model/list", {"limit": 100, "includeHidden": False, "cursor": cursor})
            else:
                self.content = list({row["id"]: row for row in self.rows}.values())
                self._finish()

    def _notification(self, method, params):
        if method and (method.startswith(("item/", "turn/", "hook/"))):
            raise CliRuntimeError("cli_native_action_refused")
        super()._notification(method, params)


async def discover_cli_models(cli):
    """Return the selected CLI's actual catalog, or a manual-entry fallback."""
    result = {"source": cli.source, "models": [], "manual_entry": True,
              "catalog_available": False, "reason_code": "manual_catalog_unavailable"}
    if cli.source == "claude_cli":
        return result  # No official noninteractive catalog protocol is exposed.
    runner = ProcessRunner(replace(cli, model=""))
    try:
        with tempfile.TemporaryDirectory(prefix="flyto-cli-models-") as cwd:
            protocol = ModelCatalogProtocol(runner.cli, cwd)
            code, _output, _errors = await runner._run(
                codex_argv(), cwd=cwd, timeout=cli.timeout_seconds, protocol=protocol,
            )
            if code not in (0, None) and not protocol.completed:
                raise CliRuntimeError("cli_catalog_unavailable")
            result.update(models=protocol.result()[0], catalog_available=True, reason_code="")
    except CliRuntimeError as exc:
        result["reason_code"] = exc.code
    except OSError:
        result["reason_code"] = "cli_process_unavailable"
    finally:
        await runner.close()
    return result
