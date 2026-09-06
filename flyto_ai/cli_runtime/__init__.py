"""Opt-in, local CLI inference behind the existing host Agent authority."""

from copy import deepcopy

from flyto_ai.agent import Agent
from flyto_ai.models import UsageStats

from .contracts import CliRuntimeConfig, CliRuntimeError, cli_environment, encode_json
from .process import ProcessRunner, inspect_cli_runtime, required_cli_flags
from .transport import CliTransport

__all__ = [
    "CliAgent",
    "CliRuntimeConfig",
    "CliRuntimeError",
    "cli_environment",
    "complete_json",
    "inspect_cli_runtime",
    "required_cli_flags",
]


class CliAgent(Agent):
    """Same admission, continuation and tool dispatch; selected CLI inference."""

    def __init__(self, config, *, cli: CliRuntimeConfig, tool_executor=None,
                 system_prompt=None, policies=None, completion_fn=None, **agent_options):
        if set(agent_options) - {'tools', 'dispatch_fn'}:
            raise TypeError('Unknown CliAgent option')
        tools, dispatch_fn = agent_options.get('tools'), agent_options.get('dispatch_fn')
        local = deepcopy(config)
        local.provider, local.model = cli.source, cli.model
        local.api_key, local.base_url = "", None
        local.fallback_providers = []
        local.enable_model_routing = False
        local.enable_deterministic = False
        # No embedding, evolution, vault injection or secondary model provider
        # may silently defeat the operator's explicit local sign-in selection.
        local.enable_memory = local.enable_pro = local.enable_ems = False
        local.enable_knowledge = local.enable_evolution = False
        local.vault_auto_inject = False
        local.vault_path = ""
        self.cli_runtime = CliTransport(cli, completion_fn=completion_fn)
        if tool_executor is None and not dispatch_fn:
            async def no_tools(_name, _arguments):
                return {"ok": False, "error": "This inference session has no tools."}
            dispatch_fn = no_tools
        super().__init__(local, tools=tools, dispatch_fn=dispatch_fn,
                         tool_executor=tool_executor, system_prompt=system_prompt,
                         policies=policies, api_client=self.cli_runtime)

    def _has_inference_credentials(self):
        return True  # Official CLI owns authentication; no key is fabricated.

    async def chat(self, *args, **kwargs):
        self.cli_runtime.reset()
        result = await super().chat(*args, **kwargs)
        runner = self.cli_runtime.runner
        updates = {"provider": self.cli_runtime.cli.source,
                   "model": (runner.last_model if runner else '') or self.cli_runtime.cli.model or "cli-default"}
        if self.cli_runtime.last_error:
            code = self.cli_runtime.last_error
            observed = self.cli_runtime.tool_calls
            updates.update(ok=False, error=code, message="Local CLI inference stopped: " + code,
                           tool_calls=observed,
                           execution_results=[item for item in observed if item.get("function") == "execute_module"],
                           rounds_used=self.cli_runtime.rounds,
                           usage=UsageStats(**self.cli_runtime.usage))
        return result.model_copy(update=updates)

    async def continue_execution(self, *args, **kwargs):
        self.cli_runtime.continuation = True
        try:
            return await super().continue_execution(*args, **kwargs)
        finally:
            self.cli_runtime.continuation = False

    async def close(self):
        try:
            await self.cli_runtime.close()
        finally:
            await super().close()


async def complete_json(cli: CliRuntimeConfig, *, prompt: str, schema: dict,
                        system_prompt: str = "") -> str:
    """One fresh, tool-free structured review/plan; host validates its meaning."""
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 256 * 1024:
        raise CliRuntimeError("cli_invalid_input")
    if not isinstance(system_prompt, str) or len(system_prompt.encode()) > 16 * 1024:
        raise CliRuntimeError("cli_invalid_input")
    encode_json(schema, limit=32 * 1024)
    runner = ProcessRunner(cli)
    try:
        request = encode_json({"system_prompt": system_prompt, "messages": [{"role": "user", "content": prompt}], "tools": []})
        value, _usage = await runner.infer(request, schema)
        if not isinstance(value, dict):
            raise CliRuntimeError("cli_invalid_output")
        return encode_json(value, limit=64 * 1024)
    finally:
        await runner.close()
