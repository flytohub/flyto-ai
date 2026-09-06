"""Local inference, with the original Agent retaining all action authority."""

from types import SimpleNamespace

from flyto_ai.cli_runtime import CliAgent
from .contracts import LocalModelConfig, LocalModelError, local_endpoint
from .http import complete_local_json

__all__ = ["LocalModelConfig", "LocalModelError", "LocalModelAgent", "complete_local_json", "local_endpoint"]


class LocalModelAgent(CliAgent):
    """Use local HTTP or trusted delegated inference, never both or a fallback."""

    def __init__(self, config, *, local=None, completion_fn=None, tool_executor=None,
                 system_prompt=None, policies=None, **agent_options):
        if local is not None and not isinstance(local, LocalModelConfig):
            raise TypeError("local must be a LocalModelConfig")
        if (local is None) == (completion_fn is None):
            raise ValueError("Select exactly one local configuration or delegated completion")
        native = local is not None
        if native:
            async def complete(**kwargs):
                return await complete_local_json(local, **kwargs)
            completion_fn = complete
        runtime = local or SimpleNamespace(source="local_ai", model=config.model or "", timeout_seconds=100.0)
        super().__init__(config, cli=runtime, completion_fn=completion_fn,
                         tool_executor=tool_executor, system_prompt=system_prompt,
                         policies=policies, **agent_options)
        self.local_runtime = self.cli_runtime
        if native:
            self.local_runtime.image_completion_fn = completion_fn
