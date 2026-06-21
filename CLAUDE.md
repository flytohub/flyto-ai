# Claude Guidance

Use Claude Code as an implementation worker, not as a source of credentials or hidden state.

- Prefer MCP tools for project context and `flyto-core` capability discovery.
- Require approval before file writes, shell execution, browser actions with live accounts, or any operation that can mutate external state.
- Record tool use as evidence with run id, tool name, module id or recipe name, validation result, and outcome.
- Do not ask users for secrets through MCP elicitation. Secrets must arrive through the active runtime channel only.
- Subagents may investigate or verify, but changes must preserve the public `flyto-ai` provider/tool contracts.
