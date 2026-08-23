# Operations And Verification

## Installation Profiles

This list describes what `pip install` actually resolves. It stopped doing so at
0.16.0, when Core, Pro, Blueprint and the Anthropic provider moved out of the
base install and into extras, and was not updated -- an operator reading it
would have expected browser automation and Blueprint storage in a default
install and found neither.

- default: OpenAI provider, language detection, YAML/pydantic, `aiosqlite`
  memory, rich CLI, and cryptography. No Core, no Blueprint, no Pro, no
  Anthropic;
- `browser`: adds `flyto-core[browser]`, which is what makes Core modules and
  browser automation available at all;
- `blueprint`: adds `flyto-blueprint` procedure storage;
- `pro`: adds `flyto-pro-core` contracts;
- `anthropic`: adds the Anthropic provider;
- `full`: all four of the above together, reproducing the pre-0.16.0 base;
- `lite`: YAML and pydantic only;
- `serve`: adds `aiohttp` HTTP/SSE serving;
- `agent` / `claude-sdk`: adds the Claude Agent SDK coding backend;
- `dev`: the `full` set plus pytest, asyncio test support, and Ruff.

Core browser workflows additionally require the corresponding browser runtime.
Ollama requires a separately running local service and selected model.

## Local Verification

```bash
python -m compileall -q flyto_ai
python -m ruff check flyto_ai tests
python -m pytest -q
python3 scripts/generate_reference.py --check
python3 ../.github/scripts/audit-documentation.py . --strict
python -m build
flyto-index verify . --full-scan --query agent --json
```

Provider, browser, Telegram, webhook, Cloud, and production security tests need
separate credentials/services and must be explicitly selected. The default test
suite must not silently make billable network calls.

## Release

1. synchronize `pyproject.toml`, installed package metadata, changelog, and docs;
2. regenerate references and run the full offline suite;
3. build wheel/sdist and inspect them with `twine check`;
4. tag the verified commit; trusted PyPI publishing uses OIDC, not a stored token;
5. separately publish/scan the sandbox image when a sandbox tag is intended.

## Troubleshooting

- inspect `flyto-ai version` for package/provider/Core availability;
- use the capability manifest for current tool fingerprint and module categories;
- distinguish provider failure from schema rejection, permission denial, sandbox
  failure, budget exhaustion, or MCP transport failure;
- use transcript/evidence IDs for replay while keeping secret material redacted;
- repeated browser session failures should pass through the Core bridge retry
  path rather than ad hoc provider retries.

