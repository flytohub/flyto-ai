# Operations And Verification

## Installation Profiles

- default: providers, Core browser modules, Pro contracts, blueprint storage,
  memory, rich CLI, and cryptography;
- `serve`: adds `aiohttp` HTTP/SSE serving;
- `agent`: adds Claude Agent SDK coding mode;
- `dev`: adds pytest, asyncio test support, and Ruff.

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

