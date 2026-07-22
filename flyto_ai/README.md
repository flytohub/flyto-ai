# Flyto2 AI Package

`flyto_ai` contains the provider-neutral agent runtime, tool and MCP bridges,
memory and evaluation systems, safety controls, channels, schedulers, and
operator entrypoints shipped by `flyto-ai`.

Use these references when changing the package:

- [Architecture](../ARCHITECTURE.md) explains package boundaries and runtime flow.
- [Feature map](../docs/FEATURES.md) maps product behavior to source and tests.
- [Python symbol index](../docs/reference/python/README.md) lists every class,
  function, and method with an exact source link.
- [CLI and MCP](../docs/CLI_AND_MCP.md) defines executable and protocol contracts.
- [Configuration](../docs/CONFIGURATION.md) owns environment and secret handling.
- [Security](../SECURITY.md) defines trust boundaries and reporting policy.

The generated references are authoritative for declaration inventory. Run
`python3 scripts/generate_reference.py` from the repository root after changing
package declarations; CI uses `--check` to reject drift.
