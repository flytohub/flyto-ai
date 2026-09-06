# Flyto2 AI Documentation

Use this index to move from product behavior to exact implementation contracts.

## Start Here

- [Local CLI inference runtime](local-cli-runtime.md): official Codex/Claude sign-in, host-only actions and reasoning delegation.
- [Computer-local model inference](local-model-runtime.md): explicit loopback Ollama/compatible models, actual images and cancellable host-controlled inference.

- [Architecture map](architecture-map.md): the canonical Flytohub product
  topology (ownership and integration), then this repository's internals.
- [Feature and package map](FEATURES.md): shipped behavior and source ownership.
- [Python API guide](API.md): stable package-level integration contracts.
- [CLI, HTTP, and MCP](CLI_AND_MCP.md): operator and protocol entrypoints.
- [Configuration](CONFIGURATION.md): resolution, storage, safety, and secrets.
- [Operations](OPERATIONS.md): installation, verification, release, and incident handling.
- [Technical whitepaper](WHITEPAPER.md): architecture, trust model, determinism, and evidence.

## Generated Reference

- [Reference index](reference/README.md): generated inventory overview.
- [Python symbol index](reference/python/README.md): every top-level class/function and direct class method, including internal helpers.
- [CLI reference](reference/cli.md): parser-declared commands and options.
- [Tool and MCP reference](reference/tools-and-mcp.md): static tool/protocol definitions and ownership.
- [Environment reference](reference/environment.md): every statically named runtime environment read.
- [Maintainer scripts](reference/scripts.md): operational script inventory and side effects.

Regenerate with `python3 scripts/generate_reference.py`; CI rejects stale output.

## Design And Integration

- [Provider-neutral coding control plane](CODING_CONTROL_PLANE.md): the
  audit-required `code-mcp` / `code-serve` route, startup `native|claude`
  selection, the audit state machine, and a project-scoped Codex MCP example.
- [AI/MCP closed loop](AI_MCP_CLOSED_LOOP.md)
- [Adaptive security campaigns](API.md#adaptive-security-campaigns)
- [Deterministic capability routing](CAPABILITY_ROUTING.md)
- [Structured Robotics planner](API.md#structured-robotics-planner)
- [Judge-card Mission interpretation](API.md#mission-station-card-interpretation)
- [Model compatibility](MODEL_COMPATIBILITY.md)
- [Architecture map](architecture-map.md)
- [Demo asset](demo.svg)

## Project Memory

- [Project](../PROJECT.md)
- [Architecture](../ARCHITECTURE.md)
- [Current state](../STATE.md)
- [Roadmap](../ROADMAP.md)
- [Decisions](../DECISIONS.md)
- [Task history](../tasks.md)
- [Changelog](../CHANGELOG.md)
- [Security policy](../SECURITY.md)
- [Contributing](../CONTRIBUTING.md)

Public surfaces must follow the Flyto2 Frontend Quality Gate in [AGENTS.md](../AGENTS.md).
