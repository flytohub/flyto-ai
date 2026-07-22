# Configuration

Configuration can be provided directly with `AgentConfig`, as a dictionary,
through environment variables, or via layered YAML/JSON files. The generated
[environment reference](reference/environment.md) lists every static environment
read and source owner.

## Resolution

`ConfigFile` loads `~/.flyto/config.yaml`, optionally deep-merges agent/session
overrides, and can hot-reload through a daemon watcher. `AgentConfig.from_env()`
resolves provider-specific values and up to three fallback providers. Explicit
constructor values are preferable in libraries because they are easier to test.

## Provider And Limits

- select OpenAI, Anthropic, Ollama, DeepSeek/OpenAI-compatible endpoints, or an
  ordered fallback chain;
- configure model, temperature, max tokens, tool/validation rounds, and optional
  session/global budgets;
- custom base URLs are cleared when URL policy rejects them;
- the runtime clamps unsafe temperature/token limits.

## Storage

Memory SQLite, transcripts, evidence, eval results, cache, and vault files belong
under operator-controlled paths outside the repository. Transcript and telemetry
content can contain user/tool data even after redaction; apply retention and file
permissions appropriate to the deployment.

## Safety Defaults

- permission defaults to `workspace_write`;
- prompt injection detection, transcript, memory, EMS, knowledge, contract
  validation, and deterministic planning default on;
- Docker sandbox and prompt evolution default off;
- HTTP server defaults to loopback and should require a key outside local use;
- production-target security workflow generation requires explicit operator
  override and authorization.

## Secrets

Provider keys, bot tokens, webhook credentials, service tokens, server keys,
vault passphrases, and channel allowlists are secrets. Keep populated `.env`,
config files, transcript/evidence exports, and vault files out of Git. Use GitHub
Environment/Actions secrets for CI and an external secret manager in deployments.

