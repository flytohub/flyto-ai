<p align="center">
  <img src="https://raw.githubusercontent.com/flytohub/flyto-ai/main/docs/logo.svg" alt="Flyto2 AI" width="400">
</p>

<h3 align="center">Natural language → executable automation workflows</h3>

<p align="center">
  <em>aider writes code. open-interpreter runs code. <strong>flyto-ai builds workflows.</strong></em>
</p>

<p align="center">
  <a href="https://pypi.org/project/flyto-ai/"><img src="https://img.shields.io/pypi/v/flyto-ai?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/flyto-ai/"><img src="https://img.shields.io/pypi/pyversions/flyto-ai" alt="Python"></a>
  <a href="https://github.com/flytohub/flyto-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

---

## What is flyto-ai?

An AI agent that turns natural language into **reusable, structured automation workflows** — not throwaway scripts, not chat responses.

You describe what you want. The agent calls tools, validates parameters, and produces a **YAML workflow** you can save, share, schedule, and run again.

```
"scrape the title from example.com"
         ↓
┌─────────────────────────────────────┐
│ name: Scrape Title                  │
│ steps:                              │
│   - id: launch                      │
│     module: browser.launch          │
│   - id: goto                        │
│     module: browser.goto            │
│     params:                         │
│       url: "https://example.com"    │
│   - id: extract                     │
│     module: browser.extract         │
│     params:                         │
│       selector: "h1"               │
└─────────────────────────────────────┘
```

## Quick Start

```bash
pip install flyto-ai
export OPENAI_API_KEY=sk-...
flyto-ai chat "scrape the title from https://example.com"
```

That's it. One install — includes OpenAI, Anthropic, Ollama, **412 automation modules**, browser automation, and self-learning blueprints.

## Why flyto-ai?

| | aider | open-interpreter | flyto-ai |
|---|---|---|---|
| **Output** | Code changes (git diff) | One-time code execution | **Reusable YAML workflows** |
| **Tools** | Your codebase | Raw Python/JS/Shell | **412 pre-built modules** |
| **Learns** | No | No | **Yes — self-learning blueprints** |
| **Reusable** | Yes (code) | No (ephemeral) | **Yes (save, share, schedule)** |
| **Webhook/API** | No | No | **Yes** |
| **For** | Developers | Power users | **Anyone** |
| **License** | Apache-2.0 | AGPL-3.0 | **Apache-2.0** |

## 412 Batteries Included

Powered by [flyto-core](https://pypi.org/project/flyto-core/) — 412 automation modules across 78 categories:

| Category | Modules | Examples |
|----------|---------|---------|
| Browser | 38 | launch, goto, click, type, extract, screenshot, wait |
| HTTP / API | 15 | GET, POST, download, upload, GraphQL |
| String | 12 | split, replace, template, regex, slugify |
| Image | 10 | resize, crop, convert, watermark, compress |
| File | 9 | read, write, copy, zip, CSV, JSON |
| Database | 6 | query, insert, SQLite, PostgreSQL |
| Notification | 5 | email, Slack, Telegram, webhook |
| + 71 more categories | 317 | array, math, crypto, convert, flow, ... |

## Self-Learning Blueprints

The agent remembers what works. Good workflows are automatically saved as **blueprints** — reusable patterns that make future tasks faster.

```
First time:  "screenshot example.com" → 15s (discover modules, build from scratch)
Second time: "screenshot another.com" → 3s  (reuse learned blueprint)
```

Blueprints are stored locally (`~/.flyto/blueprints.db`) and scored by success rate. Export and share your best ones:

```bash
flyto-ai blueprints                             # View learned blueprints
flyto-ai blueprints --export > blueprints.yaml  # Export for sharing
```

## CLI

```bash
flyto-ai chat "scrape example.com"           # Generate a workflow
flyto-ai chat "take screenshot" -p ollama    # Use Ollama (no API key)
flyto-ai chat "hello" --json                 # Raw JSON output
flyto-ai chat "..." --webhook https://...    # POST result to webhook
flyto-ai serve --port 8080                   # HTTP server for triggers
flyto-ai blueprints                          # List learned blueprints
flyto-ai version                             # Version + dependency status
```

## Webhook & HTTP Server

**Send results anywhere:**

```bash
flyto-ai chat "scrape example.com" --webhook https://hook.site/xxx
```

**Accept triggers from anywhere:**

```bash
flyto-ai serve --port 8080

# Then from Slack, n8n, Make, or any HTTP client:
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "take a screenshot of example.com"}'
```

## Python API

```python
from flyto_ai import Agent, AgentConfig

# Auto-detects API keys from environment
agent = Agent(config=AgentConfig.from_env())
result = await agent.chat("extract all links from https://example.com")
print(result.message)    # YAML workflow
print(result.tool_calls) # Tool call log
```

## Multi-Provider

Works with any LLM:

```bash
export OPENAI_API_KEY=sk-...          # GPT-4o, GPT-4o-mini
export ANTHROPIC_API_KEY=sk-ant-...   # Claude Sonnet, Opus
flyto-ai chat "..." -p ollama         # Llama, Mistral, local models
flyto-ai chat "..." -p openai --model gpt-4o
```

## Architecture

```
User message
  → LLM (OpenAI / Anthropic / Ollama)
    → Function calling: search_modules, get_module_info, validate_params, ...
      → 412 flyto-core modules
      → Self-learning blueprints
      → Browser page inspection
    → YAML validation loop (auto-retry on errors)
  → Structured workflow output
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FLYTO_AI_PROVIDER` | `openai`, `anthropic`, or `ollama` |
| `FLYTO_AI_API_KEY` | API key (or use provider-specific vars below) |
| `FLYTO_AI_MODEL` | Model name override |
| `OPENAI_API_KEY` | Fallback for OpenAI provider |
| `ANTHROPIC_API_KEY` | Fallback for Anthropic provider |
| `FLYTO_AI_BASE_URL` | Custom API endpoint (OpenAI-compatible) |

## License

Apache-2.0 — use it commercially, fork it, build on it.
