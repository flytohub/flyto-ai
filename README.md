<p align="center">
  <img src="https://raw.githubusercontent.com/flytohub/flyto-ai/main/docs/logo.svg" alt="Flyto2 AI" width="120">
</p>

<h1 align="center">flyto-ai</h1>

<h3 align="center">Natural language → executable automation workflows</h3>

<p align="center">
  <em>aider writes code. open-interpreter runs code. <strong>flyto-ai builds &amp; runs workflows.</strong></em>
</p>

<p align="center">
  <a href="https://pypi.org/project/flyto-ai/"><img src="https://img.shields.io/pypi/v/flyto-ai?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/flyto-ai/"><img src="https://img.shields.io/pypi/pyversions/flyto-ai" alt="Python"></a>
  <a href="https://github.com/flytohub/flyto-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

---

## What is flyto-ai?

An AI agent that turns natural language into **real results + reusable automation workflows**.

Say "scrape the title from example.com" — the agent **executes it immediately** and gives you the result, plus a YAML workflow you can save, share, schedule, and run again.

```
> scrape the title from example.com

Result: "Example Domain"

```yaml
name: Scrape Title
params:
  url: "https://example.com"
steps:
  - id: launch
    module: browser.launch
  - id: goto
    module: browser.goto
    params:
      url: "${{params.url}}"
  - id: extract
    module: browser.extract
    params:
      selector: "h1"
```

## Quick Start

```bash
pip install flyto-ai
export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY
flyto-ai
```

One install, one command — interactive chat with **412 automation modules**, browser automation, and self-learning blueprints.

<p align="center">
  <img src="https://raw.githubusercontent.com/flytohub/flyto-ai/main/docs/demo.svg" alt="flyto-ai demo" width="800">
</p>

## Why flyto-ai?

| | aider | open-interpreter | flyto-ai |
|---|---|---|---|
| **Output** | Code changes (git diff) | One-time code execution | **Results + reusable YAML workflows** |
| **Tools** | Your codebase | Raw Python/JS/Shell | **412 pre-built modules** |
| **Learns** | No | No | **Yes — self-learning blueprints** |
| **Reusable** | Yes (code) | No (ephemeral) | **Yes (save, share, schedule)** |
| **Webhook/API** | No | No | **Yes** |
| **For** | Developers | Power users | **Developers & ops automation** |
| **License** | Apache-2.0 | AGPL-3.0 | **Apache-2.0** |

## Use Cases

### Web Scraping

```
> extract all product names and prices from example-shop.com/products
```

```yaml
name: Scrape Products
params:
  url: "https://example-shop.com/products"
steps:
  - id: launch
    module: browser.launch
  - id: goto
    module: browser.goto
    params:
      url: "${{params.url}}"
  - id: extract
    module: browser.extract
    params:
      selector: ".product"
      fields:
        name: ".product-name"
        price: ".product-price"
```

### Form Automation

```
> log in to staging.example.com, fill the contact form, and take a screenshot
```

```yaml
name: Fill Contact Form
steps:
  - id: launch
    module: browser.launch
  - id: login
    module: browser.login
    params:
      url: "https://staging.example.com/login"
      username_selector: "#email"
      password_selector: "#password"
      submit_selector: "button[type=submit]"
  - id: fill
    module: browser.form
    params:
      url: "https://staging.example.com/contact"
      fields:
        name: "Test User"
        message: "Hello from flyto-ai"
  - id: proof
    module: browser.screenshot
```

### API Monitoring + Notification

```
> check if https://api.example.com/health returns 200, if not send a Slack message
```

```yaml
name: Health Check Alert
params:
  endpoint: "https://api.example.com/health"
steps:
  - id: check
    module: http.get
    params:
      url: "${{params.endpoint}}"
  - id: notify
    module: notification.slack
    params:
      webhook_url: "${{params.slack_webhook}}"
      message: "Health check failed: ${{steps.check.status_code}}"
    condition: "${{steps.check.status_code}} != 200"
```

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
| + 71 more | 317 | array, math, crypto, convert, flow, ... |

Browse available modules:

```bash
flyto-ai version   # Shows installed module count
```

## Self-Learning Blueprints

The agent remembers what works. Good workflows are automatically saved as **blueprints** — reusable patterns that make future tasks faster.

```
First time:  "screenshot example.com" → 15s (discover modules, build from scratch)
Second time: "screenshot another.com" → 3s  (reuse learned blueprint)
```

**When are blueprints saved?**

- Only after validation passes + execution succeeds
- Trivial 1-2 step workflows are skipped
- Each blueprint has a score based on success/fail ratio — bad ones decay naturally
- Stored locally in `~/.flyto/blueprints.db`

```bash
flyto-ai blueprints                             # View learned blueprints
flyto-ai blueprints --export > blueprints.yaml  # Export for sharing
```

## CLI

```bash
flyto-ai                                     # Interactive chat — executes tasks directly
flyto-ai chat "scrape example.com"           # One-shot execute mode
flyto-ai chat "scrape example.com" --plan    # YAML-only mode (don't execute)
flyto-ai chat "take screenshot" -p ollama    # Use Ollama (no API key needed)
flyto-ai chat "..." --webhook https://...    # POST result to webhook
flyto-ai serve --port 8080                   # HTTP server for triggers
flyto-ai blueprints                          # List learned blueprints
flyto-ai version                             # Version + dependency status
```

### Interactive Mode

Just run `flyto-ai` — multi-turn conversation with up/down arrow history:

```
$ flyto-ai

  _____ _       _        ____       _    ___
 |  ___| |_   _| |_ ___ |___ \     / \  |_ _|
 | |_  | | | | | __/ _ \  __) |   / _ \  | |
 |  _| | | |_| | || (_) |/ __/   / ___ \ | |
 |_|   |_|\__, |\__\___/|_____|  /_/   \_\___|
           |___/

  v0.4.0  Interactive Mode

 > scrape the title from example.com
 (executes browser.launch → browser.goto → browser.extract)

 Result: "Example Domain"

   2 executed · 5 tool calls

 > now also take a screenshot
 (builds on previous context)

 > /clear
 Conversation cleared.
```

Commands: `/clear`, `/history`, `/version`, `/help`, `/exit`

## Webhook & HTTP Server

**Send results anywhere:**

```bash
flyto-ai chat "scrape example.com" --webhook https://hook.site/xxx
```

**Accept triggers from anywhere:**

```bash
flyto-ai serve --port 8080

# From Slack, n8n, Make, or any HTTP client:
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "take a screenshot of example.com"}'

# Execute mode (default) or plan-only:
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "scrape example.com", "mode": "yaml"}'
```

## Python API

```python
from flyto_ai import Agent, AgentConfig

agent = Agent(config=AgentConfig.from_env())

# Execute mode (default) — runs modules and returns results
result = await agent.chat("extract all links from https://example.com")
print(result.message)            # Result + YAML workflow
print(result.execution_results)  # Module execution results

# Plan-only mode — generates YAML without executing
result = await agent.chat("extract all links from example.com", mode="yaml")
print(result.message)            # YAML workflow only
```

## Multi-Provider

Works with any LLM provider:

```bash
export OPENAI_API_KEY=sk-...          # OpenAI models
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic models
flyto-ai chat "..." -p ollama         # Local models (Llama, Mistral, etc.)
flyto-ai chat "..." --model <name>    # Any specific model
```

## Security

- **Workflows are auditable** — YAML is human-readable, reviewable, and version-controllable
- **Module policies** — whitelist/denylist categories (e.g. block `file.*` or `database.*`)
- **Sensitive param redaction** — API keys and passwords are masked in tool call logs
- **Local-first** — blueprints stored in local SQLite, nothing sent to third parties
- **Webhook output** — structured JSON only, no raw credentials in payload

## Architecture

```
User message
  → LLM (OpenAI / Anthropic / Ollama)
    → Function calling: search_modules, get_module_info, execute_module, ...
      → 412 flyto-core modules
      → Self-learning blueprints
      → Browser page inspection
    → Execute mode: run modules, return results + YAML
    → Plan mode: YAML validation loop (auto-retry on errors)
  → Structured output (results + reusable workflow)
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
