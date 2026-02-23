# HN Launch Kit

## Post Title

```
Show HN: 412 deterministic modules so AI agents stop hallucinating shell commands
```

## Post URL

```
https://github.com/flytohub/flyto-ai
```

## Timing

Tuesday–Thursday, 台灣 00:00–02:00（美西 8–10 AM）

---

## First Comment（發文後立刻貼）

```
I built this because I kept watching AI agents write shell commands
that almost work. Wrong curl flags, hallucinated Python APIs,
slightly different output every time. The fix felt obvious in
hindsight: don't let the LLM write code at all.

flyto-ai has 412 pre-built modules (browser, HTTP, file, database,
image, notifications, etc.). The LLM's only job is to search for the
right module and fill in the parameters — which are validated against
the module's schema before execution. If the params are wrong, it
fails fast, not at runtime.

Every execution produces a reusable YAML workflow. The interesting
part: successful runs are auto-saved as "blueprints." Next time a
similar task comes in, the agent replays the blueprint with zero
LLM inference — same result, instant, free.

    pip install flyto-ai && flyto-ai

Works with OpenAI, Anthropic, or local models via Ollama. Apache-2.0.

Curious whether others have landed on a similar separation between
"LLM decides what" and "deterministic engine does how."
```

---

## Follow-up: Technical Details（有人問細節時貼）

```
Some technical details:

The agent loop is simple:
1. search_modules("browser scrape extract") → finds relevant modules
2. get_module_info("browser.extract") → reads param schema
3. execute_module("browser.extract", {selector: "h1"}) → runs it
4. LLM never writes code — only picks modules and fills params

The 412 modules come from flyto-core (separate PyPI package, also
Apache-2.0). Categories include browser (39 modules), HTTP/API, file
ops, database, image processing, cloud storage, notifications, flow
control, and more.

Quick benchmark on "scrape the title from example.com":
- open-interpreter: ~8K tokens, ~12s, non-deterministic
- flyto-ai first run: ~2K tokens, ~8s, deterministic
- flyto-ai second run (blueprint): 0 tokens, ~0.5s

Multi-provider: OpenAI, Anthropic, Ollama. The Ollama path means you
can run fully local with qwen2.5:7b — zero API cost.
```

---

## Follow-up: Blueprint Learning（有人問 learning 時貼）

```
Blueprint learning is closed-loop, zero LLM involved:

- Execution succeeds with 3+ steps → auto-saved (score 70)
- Blueprint reused + succeeds → score +5
- Blueprint fails → score -10
- Score < 10 → auto-retired, never suggested again

So the agent gets faster over time — not because the LLM improves,
but because proven workflows are replayed directly. Second run of
a similar task costs zero tokens.

    flyto-ai blueprints                  # view learned blueprints
    flyto-ai blueprints --export         # export as YAML for sharing
```

---

## Follow-up: MCP / Integration（有人問整合時貼）

```
flyto-ai also works as an MCP server:

    flyto-ai mcp

This lets Claude Desktop, ChatGPT, or any MCP-compatible host use
the 412 modules as tools. There's also an OpenClaw plugin
(flyto-openclaw) if you want to add deterministic execution to
OpenClaw agents.

HTTP server mode for webhooks:

    flyto-ai serve --port 8080
    curl -X POST http://localhost:8080/chat \
      -d '{"message": "screenshot example.com"}'
```

---

## Handling Common HN Questions

**"How is this different from Langchain / CrewAI / AutoGen?"**

```
Those are orchestration frameworks — they help you chain LLM calls
together. flyto-ai is an execution engine: the LLM picks from
pre-built, schema-validated modules instead of writing code. The
difference is at the execution layer, not the orchestration layer.
You could actually use flyto-core modules inside a Langchain chain.
```

**"Why not just let the LLM write Python? It's more flexible."**

```
Flexibility is the problem. "More flexible" means "more ways to
fail silently." When the LLM writes `requests.get(url)` it might
forget headers, timeout handling, retry logic, or error parsing.
The browser.extract module handles all of that — tested once,
reused everywhere. The trade-off is real: you can't do arbitrary
code. But for the 80% of automation tasks that map to existing
modules, deterministic > flexible.
```

**"390 modules sounds like a lot. Are they actually useful?"**

```
Fair question. The breakdown: 39 browser modules (launch, goto,
click, type, extract, screenshot, wait, form, login...), 35 atomic
building blocks, 23 flow control (conditionals, loops, error
handling), 14 cloud, 13 data transform, 12 array ops, 11 string
ops, 9 image, 9 notification (email, Slack, Telegram), 9 HTTP/API,
6 database, and ~200 more across crypto, docker, k8s, testing, etc.

The most-used ones in practice: browser.*, http.*, notification.*,
file.*, and data.*. The long tail exists for when you need it.
```

**"What about security? Running arbitrary modules sounds risky."**

```
Modules are category-gated — you can allowlist/denylist entire
categories (e.g. block file.* or database.* for untrusted inputs).
API keys and passwords are auto-redacted in logs. Workflows are
auditable YAML — you can review exactly what will run before
execution. Everything is local-first, nothing phoned home.
```
