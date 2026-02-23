# Model Compatibility Guide

flyto-ai relies on **function calling** (tool use) to discover, validate, and execute automation modules. Model quality directly affects workflow reliability.

## Recommended Models

| Provider | Model | Function Calling | Notes |
|----------|-------|:----------------:|-------|
| OpenAI | `gpt-4o` | Excellent | Best overall — fast, accurate tool use |
| OpenAI | `gpt-4o-mini` | Good | Cost-effective default, occasionally skips schema checks |
| Anthropic | `claude-sonnet-4-5` | Excellent | Strong reasoning, reliable multi-step workflows |
| Anthropic | `claude-haiku-4-5` | Good | Fastest Anthropic option, good for simple tasks |
| Ollama | `qwen2.5:14b` | Good | Best local model — reliable function calling |
| Ollama | `qwen2.5:7b` | Good | Solid local option, VRAM-friendly |
| Ollama | `qwen2.5-coder:7b` | Good | Code-focused variant, same FC quality |

## Usable (with limitations)

| Provider | Model | Function Calling | Known Limitations |
|----------|-------|:----------------:|-------------------|
| Ollama | `llama3.1:8b` | Fair | Sometimes generates invalid JSON in tool args |
| Ollama | `mistral` | Fair | Inconsistent tool format, may need retries |

## Not Recommended

| Provider | Model | Function Calling | Why |
|----------|-------|:----------------:|-----|
| Ollama | `llama3.2` | Poor | Weak function calling — often ignores tools or hallucinates module names |
| Ollama | `deepseek-r1:8b` | Poor | Reasoning model, not optimized for structured tool use |

## VRAM Requirements (Ollama)

| Model | Parameters | Min VRAM | Recommended VRAM |
|-------|-----------|----------|-----------------|
| `qwen2.5:7b` | 7B | 5 GB | 8 GB |
| `qwen2.5:14b` | 14B | 10 GB | 16 GB |
| `llama3.1:8b` | 8B | 5 GB | 8 GB |
| `llama3.2` | 3B | 3 GB | 4 GB |
| `mistral` | 7B | 5 GB | 8 GB |

## Quick Start by Provider

### OpenAI (default)
```bash
export OPENAI_API_KEY=sk-...
flyto-ai chat "fetch top stories from hacker news"
```

### Anthropic
```bash
export ANTHROPIC_API_KEY=sk-ant-...
flyto-ai chat "fetch top stories from hacker news" -p anthropic
```

### Ollama (local, free)
```bash
# Install and pull a model first:
# ollama pull qwen2.5:7b

flyto-ai chat "fetch top stories from hacker news" -p ollama -m qwen2.5:7b
```

## Rating Definitions

- **Excellent** — Consistently follows tool schemas, handles multi-step workflows (5+ tools), validates before executing
- **Good** — Reliable for most workflows, occasionally skips optional validation steps
- **Fair** — Works for simple 1-2 step tasks, unreliable for complex chains
- **Poor** — Frequently ignores tools, invents module names, or produces malformed tool calls
