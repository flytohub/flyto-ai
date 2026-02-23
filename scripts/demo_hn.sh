#!/usr/bin/env bash
# Demo: Fetch Hacker News top 5 stories and format as markdown.
# Uses api.get (HTTP) — no browser dependency, free, stable.
#
# Usage:
#   bash scripts/demo_hn.sh                          # default provider (auto-detect)
#   bash scripts/demo_hn.sh -p ollama -m qwen2.5:7b  # Ollama
#   bash scripts/demo_hn.sh -p anthropic              # Claude
#
# What it demonstrates:
#   - Streaming output (tokens appear as they're generated)
#   - Multi-step tool calling (search → schema → execute × N)
#   - Works with any provider (OpenAI, Anthropic, Ollama)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "╭──────────────────────────────────────────────╮"
echo "│  Flyto AI Demo — Hacker News Top Stories     │"
echo "╰──────────────────────────────────────────────╯"
echo ""

exec flyto-ai chat \
  "Fetch the top 5 stories from Hacker News using the API at https://hacker-news.firebaseio.com/v0/topstories.json (returns array of IDs). Then fetch each story's details from https://hacker-news.firebaseio.com/v0/item/{id}.json. Format the results as a numbered markdown list with title, score, and URL." \
  "$@"
