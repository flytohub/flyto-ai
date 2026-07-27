# Blueprint benchmark host

The benchmark answers one narrow question: when Flyto2 can safely reuse a
verified Blueprint, does the planner use fewer model calls and tokens without
becoming less reliable?

The experiment runs ten routing and trust-boundary questions twenty times in
four paired modes. That produces 800 metrics-only records:

| Mode | What changes |
| --- | --- |
| `agent_baseline` | A local model makes the planning decision without Flyto routing. |
| `flyto_no_blueprint` | Flyto routing is active, but Blueprint is disabled. |
| `blueprint_cold` | Blueprint is installed without learned reusable experience. |
| `blueprint_warm` | Verified compatible experience is available. |

The host uses Ollama's native counters, `prompt_eval_count` and `eval_count`.
Missing counters fail the run; they are never converted to zero. Qwen thinking
is disabled so the run is bounded and repeatable. The four modes share one seed,
and their execution order rotates on every trial to reduce warm-cache bias.

Prompts and model responses stay in memory. The JSONL output contains suite and
environment digests, assertion counts, planner tokens, model calls, latency,
retries, tool calls, and false reuse only. The sealed holdout enters through an
environment variable and is checked against the suite's SHA-256 commitment.

Run it from a checkout after copying the host template from
`flyto-blueprint/benchmarks/templates/host-run-template.yaml`:

```bash
export FLYTO_BENCHMARK_SEALED_PROMPT='<private holdout>'
PYTHONPATH=/path/to/flyto-ai:/path/to/flyto-blueprint \
python3.11 scripts/run_blueprint_benchmark.py \
  --suite /path/to/flyto-blueprint/benchmarks/suites/blueprint-effectiveness-v2.yaml \
  --config /path/to/flyto-blueprint/benchmarks/templates/host-run-template.yaml \
  --dataset-commit <flyto-blueprint-suite-commit> \
  --flyto-ai-commit <flyto-ai-runner-commit> \
  --flyto-blueprint-commit <flyto-blueprint-suite-commit> \
  --output /path/to/result.runs.jsonl
```

The runner refuses fewer than twenty trials, a changed model digest, a
credential-bearing or non-loopback Ollama URL, an unknown task/assertion, an
unmatched sealed prompt, and an existing output unless `--overwrite` is
explicitly supplied.

This is a planner benchmark, not a claim that every model-backed workflow step
is token-free. It exercises the production intent classifier and Blueprint
search, trust, import, compatibility guard, and expansion primitives. Core
workflow execution remains outside the claim.
