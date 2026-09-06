# Computer-local model inference

`flyto_ai.local_runtime` supports an explicitly selected Ollama or
OpenAI-compatible service on this computer. It does not download a model,
borrow an API credential, discover a remote server or switch to a fallback.
Cloud owns authenticated computer/source selection and model discovery.

```python
from flyto_ai import AgentConfig
from flyto_ai.local_runtime import LocalModelAgent, LocalModelConfig, complete_local_json

local = LocalModelConfig(
    provider="ollama", endpoint="http://127.0.0.1:11434",
    model=selected_model_id, timeout_seconds=45,
)
agent = LocalModelAgent(
    AgentConfig(provider="local_ai", model=selected_model_id), local=local,
    tool_executor=owned_executor, policies=owned_policy,
    system_prompt=supervisor_instructions,
)
try:
    response = await agent.chat(goal)
    # The host independently verifies observed results before choosing repair.
    response = await agent.continue_execution(goal=goal, message=correction)
finally:
    await agent.close()

review_json = await complete_local_json(
    local, prompt=observations, schema=review_schema,
    system_prompt=review_instructions, images=owned_images,
)
```

The existing Agent continues to own admission, permissions, continuation,
computer-local Core dispatch and receipts. The model receives a bounded JSON
intent schema and a host tool catalog as data. Its response cannot run native
tools, expand that catalog or manufacture execution evidence. Its answer is
not a goal verdict. Independent goal review remains necessary for every source.

## Endpoint, schema and image contract

`LocalModelConfig` requires `provider`, `endpoint` and an exact, nonempty model
identifier. There is no model list or model default hardcoded in the adapter.
Only literal loopback addresses or `localhost` are accepted; localhost is
canonicalized to `127.0.0.1`. Credentials, DNS hostnames, queries, fragments,
non-loopback addresses, ambiguous paths and IPv4-mapped IPv6 are refused.

Ollama uses `/api/chat` with `format` set to the supplied JSON schema.
OpenAI-compatible endpoints normalize to `/v1` and use `/chat/completions`
with strict `response_format.json_schema`. No native `tools` or `tool_choice`
is sent. Unsupported structured output, malformed/truncated output, provider
tool calls, mismatched returned model names and failed schema validation are
explicit errors. External schema references cannot initiate network access.

Host images are dictionaries containing `media_type` and validated `base64`.
They travel as actual image attachments, not paths or prose descriptions.
Ollama receives message `images`; compatible servers receive data URLs in
`image_url` content blocks. A service/model that rejects images or strict JSON
returns an actionable failure rather than a text-only fallback.

Inputs are limited to 256 KiB of UTF-8 prompt, 16 KiB of system instructions,
32 KiB of schema, eight images of at most 5 MB each and 64 KiB of returned JSON.
The HTTP response has a 2 MB bound. Timeout is 0.1–300 seconds and must fit the
owning job's remaining budget, including cleanup and acknowledgement margin.

HTTP uses an async client with environment credentials/proxies disabled and
redirects disabled. Timeout or cancellation closes the response/client socket;
there are no implicit retries. Closing the Agent cancels in-flight inference
and refuses later dispatch. No user model service is started or stopped.

## Delegated reasoning

`LocalModelAgent(config, local=None, completion_fn=...)` accepts the same trusted
three-keyword callback as `CliAgent`: `prompt`, `schema` and `system_prompt`.
This path never resolves an endpoint or constructs an HTTP client in Cloud.
Exactly one of native local configuration or delegated callback is required.
The selected computer performs inference using its own current local settings.
The initial delegated callback contract refuses image sidebands explicitly.

`provider="local_ai"` and an empty model in a delegated session mean the actual
remote selection is unavailable here; they do not imply GPT-4o or another
hardcoded default. The application can retain the authoritative selected model
metadata returned by the computer's reasoning job.

## Verification and rollback

Tests run actual loopback HTTP servers and the real host Agent loop. They
assert request model/schema/image bytes, secret-free headers, permission
refusal, actual host writes and observed receipts, cancellation disconnects,
timeout, invalid schema/model/output, no redirects and no hidden fallback.
These fixtures are protocol and authority proofs, not evidence that a real
local model solved a production goal. Real-model product acceptance remains
an application check against independent output evidence.

Rollback removes the application's opt-in local source and leaves native API
Agent behavior unchanged. There are no migrations, model downloads or server
configuration changes to undo.
