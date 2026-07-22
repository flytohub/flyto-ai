# Flyto2 AI Runtime And Trust Whitepaper

## Thesis

An LLM is useful for interpreting intent and selecting capabilities, but it is a
poor execution substrate. Flyto2 AI narrows the model's role: it chooses typed,
registry-backed operations, while deterministic code validates, authorizes,
executes, records, and learns from those operations.

## Execution Model

A request is normalized, language/policy context is built, and deterministic
planning or blueprint matching runs before provider fallback. The provider sees
tool definitions, not raw execution privileges. A tool call passes registry
lookup, schema validation, permission decision, optional confirmation/sandbox,
dispatch, result normalization, evidence logging, and feedback. Successful
multi-step executions can become reusable blueprints; failures can improve
future routing without replaying secrets.

## Control Planes

- Flyto2 AI owns agent/provider orchestration and user-facing runtime contracts.
- Flyto2 Core owns executable module/recipe definitions and parameter schemas.
- Flyto2 Blueprint owns reusable workflow learning and confidence lifecycle.
- Flyto2 Pro Core provides shared plan/decision/evidence contracts.
- Flyto2 Cloud consumes stable AI contracts rather than importing Core internals.
- Flyto2 Indexer supplies repository context and impact evidence for coding mode.

## Trust Boundaries

User prompts, web pages, tool results, extension manifests, hook output, MCP
servers, provider responses, config files, channel updates, and imported findings
are untrusted. Controls include injection scanning, URL policy, output truncation,
credential redaction/vaulting, allowlisted hook environments, explicit permission
levels, network-disabled sandboxing, cost/round limits, evidence, and bounded
transcript/memory storage.

Security workflow generation is dual-use. It is constrained to structured
finding categories, rejects metadata/private targets, defaults to staging hosts,
and requires explicit production override. Authorization to test a target remains
an operator responsibility and cannot be inferred from a URL.

## Failure Semantics

Provider, MCP, validation, permission, budget, sandbox, browser, and downstream
module failures remain distinguishable. Failover may change provider only when
configured; it must not bypass a safety decision. Prompt evolution archives and
scores candidates but never promotes one automatically. Missing optional systems
degrade to explicit unavailable states rather than fabricated success.

## Evidence And Reproducibility

Tool calls record normalized input/result metadata, provider usage, MCP source
and contract version, module/recipe identity, and verification outcomes. YAML
workflows and blueprint references make successful execution replayable. Generated
source references and CI documentation checks make public claims reviewable
against the actual package.

## Verification Claim

This architecture is not itself a security certification. Evidence consists of
offline tests, type/runtime contracts, Ruff/compile checks, package build checks,
secret/dependency/container scans, source-reference drift checks, and Flyto2
Indexer verification. Provider-specific quality, live browser behavior, channel
delivery, and Cloud integration require credentialed staging tests.

