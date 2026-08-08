# flyto-ai Agent Rules

- `flyto-ai` is the agent/runtime adapter between LLM providers, Flyto2 Cloud, blueprint learning, and `flyto-core` MCP tools.
- Do not store credentials, API keys, cookies, bearer tokens, screenshots containing secrets, or user passwords in source, tests, docs, handoffs, evidence, or generated YAML.
- Treat `flyto-core` as the module/runtime authority. Read its MCP schemas and recipes, but do not modify `flyto-core` from this repo.
- All `flyto-core` execution must flow through `flyto_ai.tools.core_tools` so validation, browser retry, permission checks, and MCP evidence metadata stay consistent.
- New agent capabilities need a closed loop: typed contract, guardrail, evidence/trace, test, docs, and rollback notes.
- Keep `.flyto-index/`, transcripts, evidence, build output, local DBs, and eval results out of commits.
- For deep changes, run flyto-indexer impact/search/verify before finalizing.

## Coding Route Rules

- The public `code-mcp` / `code-serve` service is one audited entry. Host-owned
  lanes surround whichever implementer startup selected: a mandatory Indexer
  gate before any model edit and again after the source-controlled checks,
  a mandatory read-only Blueprint discovery lane, and Core validation. All
  four lanes are configured on every strict public route; none is detachable,
  and Blueprint and Core may resolve only applied or not-applicable.
- Never let model prose assert that a lane, plan step, or gate ran. A lane
  outcome must come from a completed allowlisted call recorded in the route
  receipt.
- Do not add a route-bypass flag, do not let a green repository check stand in
  for the Indexer post-gate, and do not grant the implementer the audit tool,
  Blueprint execution authority, or Core danger/browser authority.
- Core execution keeps flowing through `flyto_ai.tools.core_tools`. Do not
  import sibling `flyto-core`, `flyto-blueprint`, or `flyto-indexer` source.
- Changing a lane, allowlist, bound, or receipt field is an architecture change:
  update `ARCHITECTURE.md`, `docs/architecture-map.md`, `STATE.md`, and
  `DECISIONS.md` in the same change.

## Architecture Invariant: Flytohub Product Topology

The canonical product topology in `ARCHITECTURE.md` and
`docs/architecture-map.md` is an architecture invariant, not a drawing.

- Any AI or agent that changes cross-repo ownership, a product role, an
  integration arrow, the coding route, or a named repository must update
  `ARCHITECTURE.md`, `docs/architecture-map.md`, `STATE.md`, and
  `DECISIONS.md` in the same change.
- Preserve the placement of `flyto-cloud` parallel to the combined
  `flyto-code` / `flyto-engine` column at the same level. Never nest Code or
  Engine below Cloud unless a dated `DECISIONS.md` entry explicitly supersedes
  this invariant.
- Verify the current repository reality before editing the map. Label an edge
  that does not exist yet as planned rather than implemented; do not silently
  rewrite the map to match an assumption.
- Keep the product topology visually and textually separate from the
  runtime-call diagrams. The topology records ownership and integration, not
  that every arrow is a synchronous call.

## Flyto2 Project Memory Contract

Every Flyto2 repository must keep this project-memory scaffold current:

- `AGENTS.md`: agent operating rules, repo-specific constraints, verification commands.
- `CLAUDE.md`: Claude-facing handoff rules when this repo is edited outside Codex.
- `PROJECT.md`: product purpose, owned surfaces, users, and non-goals.
- `ARCHITECTURE.md`: module boundaries, runtime shape, data flow, and integration points.
- `STATE.md`: current status, known risks, release/deploy state, and last verification.
- `ROADMAP.md`: near-term, later, and explicitly out-of-scope work.
- `tasks.md`: actionable checklist with owners/status when known.
- `DECISIONS.md`: durable architectural/product decisions with dates and rationale.
- `CHANGELOG.md`: user-visible or operator-visible changes.
- `docs/README.md`: index for durable docs in this repo.
- `workflows/*.md`: repeatable agent workflows for idea capture, planning, implementation, bugfix, refactor, investigation, and wrap-up.
- `handoffs/_registry.md`: index of handoffs; new handoffs use `YYYY-MM-DD-topic.md`.

When changing behavior, public copy, deployment, security posture, or frontend UX, update the relevant memory files in the same change. Do not leave stale brand, email, module count, route, or deployment information behind.

## Flyto2 Frontend Quality Gate

Any frontend, website, dashboard, extension webview, app screen, or generated UI in this repository must avoid these eight failures:

1. Ignoring accessibility: every interactive control needs keyboard access, visible focus, semantic HTML or ARIA, sufficient contrast, and useful alt/labels.
2. Missing responsive design: verify mobile, tablet, and desktop; no clipped text, overflow, hidden primary actions, or broken navigation.
3. Weak visual hierarchy: users must immediately see page purpose, primary action, status, and next step.
4. Template-looking UI: reuse Flyto2 design tokens and local components, but tailor layout and copy to the actual product surface.
5. Useless elements: remove decorative or placeholder UI that does not help the workflow, trust, navigation, or comprehension.
6. Unclear hierarchy: controls, cards, tables, panels, and modals must have clear grouping, spacing, headings, and state.
7. Unintuitive navigation: current location, back/forward paths, and cross-links to docs/blog/product pages must be obvious.
8. Hard-to-understand content: copy must be concrete, scannable, current, and consistent with Flyto2 terminology.

Frontend verification must include the relevant automated checks plus manual or screenshot review for responsive layout, accessibility states, navigation clarity, loading/empty/error states, and content readability. Public pages must preserve SEO basics: canonical URL, sitemap coverage, metadata, structured data when relevant, and no broken internal or external links.
