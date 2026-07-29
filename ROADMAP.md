# Roadmap

Near term:
- Expand agent eval datasets for tool selection, schema validity, guardrail blocks, and MCP execution regressions.
- Add production service wiring and offline top-k recall/plan-validity datasets
  for the capability-router contract.
- Add trace graders for wrong tool, wrong handoff, missing approval, and unsafe credential handling.
- Promote successful Cloud AI workflows into blueprint candidates with evidence links.

Next:
- Align `flyto-pro-core` PlanContract, DecisionCard, Evidence, Intervention, and ObservationPacket with `flyto-ai` run evidence.
- Add UI-friendly streaming event docs for tool discovery, approval, execution, and result states.
- Reduce provider and CLI complexity hotspots without changing public contracts.
