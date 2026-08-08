# Mission Stations interpretation handoff

## Closed

- Judges physically draw the Zone and Objective cards. The operator records
  `card_source=judge_draw`; `flyto-ai` does not draw or randomize tasks.
- `flyto_ai.mission_interpretation` validates the card, evidence, and APPROVED
  capability ceiling before a model call, then retains an isolated canonical
  JSON snapshot across the provider await.
- Provider output is limited to reading, clarification, and approved
  capability IDs. Card evidence remains outside that object.
- Hostile/invalid output or provider failure produces a deterministic card-only
  fallback and a content-addressed attestation without raw provider errors.
- Resource selection, assignments, execution authorization, physical control,
  and Task completion remain external boundaries.

## Proof

```bash
/opt/homebrew/bin/python3.11 -m pytest -q tests/test_mission_interpretation.py
/opt/homebrew/bin/python3.11 -m ruff check flyto_ai/mission_interpretation.py tests/test_mission_interpretation.py
python3 scripts/generate_reference.py --check
```

## Next integration

Cloud should submit the recorded judge-card Task and its approved capability
projection, retain the returned attestation on the Task timeline, and hand only
a separately planned and assigned dispatch to Robotics.
