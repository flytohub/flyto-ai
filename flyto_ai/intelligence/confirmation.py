"""Reading a bare "yes" as the answer the assistant just asked for.

`classify_tool_intent` looks at one message in isolation, which is right for a
safety boundary and wrong for a conversation: "確認" -- the exact word the
assistant had just asked for -- scored as small talk, exposed no tools, and
left the assistant able only to ask again. The operator confirmed six times and
nothing ran.

Kept out of `agent.py` for the same reason the verb tables are kept out of
`planner.py`: the agent orchestrates a turn, and deciding what a turn MEANS is
somebody else's job.
"""

import re

from flyto_ai.intelligence.action_verbs import is_bare_affirmation
from flyto_ai.intelligence.planner import ToolIntentDecision, classify_tool_intent

# Re-exported so a caller that routes through this module needs one import for
# the decision and its type, rather than reaching past it into the planner.
__all__ = [
    "ToolIntentDecision",
    "classify_tool_intent",
    "last_turn_proposed_an_action",
    "route_with_confirmation",
]

#: An assistant turn that asked permission to do something. Matched on the ask
#: rather than on a stored flag, so a resumed or forked session behaves the
#: same: the transcript still has the words, and a flag would have to be kept
#: in step with them.
_PROPOSAL_RE = re.compile(
    r"(?:confirm|proceed|shall\s+i|would\s+you\s+like|do\s+you\s+want|"
    r"請確認|请确认|確認|确认|是否|要不要|要我)",
    re.IGNORECASE,
)


def last_turn_proposed_an_action(history) -> bool:
    """Whether the assistant's most recent message asked to do something.

    Only an explicit ask counts. Running the classifier over the assistant's
    own words was too loose -- a plain statement like "today is sunny" carries
    no action verb but is not answer_only either, so every "yes" after any
    reply was promoted.
    """
    for entry in reversed(list(history or [])):
        role = entry.get("role") if isinstance(entry, dict) else getattr(entry, "role", "")
        if role != "assistant":
            continue
        content = (
            entry.get("content") if isinstance(entry, dict)
            else getattr(entry, "content", "")
        )
        return bool(_PROPOSAL_RE.search(str(content or "")))
    return False


def route_with_confirmation(message, history) -> ToolIntentDecision:
    """Classify this turn, reading a bare affirmation as the answer it is.

    A bare affirmation is promoted to `action` ONLY when the assistant's
    previous turn actually proposed one. Agreement to a remark about the
    weather is still conversation, and a message that agrees and then asks for
    something else is not a bare affirmation at all -- it classifies on its own
    terms.
    """
    decision = classify_tool_intent(message)
    if decision.mode == "action":
        return decision
    if not is_bare_affirmation(message):
        return decision
    if not last_turn_proposed_an_action(history):
        return decision
    return ToolIntentDecision(
        "action", 0.9, "confirmed_previous_proposal", ("affirmation",),
    )
