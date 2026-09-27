"""
TutorDecisionEngine - centralizes what Jarvis does after evaluating a
student's answer to a comprehension-check question, instead of leaving
that policy embedded inline in main.py's _handle_teaching_answer.

Today this only formalizes the EXISTING hint/reteach/continue/challenge
policy (see _handle_teaching_answer's prior inline version) as an
explicit, independently testable decision - it does not yet add new
actions like ASK/WAIT/ENCOURAGE, which depend on a richer StudentModel
and InteractionSignals that don't exist yet. Building this now, even with
just today's four actions, gives that future work (proactive
QuestionEngine, ReviewScheduler weighting, ExpressionController) a single
seam to plug into instead of more inline branching in main.py.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class TutorAction(Enum):
    CONTINUE = "continue"
    HINT = "hint"
    RETEACH = "reteach"
    CHALLENGE = "challenge"


# Actions the student was right for (used to update mastery) - the same
# split main.py's _handle_teaching_answer already made inline.
_CORRECT_ACTIONS = {TutorAction.CONTINUE, TutorAction.CHALLENGE}
_HINT_USED_ACTIONS = {TutorAction.HINT, TutorAction.RETEACH}


@dataclass
class TutorDecision:
    action: TutorAction
    response_text: str
    is_correct: bool
    used_hint: bool


class TutorDecisionEngine:
    """
    Turns an LLM answer-evaluation result (see LLMClient.evaluate_answer)
    into a concrete TutorDecision. A recognized recommended_action is
    trusted directly; anything unrecognized (a malformed/unexpected LLM
    response) falls back to "hint" - the same safe default
    evaluate_answer() itself already uses on a parse failure, kept
    consistent here for a value that parsed but doesn't match a known
    action.
    """

    _VALID_ACTIONS = {a.value for a in TutorAction}

    def decide(self, evaluation: Dict) -> TutorDecision:
        raw_action = evaluation.get("recommended_action", "hint")
        if raw_action not in self._VALID_ACTIONS:
            raw_action = "hint"
        action = TutorAction(raw_action)

        return TutorDecision(
            action=action,
            response_text=evaluation.get("response") or "Let's keep going.",
            is_correct=action in _CORRECT_ACTIONS,
            used_hint=action in _HINT_USED_ACTIONS,
        )
