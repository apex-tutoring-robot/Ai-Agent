"""
ExpressionController - maps a tutoring moment to one of a small
vocabulary of delivery states, then to concrete TTS SSML prosody
(rate/pitch) - see azure_services/tts_client.py's synthesize_stream()
`delivery` parameter. Each Expression also drives FaceWidget via
UISignals.set_expression - see visuals/ui/main_window.py's show_expression().

SAD is wired to TutorAction.RETEACH - the moment a student gets the same
question wrong twice, teaching stops re-testing it and explains the
concept a different way instead. That's the one point in the current
tutoring flow with a genuinely "this isn't landing" quality to it, as
opposed to HINT's first-attempt "close, try again" (still upbeat/
patient, stays ENCOURAGING/CALM).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tutor.decision_engine import TutorAction


class Expression(Enum):
    FRIENDLY = "friendly"
    ENCOURAGING = "encouraging"
    CALM = "calm"
    EXCITED = "excited"
    CONFUSED = "confused"
    SAD = "sad"
    NEUTRAL = "neutral"


@dataclass
class DeliveryStyle:
    expression: Expression
    rate_percent: int   # SSML prosody rate nudge, e.g. -10 for 10% slower
    pitch_percent: int  # SSML prosody pitch nudge, e.g. +5 for 5% higher


_STYLES = {
    Expression.FRIENDLY: DeliveryStyle(Expression.FRIENDLY, rate_percent=0, pitch_percent=0),
    Expression.ENCOURAGING: DeliveryStyle(Expression.ENCOURAGING, rate_percent=-10, pitch_percent=0),
    Expression.CALM: DeliveryStyle(Expression.CALM, rate_percent=-15, pitch_percent=-3),
    Expression.EXCITED: DeliveryStyle(Expression.EXCITED, rate_percent=8, pitch_percent=5),
    Expression.CONFUSED: DeliveryStyle(Expression.CONFUSED, rate_percent=-8, pitch_percent=-2),
    Expression.SAD: DeliveryStyle(Expression.SAD, rate_percent=-12, pitch_percent=-4),
    Expression.NEUTRAL: DeliveryStyle(Expression.NEUTRAL, rate_percent=0, pitch_percent=0),
}

# Default mapping from a tutoring decision to an expression - not yet
# adjusted by anything richer (a real StudentModel/mood history); this is
# a starting point, applied fresh each turn.
_ACTION_EXPRESSION = {
    TutorAction.CONTINUE: Expression.FRIENDLY,
    TutorAction.CHALLENGE: Expression.EXCITED,
    TutorAction.HINT: Expression.ENCOURAGING,
    TutorAction.RETEACH: Expression.SAD,
}


class ExpressionController:
    def for_action(self, action: TutorAction, repeated_struggle: bool = False, hesitation: bool = False) -> DeliveryStyle:
        """
        Picks a DeliveryStyle for a tutor action.

        repeated_struggle (see affect/interaction_signals.py) nudges HINT
        toward the calmer delivery instead of the default
        encouraging-but-upbeat one - a student on their second-plus
        attempt needs more patience, not the same energy as the first hint.

        hesitation nudges an otherwise upbeat delivery (FRIENDLY/EXCITED -
        i.e. the student was actually right) toward ENCOURAGING instead: a
        hedging "um, maybe 20?" that turns out correct still deserves a
        gentler "yes, see, you knew it!" rather than big excited energy
        that would feel mismatched with how unsure they sounded.
        """
        expression = _ACTION_EXPRESSION.get(action, Expression.NEUTRAL)
        if repeated_struggle and expression == Expression.ENCOURAGING:
            expression = Expression.CALM
        elif hesitation and expression in (Expression.FRIENDLY, Expression.EXCITED):
            expression = Expression.ENCOURAGING
        return _STYLES[expression]

    def neutral(self) -> DeliveryStyle:
        return _STYLES[Expression.NEUTRAL]

    def refusal(self) -> DeliveryStyle:
        """
        For a guardrails/fast-filter safety refusal (see main.py's
        _run_output_safety_check and the FastContentFilter block paths) -
        CONFUSED reads as "I'm not sure how to answer that", which is a
        more honest expression for the moment than a flat neutral face.
        """
        return _STYLES[Expression.CONFUSED]
