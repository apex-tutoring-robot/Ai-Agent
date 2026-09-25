"""
ExpressionController - maps a tutoring moment to one of a small
vocabulary of delivery states, then to concrete TTS SSML prosody
(rate/pitch) - see azure_services/tts_client.py's synthesize_stream()
`delivery` parameter.

Deliberately small (5 states) and voice-only today: the FaceWidget side
of the architecture review's ExpressionController (EXCITED -> brighter
avatar, etc.) isn't wired up here - this only changes how Jarvis SOUNDS,
not the face animation, which would be a separate UI-side change.
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
    Expression.NEUTRAL: DeliveryStyle(Expression.NEUTRAL, rate_percent=0, pitch_percent=0),
}

# Default mapping from a tutoring decision to an expression - not yet
# adjusted by anything richer (a real StudentModel/mood history); this is
# a starting point, applied fresh each turn.
_ACTION_EXPRESSION = {
    TutorAction.CONTINUE: Expression.FRIENDLY,
    TutorAction.CHALLENGE: Expression.EXCITED,
    TutorAction.HINT: Expression.ENCOURAGING,
    TutorAction.RETEACH: Expression.CALM,
}


class ExpressionController:
    def for_action(self, action: TutorAction, repeated_struggle: bool = False) -> DeliveryStyle:
        """
        Picks a DeliveryStyle for a tutor action. repeated_struggle (see
        affect/interaction_signals.py) nudges HINT toward the calmer
        delivery instead of the default encouraging-but-upbeat one - a
        student on their second-plus attempt needs more patience, not the
        same energy as the first hint.
        """
        expression = _ACTION_EXPRESSION.get(action, Expression.NEUTRAL)
        if repeated_struggle and expression == Expression.ENCOURAGING:
            expression = Expression.CALM
        return _STYLES[expression]

    def neutral(self) -> DeliveryStyle:
        return _STYLES[Expression.NEUTRAL]
