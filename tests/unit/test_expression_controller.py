"""
Tests for expression.controller.ExpressionController - the tutor-action
-> voice-delivery mapping (see azure_services/tts_client.py's
synthesize_stream `delivery` parameter).
"""

from expression.controller import Expression, ExpressionController
from tutor.decision_engine import TutorAction


class TestForAction:
    def test_continue_is_friendly(self):
        style = ExpressionController().for_action(TutorAction.CONTINUE)
        assert style.expression == Expression.FRIENDLY

    def test_challenge_is_excited(self):
        style = ExpressionController().for_action(TutorAction.CHALLENGE)
        assert style.expression == Expression.EXCITED
        assert style.rate_percent > 0

    def test_hint_is_encouraging_by_default(self):
        style = ExpressionController().for_action(TutorAction.HINT, repeated_struggle=False)
        assert style.expression == Expression.ENCOURAGING
        assert style.rate_percent < 0

    def test_hint_becomes_calm_on_repeated_struggle(self):
        style = ExpressionController().for_action(TutorAction.HINT, repeated_struggle=True)
        assert style.expression == Expression.CALM

    def test_reteach_is_calm(self):
        style = ExpressionController().for_action(TutorAction.RETEACH)
        assert style.expression == Expression.CALM

    def test_repeated_struggle_does_not_affect_non_hint_actions(self):
        style = ExpressionController().for_action(TutorAction.CONTINUE, repeated_struggle=True)
        assert style.expression == Expression.FRIENDLY

    def test_neutral_style_has_no_adjustment(self):
        style = ExpressionController().neutral()
        assert style.rate_percent == 0
        assert style.pitch_percent == 0
