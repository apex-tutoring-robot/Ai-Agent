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
        assert style.style == "excited"

    def test_hint_is_encouraging_by_default(self):
        style = ExpressionController().for_action(TutorAction.HINT, repeated_struggle=False)
        assert style.expression == Expression.ENCOURAGING
        assert style.rate_percent < 0

    def test_hint_becomes_calm_on_repeated_struggle(self):
        style = ExpressionController().for_action(TutorAction.HINT, repeated_struggle=True)
        assert style.expression == Expression.CALM

    def test_reteach_is_sad(self):
        style = ExpressionController().for_action(TutorAction.RETEACH)
        assert style.expression == Expression.SAD
        assert style.style == "sad"

    def test_repeated_struggle_does_not_affect_non_hint_actions(self):
        style = ExpressionController().for_action(TutorAction.CONTINUE, repeated_struggle=True)
        assert style.expression == Expression.FRIENDLY

    def test_neutral_style_has_no_adjustment(self):
        style = ExpressionController().neutral()
        assert style.rate_percent == 0
        assert style.pitch_percent == 0

    def test_refusal_style_is_confused(self):
        style = ExpressionController().refusal()
        assert style.expression == Expression.CONFUSED
        assert style.style == "whispering"

    def test_friendly_accessor_is_friendly(self):
        # The default delivery for any speech that doesn't have a more
        # specific mood decided for it - see main.py's
        # _speak_fixed_phrase/_speak_system_message/_speak_standalone.
        style = ExpressionController().friendly()
        assert style.expression == Expression.FRIENDLY
        assert style.style == "friendly"

    def test_encouraging_accessor_is_encouraging(self):
        style = ExpressionController().encouraging()
        assert style.expression == Expression.ENCOURAGING
        assert style.style == "hopeful"

    def test_every_expression_has_its_own_distinct_style(self):
        # Direct requirement: every mood should actually sound different
        # from every other mood, not share a style with one it's meant
        # to be distinguishable from (e.g. CONFUSED used to sit on the
        # same "chat" style as NEUTRAL, making a refusal and an ordinary
        # neutral line indistinguishable by ear).
        controller = ExpressionController()
        deliveries = [
            controller.for_action(TutorAction.CONTINUE),
            controller.for_action(TutorAction.CHALLENGE),
            controller.for_action(TutorAction.HINT, repeated_struggle=False),
            controller.for_action(TutorAction.HINT, repeated_struggle=True),
            controller.for_action(TutorAction.RETEACH),
            controller.neutral(),
            controller.refusal(),
        ]
        styles_used = [d.style for d in deliveries]
        assert len(styles_used) == len(set(styles_used))

    def test_every_style_has_a_non_empty_azure_style_name(self):
        # Every DeliveryStyle this controller can hand out must carry a
        # real Azure express-as style (even if it's just "chat", the
        # base default) - an accidentally-empty one would silently drop
        # the <mstts:express-as> wrapper in _build_ssml, losing whatever
        # vocal performance this Expression was supposed to carry.
        controller = ExpressionController()
        styles = [
            controller.for_action(TutorAction.CONTINUE),
            controller.for_action(TutorAction.CHALLENGE),
            controller.for_action(TutorAction.HINT, repeated_struggle=False),
            controller.for_action(TutorAction.HINT, repeated_struggle=True),
            controller.for_action(TutorAction.RETEACH),
            controller.neutral(),
            controller.refusal(),
        ]
        for delivery in styles:
            assert delivery.style
            assert 0.01 <= delivery.styledegree <= 2.0
