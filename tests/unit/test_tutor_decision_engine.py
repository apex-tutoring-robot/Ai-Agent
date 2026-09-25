"""
Tests for tutor.decision_engine.TutorDecisionEngine - pure logic, no
dependencies, formalizing the hint/reteach/continue/challenge policy
previously inline in main.py's _handle_teaching_answer.
"""

from tutor.decision_engine import TutorAction, TutorDecisionEngine


class TestDecide:
    def test_continue_is_correct_and_not_hint_used(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "continue", "response": "Nice job!"})
        assert decision.action == TutorAction.CONTINUE
        assert decision.is_correct is True
        assert decision.used_hint is False
        assert decision.response_text == "Nice job!"

    def test_challenge_is_correct_and_not_hint_used(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "challenge", "response": "Try a harder one."})
        assert decision.action == TutorAction.CHALLENGE
        assert decision.is_correct is True
        assert decision.used_hint is False

    def test_hint_is_not_correct_and_used_hint(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "hint", "response": "Think about the formula."})
        assert decision.action == TutorAction.HINT
        assert decision.is_correct is False
        assert decision.used_hint is True

    def test_reteach_is_not_correct_and_used_hint(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "reteach", "response": "Let's try a different way."})
        assert decision.action == TutorAction.RETEACH
        assert decision.is_correct is False
        assert decision.used_hint is True

    def test_unrecognized_action_falls_back_to_hint(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "explode", "response": "..."})
        assert decision.action == TutorAction.HINT

    def test_missing_action_falls_back_to_hint(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"response": "..."})
        assert decision.action == TutorAction.HINT

    def test_missing_response_falls_back_to_generic_phrase(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "continue"})
        assert decision.response_text == "Let's keep going."

    def test_empty_response_falls_back_to_generic_phrase(self):
        engine = TutorDecisionEngine()
        decision = engine.decide({"recommended_action": "continue", "response": ""})
        assert decision.response_text == "Let's keep going."
