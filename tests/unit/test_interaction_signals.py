"""
Tests for affect.interaction_signals.InteractionSignalEngine - text
heuristics only, see the module docstring for why (no real audio affect
analysis exists yet).
"""

from affect.interaction_signals import InteractionSignalEngine


class TestAnalyze:
    def test_confident_answer_has_no_signals(self):
        signals = InteractionSignalEngine().analyze("The answer is 24 square centimeters.", attempts_so_far=0)
        assert signals.hesitation is False
        assert signals.low_confidence is False
        assert signals.repeated_struggle is False

    def test_hedging_language_is_hesitation(self):
        signals = InteractionSignalEngine().analyze("Um, maybe 12?", attempts_so_far=0)
        assert signals.hesitation is True

    def test_short_hedging_answer_is_low_confidence(self):
        signals = InteractionSignalEngine().analyze("I don't know", attempts_so_far=0)
        assert signals.low_confidence is True

    def test_long_hedging_answer_is_not_low_confidence(self):
        text = "I guess it could maybe be twelve but I worked it out using the formula we just talked about"
        signals = InteractionSignalEngine().analyze(text, attempts_so_far=0)
        assert signals.hesitation is True
        assert signals.low_confidence is False

    def test_second_attempt_is_repeated_struggle(self):
        signals = InteractionSignalEngine().analyze("20", attempts_so_far=1)
        assert signals.repeated_struggle is True

    def test_first_attempt_is_not_repeated_struggle(self):
        signals = InteractionSignalEngine().analyze("20", attempts_so_far=0)
        assert signals.repeated_struggle is False
