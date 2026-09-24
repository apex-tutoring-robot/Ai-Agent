"""
Tests for JarvisBot._REPEAT_REQUEST_PATTERN - detecting "please repeat the
question" during a teaching turn's comprehension check, as opposed to an
actual (wrong) answer attempt.

Regression test for a real bug found via live testing: "can you say the
question again?" was fed straight into evaluate_answer(), which judged it
as a wrong answer ("the student did not attempt to answer the question"),
burned an attempt, and gave a hint instead of just repeating the question -
which then made the student's real next attempt get unfairly escalated
straight to "reteach" since the attempt counter was already at 1.
"""

import pytest

from main import JarvisBot


class TestDetectsRepeatRequests:
    @pytest.mark.parametrize(
        "text",
        [
            "can you, can you say the question again?",
            "what was the question",
            "can you repeat that",
            "repeat the question please",
            "I didn't catch that",
            "I didn't hear that",
            "come again?",
            "one more time please",
            "say that again",
            "say it again",
        ],
    )
    def test_recognized_as_repeat_request(self, text):
        assert JarvisBot._REPEAT_REQUEST_PATTERN.search(text) is not None


class TestDoesNotFalsePositiveOnRealAnswers:
    @pytest.mark.parametrize(
        "text",
        [
            "multiply",
            "20",
            "is it 24",
            "can I try again",  # contains "again" but isn't a repeat request
            "yes because if you simplify 2/4 you get 1/2",
            "no because 4 is bigger than 2",
        ],
    )
    def test_not_recognized_as_repeat_request(self, text):
        assert JarvisBot._REPEAT_REQUEST_PATTERN.search(text) is None
