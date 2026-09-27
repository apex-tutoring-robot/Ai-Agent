"""
Tests for JarvisBot.is_homework_help_request() - the trigger that routes a
turn to the photo-capture Homework Help flow (see main.py's
_run_homework_help_turn) instead of a normal spoken math question.

Doesn't touch `self`, so it's called unbound (JarvisBot.
is_homework_help_request(None, text)) rather than constructing a real
JarvisBot - same pattern as test_is_math_query.py.
"""

import pytest

from main import JarvisBot


def is_homework_help_request(text: str) -> bool:
    return JarvisBot.is_homework_help_request(None, text)


class TestIsHomeworkHelpRequest:
    @pytest.mark.parametrize(
        "text",
        [
            "Can you help me with my homework?",
            "I need help with my homework, it's about fractions",
            "Can I show you my homework paper?",
            "My homework is due tomorrow, can you help",
        ],
    )
    def test_matches_homework_requests(self, text):
        assert is_homework_help_request(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What is the area of a rectangle with width 6 and height 4?",
            "Can you help me with fractions?",
            "What's 5 plus 3?",
            "How's your day going?",
        ],
    )
    def test_does_not_match_normal_questions(self, text):
        assert is_homework_help_request(text) is False
