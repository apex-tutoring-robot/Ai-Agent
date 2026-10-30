"""
Tests for JarvisBot.is_curiosity_question() - routes a general "what is X" /
"how does X work" / "I want to know about X" learning request on ANY K-8
subject to the teaching-plan pipeline (see main.py's _run_teaching_turn),
the same way is_math_query() does for math specifically.

Doesn't touch `self`, so it's called unbound (JarvisBot.
is_curiosity_question(None, text)) rather than constructing a real
JarvisBot - same pattern as test_is_math_query.py.
"""

import pytest

from main import JarvisBot


def is_curiosity_question(text: str) -> bool:
    return JarvisBot.is_curiosity_question(None, text)


class TestMatchesSubjectCuriosityQuestions:
    @pytest.mark.parametrize(
        "text",
        [
            "what is photosynthesis",
            "what are prime numbers",
            "what's gravity",
            "how does a volcano work",
            "how do plants grow",
            "why do leaves change color",
            "why is the sky blue",
            "tell me about volcanoes",
            "tell me about the water cycle",
            "explain the water cycle",
            "can you explain gravity",
            "I'm curious about dinosaurs",
            "I am curious about space",
            "I want to know about the solar system",
            "I want to learn about fractions",
            "teach me about the american revolution",
        ],
    )
    def test_matches(self, text):
        assert is_curiosity_question(text) is True


class TestDoesNotMatchQuestionsAboutJarvisItself:
    """
    Regression guard for a real false-positive found while building this:
    the trigger phrases above ("what is...", "tell me about...") also
    match ordinary small talk directed AT Jarvis ("what's your name",
    "tell me about yourself") - without excluding an immediately-following
    you/your/yourself, those would have been forced into a teaching-plan
    turn complete with a mandatory whiteboard diagram, instead of being
    treated as identity small talk.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "tell me about yourself",
            "what's your name",
            "what are you",
            "what is your favorite color",
            "how do you work",
            "how does your brain work",
            "why are you a robot",
            "explain yourself",
            "I'm curious about you",
            "I want to know about you",
            "teach me about yourself",
        ],
    )
    def test_does_not_match(self, text):
        assert is_curiosity_question(text) is False


class TestDoesNotMatchUnrelatedSmallTalk:
    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "I'm done",
            "yes",
            "it was good, we had a pizza party",
            "can we talk about something else",
        ],
    )
    def test_does_not_match(self, text):
        assert is_curiosity_question(text) is False
