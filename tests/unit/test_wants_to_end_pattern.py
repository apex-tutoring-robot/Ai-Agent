"""
Tests for JarvisBot._WANTS_TO_END_PATTERN - a clear "I want to leave now"
signal ("I'm done", "let's finish here", "talk later", "bye"), distinct
from _DISCOVERY_DECLINE_PATTERN (declining a specific offer of help).

Checked in _handle_teaching_answer (so a pending comprehension check
actually ends the conversation instead of just acknowledging the exit
and leaving the question open), and alongside _DISCOVERY_DECLINE_PATTERN
in _handle_warmup_step's discovery branch and
_handle_proactive_suggestion_reply.

Regression test for a real production bug: with a check question
pending, "I'm done" / "let's talk later" / "let's finish here" were each
correctly NOT graded as wrong answers (see TutorAction.NOT_AN_ANSWER),
but then just got acknowledged with "that's okay, let me know when
you're ready" and left the conversation open - repeated three times in a
row in the same real conversation without the student ever actually
getting to leave.
"""

import pytest

from main import JarvisBot


class TestRecognizedAsWantingToEnd:
    @pytest.mark.parametrize(
        "text",
        [
            "I'm done",
            "I am done",
            "OK, uh, I'm done.",
            "let's finish",
            "let's finish here",
            "let's stop",
            "let's end",
            "talk later",
            "talk to you later",
            "chat later",
            "Now let's talk later.",
            "bye",
            "goodbye",
            "good bye",
            "see you later",
            "I'm going to go",
            "I'm gonna go",
            "I'm about to go",
            "gotta go",
            "got to go",
            "have to go",
            "need to go",
        ],
    )
    def test_recognized(self, text):
        assert JarvisBot._WANTS_TO_END_PATTERN.search(text) is not None


class TestDoesNotFalsePositiveOnRealAnswersOrContent:
    @pytest.mark.parametrize(
        "text",
        [
            "3/8",
            "is it 24",
            "I don't know",
            "can I try again",
            "I want to know about fractions",
            "help me with multiplication",
            "the homework is done",  # "done" alone, not "i'm done"/"i am done"
        ],
    )
    def test_not_recognized(self, text):
        assert JarvisBot._WANTS_TO_END_PATTERN.search(text) is None
