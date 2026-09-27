"""
Tests for JarvisBot's _YES_PATTERN/_NO_PATTERN - the "is that you?"
confirmation classifier used by _handle_identity_checkin_step. A false
accept here (an unclear answer treated as "yes") would risk attaching a
conversation's learning history to the wrong student, so "no match at
all" (ambiguous) is the behavior actually being protected here - see
JarvisBot._classify_yes_no, which returns None (not a guess) in that case.
"""

import pytest

from main import JarvisBot


class TestRecognizesYes:
    @pytest.mark.parametrize(
        "text",
        ["yes", "yeah", "yep", "yup", "that's me", "thats me", "correct", "yeah, that's right", "uh huh"],
    )
    def test_matches_yes_pattern(self, text):
        assert JarvisBot._YES_PATTERN.search(text) is not None


class TestRecognizesNo:
    @pytest.mark.parametrize(
        "text",
        ["no", "nope", "nah", "not me", "no, that's wrong", "that's incorrect"],
    )
    def test_matches_no_pattern(self, text):
        assert JarvisBot._NO_PATTERN.search(text) is not None


class TestAmbiguousAnswersMatchNeither:
    @pytest.mark.parametrize(
        "text",
        ["David", "I don't know", "maybe", "what?"],
    )
    def test_matches_neither_pattern(self, text):
        assert JarvisBot._YES_PATTERN.search(text) is None
        assert JarvisBot._NO_PATTERN.search(text) is None
