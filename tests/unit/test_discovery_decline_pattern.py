"""
Tests for JarvisBot._DISCOVERY_DECLINE_PATTERN - use case 3 of the
discovery pivot: a CLEAR decline ("no", "I'm good") at main.py's
_handle_warmup_step discovery branch or _handle_proactive_suggestion_reply
ends the conversation with a warm goodbye (see _end_conversation_with_goodbye)
instead of still pushing a proactive suggestion or falling through to
generic chat.
"""

import pytest

from main import JarvisBot


class TestRecognizedAsClearDecline:
    @pytest.mark.parametrize(
        "text",
        [
            "no",
            "no thanks",
            "nope",
            "nah",
            "not right now",
            "not today",
            "nothing",
            "nothing right now",
            "nothing today",
            "I'm good",
            "I'm okay",
            "I'm ok",
            "I don't want anything",
            "I don't need any help",
            "I don't need anything",
        ],
    )
    def test_recognized_as_decline(self, text):
        assert JarvisBot._DISCOVERY_DECLINE_PATTERN.search(text) is not None


class TestDoesNotFalsePositiveOnRealRequestsOrAmbiguousReplies:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't know",  # ambiguous, not a clear decline - still gets a proactive suggestion
            "help me with fractions",
            "can you teach me about volcanoes",
            "what is the area of a rectangle",
            "I'm curious about dinosaurs",
        ],
    )
    def test_not_recognized_as_decline(self, text):
        assert JarvisBot._DISCOVERY_DECLINE_PATTERN.search(text) is None

    def test_word_boundary_does_not_match_no_inside_know(self):
        # Regression guard: \bno\b must not match the "no" substring
        # inside "know" - word-boundary regex semantics handle this
        # correctly (no transition between two word characters), but
        # it's exactly the kind of subtle gotcha worth pinning down.
        assert JarvisBot._DISCOVERY_DECLINE_PATTERN.search("I don't know") is None
