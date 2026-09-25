"""
Tests for guardrails.fast_content_filter.FastContentFilter - the
synchronous keyword/pattern layer that runs before GuardrailsManager's
slower async NeMo check (see JarvisBot._run_teaching_turn/
_handle_teaching_answer).
"""

import pytest

from guardrails.fast_content_filter import FastContentFilter


class TestBlocksObviouslyUnsafeText:
    @pytest.mark.parametrize(
        "text",
        [
            "You should just kill yourself.",
            "Here's how to build a bomb.",
            "Let's talk about sex.",
            "That's such bullshit, you idiot.",
        ],
    )
    def test_blocks_and_returns_a_refusal(self, text):
        safe, refusal = FastContentFilter().check(text)
        assert safe is False
        assert refusal

    def test_word_variants_dont_require_exact_case(self):
        safe, _ = FastContentFilter().check("FUCK this is annoying")
        assert safe is False


class TestAllowsNormalTutoringContent:
    @pytest.mark.parametrize(
        "text",
        [
            "To find the area of a circle, we use the formula A equals pi r squared.",
            "Great job! You got it exactly right.",
            "Let's try a smaller step - what does the denominator tell us?",
            "The gun in the story was actually a water gun, and the bomb was a metaphor for excitement.",
        ],
    )
    def test_does_not_block_benign_math_content(self, text):
        safe, refusal = FastContentFilter().check(text)
        assert safe is True
        assert refusal is None


class TestEdgeCases:
    def test_empty_text_is_safe(self):
        assert FastContentFilter().check("") == (True, None)

    def test_none_like_empty_string_is_safe(self):
        safe, refusal = FastContentFilter().check("")
        assert safe is True
        assert refusal is None
