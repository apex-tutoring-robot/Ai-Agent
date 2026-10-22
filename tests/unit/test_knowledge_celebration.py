"""
Tests for JarvisBot._build_knowledge_celebration() - the brain-to-
knowledge-base celebration played after a curiosity-question teaching
turn's explanation (see main.py's _run_teaching_turn, gated on
is_curiosity_question).

Doesn't touch any `self` state, so it's called unbound (JarvisBot.
_build_knowledge_celebration(None, text)) rather than constructing a real
JarvisBot - same pattern as test_is_math_query.py.
"""

import pytest

from main import JarvisBot


def build_knowledge_celebration(concept: str):
    return JarvisBot._build_knowledge_celebration(None, concept)


class TestBuildKnowledgeCelebration:
    def test_returns_actions_and_phrase(self):
        actions, phrase = build_knowledge_celebration("photosynthesis")
        assert isinstance(actions, list)
        assert isinstance(phrase, str) and phrase

    def test_phrase_names_the_concept(self):
        _, phrase = build_knowledge_celebration("photosynthesis")
        assert "photosynthesis" in phrase.lower()

    def test_phrase_uses_readable_concept_not_snake_case(self):
        _, phrase = build_knowledge_celebration("water_cycle")
        assert "water_cycle" not in phrase
        assert "water cycle" in phrase.lower()

    def test_draws_a_book_and_a_brain_icon(self):
        actions, _ = build_knowledge_celebration("fractions")
        icon_actions = [a for a in actions if a.get("action") == "draw_icon"]
        icons = {a["icon"] for a in icon_actions}
        assert icons == {"book", "brain"}

    def test_brain_icon_has_sparkle_and_glow_flags(self):
        actions, _ = build_knowledge_celebration("fractions")
        brain = next(a for a in actions if a.get("action") == "draw_icon" and a["icon"] == "brain")
        assert brain["sparkle_trail"] is True
        assert brain["glow_on_arrival"] is True

    def test_book_icon_has_no_sparkle_or_glow_flags(self):
        # The book is static - the moving brain carries both flags and
        # the glow burst renders at ITS destination, which happens to be
        # where the book sits (see teaching_canvas.py's
        # _paint_celebration_effects). The book itself needs neither flag.
        actions, _ = build_knowledge_celebration("fractions")
        book = next(a for a in actions if a.get("action") == "draw_icon" and a["icon"] == "book")
        assert book.get("sparkle_trail", False) is False
        assert book.get("glow_on_arrival", False) is False

    def test_animates_the_brain_toward_the_book(self):
        actions, _ = build_knowledge_celebration("fractions")
        move = next(a for a in actions if a.get("action") == "animate_icon")
        assert move["id"] == "celebration_brain"
        assert move["animation"] == "move"
        assert move["to_x"] > 0

    def test_clears_the_board_first(self):
        actions, _ = build_knowledge_celebration("fractions")
        assert actions[0] == {"action": "clear"}

    @pytest.mark.parametrize("_run", range(20))
    def test_phrase_always_comes_from_the_template_list(self, _run):
        # Templates are chosen randomly for variety - run several times
        # to exercise more than just whichever one random.choice picks first.
        _, phrase = build_knowledge_celebration("gravity")
        assert any(
            t.format(concept="gravity", concept_title="Gravity") == phrase
            for t in JarvisBot._KNOWLEDGE_CELEBRATION_TEMPLATES
        )
