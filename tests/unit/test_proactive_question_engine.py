"""
Tests for tutor.question_engine.ProactiveQuestionEngine - uses a fake
profile_manager rather than a real one, since this class is pure
orchestration (a sensible "next" concept from CurriculumGraph).

No longer tests a "prefers the weakest concept" branch - that was
removed after a real production problem: concept_mastery accumulates for
the life of a profile, so an old wrong attempt (even from early dev/
testing) could keep a concept looking "weak" indefinitely, and the
resulting "I remember that gave you some trouble" callout kept getting
proactively re-surfaced long after it stopped meaning anything real -
see question_engine.py's module docstring.
"""

from curriculum.graph import CurriculumGraph
from tutor.question_engine import ProactiveQuestionEngine


class FakeProfileManager:
    def __init__(self, grade=None, mastered=None):
        self._grade = grade
        self._mastered = mastered or set()

    def get_grade(self, profile_id):
        return self._grade

    def get_mastered_concepts(self, profile_id):
        return self._mastered


class TestSuggest:
    def test_suggests_the_next_curriculum_concept(self):
        engine = ProactiveQuestionEngine(FakeProfileManager(grade="3"), CurriculumGraph())

        suggestion = engine.suggest(profile_id=1)

        assert suggestion is not None
        assert "concept" in suggestion and "prompt" in suggestion

    def test_prompt_is_forward_looking_not_a_past_struggle_callout(self):
        engine = ProactiveQuestionEngine(FakeProfileManager(grade="3"), CurriculumGraph())
        suggestion = engine.suggest(profile_id=1)
        assert "trouble" not in suggestion["prompt"].lower()
        assert "remember" not in suggestion["prompt"].lower()

    def test_returns_none_when_nothing_to_suggest(self):
        class ExhaustedGraph(CurriculumGraph):
            def suggest_next(self, grade, mastered):
                return None

        engine = ProactiveQuestionEngine(FakeProfileManager(grade="8"), ExhaustedGraph())
        assert engine.suggest(profile_id=1) is None

    def test_does_not_suggest_an_already_mastered_concept(self):
        engine = ProactiveQuestionEngine(
            FakeProfileManager(grade="3", mastered={"perimeter_rectangle"}),
            CurriculumGraph(),
        )
        suggestion = engine.suggest(profile_id=1)
        assert suggestion["concept"] != "perimeter_rectangle"
