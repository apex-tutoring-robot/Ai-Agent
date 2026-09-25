"""
Tests for tutor.question_engine.ProactiveQuestionEngine - uses a fake
profile_manager rather than a real one, since this class is pure
orchestration (prefer a weak concept, fall back to CurriculumGraph).
"""

from curriculum.graph import CurriculumGraph
from tutor.question_engine import ProactiveQuestionEngine


class FakeProfileManager:
    def __init__(self, weak_concept=None, grade=None):
        self._weak_concept = weak_concept
        self._grade = grade

    def get_weak_concept_for_review(self, profile_id):
        return self._weak_concept

    def get_grade(self, profile_id):
        return self._grade


class TestSuggest:
    def test_prefers_a_weak_concept_when_one_exists(self):
        weak = {"concept": "equivalent_fractions", "question": "What is 2/4 equal to?"}
        engine = ProactiveQuestionEngine(FakeProfileManager(weak_concept=weak))

        suggestion = engine.suggest(profile_id=1)

        assert suggestion["concept"] == "equivalent_fractions"
        assert "equivalent fractions" in suggestion["prompt"]

    def test_falls_back_to_curriculum_graph_when_no_weak_concept(self):
        engine = ProactiveQuestionEngine(FakeProfileManager(weak_concept=None, grade="3"), CurriculumGraph())

        suggestion = engine.suggest(profile_id=1)

        assert suggestion is not None
        assert "concept" in suggestion and "prompt" in suggestion

    def test_returns_none_when_nothing_to_suggest(self):
        class ExhaustedGraph(CurriculumGraph):
            def suggest_next(self, grade, mastered):
                return None

        engine = ProactiveQuestionEngine(FakeProfileManager(weak_concept=None, grade="8"), ExhaustedGraph())
        assert engine.suggest(profile_id=1) is None
