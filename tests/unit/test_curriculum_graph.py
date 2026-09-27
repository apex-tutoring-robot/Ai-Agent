"""
Tests for curriculum.graph.CurriculumGraph - a starter-scaffold
prerequisite graph (see the module docstring), not an authoritative
curriculum. Tests cover the traversal logic against the seeded concepts,
not the correctness of the seed data itself.
"""

from curriculum.graph import CurriculumGraph


class TestPrerequisites:
    def test_known_concept_returns_its_prerequisites(self):
        graph = CurriculumGraph()
        assert graph.get_prerequisites("area_rectangle") == ["perimeter_rectangle"]

    def test_concept_with_no_prerequisites_returns_empty_list(self):
        graph = CurriculumGraph()
        assert graph.get_prerequisites("counting_to_20") == []

    def test_unknown_concept_returns_empty_list(self):
        graph = CurriculumGraph()
        assert graph.get_prerequisites("not_a_real_concept") == []


class TestPrerequisitesMet:
    def test_true_when_all_prerequisites_are_mastered(self):
        graph = CurriculumGraph()
        assert graph.prerequisites_met("area_rectangle", mastered={"perimeter_rectangle"}) is True

    def test_false_when_a_prerequisite_is_missing(self):
        graph = CurriculumGraph()
        assert graph.prerequisites_met("area_rectangle", mastered=set()) is False

    def test_unknown_concept_is_trivially_ready(self):
        graph = CurriculumGraph()
        assert graph.prerequisites_met("not_a_real_concept", mastered=set()) is True


class TestSuggestNext:
    def test_suggests_a_concept_with_no_prerequisites_first(self):
        graph = CurriculumGraph()
        suggestion = graph.suggest_next(grade=None, mastered=set())
        assert graph.get_prerequisites(suggestion) == [] or graph.prerequisites_met(suggestion, set())

    def test_does_not_suggest_an_already_mastered_concept(self):
        graph = CurriculumGraph()
        mastered = {"counting_to_20", "addition_within_20", "place_value_tens_ones",
                    "perimeter_rectangle", "area_rectangle", "area_triangle", "area_circle",
                    "equivalent_fractions", "fraction_addition"}
        suggestion = graph.suggest_next(grade=None, mastered=mastered)
        assert suggestion not in mastered

    def test_does_not_suggest_a_concept_with_unmet_prerequisites(self):
        graph = CurriculumGraph()
        suggestion = graph.suggest_next(grade=None, mastered=set())
        assert graph.prerequisites_met(suggestion, set())

    def test_returns_none_when_everything_is_mastered(self):
        graph = CurriculumGraph()
        all_concepts = {
            "counting_to_20", "addition_within_20", "place_value_tens_ones",
            "perimeter_rectangle", "area_rectangle", "area_triangle", "area_circle",
            "equivalent_fractions", "fraction_addition", "ratios_and_proportions",
            "solving_two_step_equations", "pythagorean_theorem",
        }
        assert graph.suggest_next(grade=None, mastered=all_concepts) is None

    def test_prefers_a_concept_at_or_below_the_students_grade(self):
        graph = CurriculumGraph()
        suggestion = graph.suggest_next(grade="3", mastered=set())
        node = graph.get_node(suggestion)
        assert node.grade == "K" or node.grade <= "3"
