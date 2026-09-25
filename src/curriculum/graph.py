"""
CurriculumGraph - a small prerequisite graph over the math concepts
Jarvis actually teaches (the same concept ids generate_teaching_plan()
already produces, e.g. "area_rectangle", "equivalent_fractions").

This is a STARTER SCAFFOLD, not an authoritative K-8 curriculum: the
concepts and edges below are hand-seeded from concepts already seen in
live testing, not sourced from a reviewed curriculum standard - real
curriculum review/expansion is a content-authoring task, not something to
invent wholesale in code. Still useful as-is: it gives
ProactiveQuestionEngine a real "what's a sensible next concept" answer
instead of nothing, and grows naturally as more concept ids are seen.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ConceptNode:
    concept: str
    grade: str  # 'K' or '1'-'8', matching knowledge.textbook_search.normalize_grade()
    prerequisites: List[str] = field(default_factory=list)


_CONCEPTS: Dict[str, ConceptNode] = {
    "perimeter_rectangle": ConceptNode("perimeter_rectangle", grade="3"),
    "area_rectangle": ConceptNode("area_rectangle", grade="3", prerequisites=["perimeter_rectangle"]),
    "area_triangle": ConceptNode("area_triangle", grade="4", prerequisites=["area_rectangle"]),
    "area_circle": ConceptNode("area_circle", grade="5", prerequisites=["area_rectangle"]),
    "equivalent_fractions": ConceptNode("equivalent_fractions", grade="4"),
    "fraction_addition": ConceptNode("fraction_addition", grade="5", prerequisites=["equivalent_fractions"]),
    "pythagorean_theorem": ConceptNode("pythagorean_theorem", grade="8", prerequisites=["area_triangle"]),
}


class CurriculumGraph:
    def get_node(self, concept: str) -> Optional[ConceptNode]:
        return _CONCEPTS.get(concept)

    def get_prerequisites(self, concept: str) -> List[str]:
        node = _CONCEPTS.get(concept)
        return list(node.prerequisites) if node else []

    def prerequisites_met(self, concept: str, mastered: Set[str]) -> bool:
        """True if every prerequisite for `concept` is already in
        `mastered`. A concept not in this graph has no prerequisites to
        check, so it's trivially considered ready - this graph is a
        starter scaffold, not exhaustive (see module docstring)."""
        return all(p in mastered for p in self.get_prerequisites(concept))

    def suggest_next(self, grade: Optional[str], mastered: Set[str]) -> Optional[str]:
        """
        Picks one concept a student is ready for: not already mastered
        and prerequisites satisfied, preferring one at or below their
        grade when grade is known (grades are single characters here, so
        plain string comparison matches numeric order for '1'-'8'; 'K'
        is left out of that comparison and always passes through).
        Returns None if every known concept is already mastered.
        """
        ready = [
            node for concept, node in _CONCEPTS.items()
            if concept not in mastered and self.prerequisites_met(concept, mastered)
        ]
        if not ready:
            return None
        if grade and grade.isdigit():
            at_grade = [n for n in ready if not n.grade.isdigit() or n.grade <= grade]
            ready = at_grade or ready
        return ready[0].concept
