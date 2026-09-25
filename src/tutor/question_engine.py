"""
ProactiveQuestionEngine - lets Jarvis suggest something to work on when a
student doesn't ask a specific question themselves (see JarvisBot.
_handle_warmup_step's discovery step), instead of only ever waiting to be
asked. Picks the student's weakest concept if they have review history,
or a sensible "next" concept from CurriculumGraph otherwise.
"""

import logging
from typing import Optional

from curriculum.graph import CurriculumGraph

logger = logging.getLogger(__name__)


class ProactiveQuestionEngine:
    def __init__(self, profile_manager, curriculum_graph: Optional[CurriculumGraph] = None):
        self._profile_manager = profile_manager
        self._curriculum_graph = curriculum_graph or CurriculumGraph()

    def suggest(self, profile_id: int) -> Optional[dict]:
        """
        Returns {"concept", "prompt"} - a concept to suggest and a short
        spoken invitation - or None if nothing sensible can be suggested
        (a new student with no grade on file and no review history).
        """
        weak = self._profile_manager.get_weak_concept_for_review(profile_id)
        if weak:
            readable = weak["concept"].replace("_", " ")
            return {
                "concept": weak["concept"],
                "prompt": f"Want to practice {readable} together? I remember that one gave you some trouble.",
            }

        grade = self._profile_manager.get_grade(profile_id)
        mastered = self._profile_manager.get_mastered_concepts(profile_id)
        next_concept = self._curriculum_graph.suggest_next(grade, mastered=mastered)
        if not next_concept:
            return None
        readable = next_concept.replace("_", " ")
        return {
            "concept": next_concept,
            "prompt": f"Want to try something new today? We could work on {readable}.",
        }
