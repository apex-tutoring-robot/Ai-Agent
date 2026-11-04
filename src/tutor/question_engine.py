"""
ProactiveQuestionEngine - lets Jarvis suggest something to work on when a
student doesn't ask a specific question themselves (see JarvisBot.
_handle_warmup_step's discovery step), instead of only ever waiting to be
asked. Suggests a sensible "next" concept from CurriculumGraph - forward
progress, not a callout of past struggles.

Deliberately does NOT suggest the student's weakest/most-often-missed
concept (get_weak_concept_for_review) the way an earlier version of this
did - confirmed live as a real problem: concept_mastery accumulates
across every session for the life of a profile, including old dev/
testing attempts, so a concept that picked up a few wrong attempts early
on stays "the weakest" more or less permanently and keeps getting
proactively re-surfaced ("I remember that gave you some trouble") long
after it stopped reflecting anything real about the student. Once
spoken, that suggestion also sits in conversation history and can pull a
LATER, unrelated vague request ("I want to know about math") back toward
it too, since generate_teaching_plan sees the full history. Retrieval
practice (tutor/review_scheduler.py, used mid-conversation right after a
genuine success - see JarvisBot._handle_teaching_answer) is a different,
narrower mechanism and is unaffected by this.
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
        (no grade on file and CurriculumGraph has nothing left to offer).
        """
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
