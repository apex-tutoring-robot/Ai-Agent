"""
ReviewScheduler - picks and prepares a retrieval-practice question for the
student's weakest older concept (see JarvisBot._handle_teaching_answer,
called once per conversation right after a genuine success).

Scoped deliberately narrow today: it wraps ProfileManager.
get_weak_concept_for_review() (selection logic unchanged) and generates a
FRESH question via LLMClient.generate_review_question() instead of
literally replaying the stored last_question - a real repetition tests
whether the student memorized that exact question, not whether they still
understand the concept.

The full "70% current concept / 20% weak concepts / 10% spaced review"
session-time allocation from the architecture review is NOT implemented
here - that's a decision about what to TEACH next among competing
options, and there's nowhere for it to plug in yet: Jarvis has no
proactive question engine, so teaching is 100% student-initiated today,
and retrieval practice is the only thing ever scheduled at all. This
class is the seam a future scheduler would call into once that exists.
"""

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ReviewScheduler:
    def __init__(self, profile_manager, llm_client):
        self._profile_manager = profile_manager
        self._llm_client = llm_client

    def get_review(self, profile_id: int, exclude: Set[str] = frozenset()) -> Optional[dict]:
        """
        Returns {"concept", "question"} for the best review candidate,
        with a freshly generated question - or None if there's nothing to
        review (new student, or no stored question to work from - see
        ProfileManager.get_weak_concept_for_review).
        """
        weak = self._profile_manager.get_weak_concept_for_review(profile_id, exclude=exclude)
        if not weak:
            return None

        fresh_question = self._llm_client.generate_review_question(weak["concept"], weak["question"])
        return {"concept": weak["concept"], "question": fresh_question}
