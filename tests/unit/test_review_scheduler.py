"""
Tests for tutor.review_scheduler.ReviewScheduler - uses fake
profile_manager/llm_client doubles rather than real ones, since this
class is pure orchestration (select a weak concept, ask for a fresh
question) with no logic of its own to verify against a real DB/API.
"""

from tutor.review_scheduler import ReviewScheduler


class FakeProfileManager:
    def __init__(self, weak_concept=None):
        self._weak_concept = weak_concept
        self.last_exclude = None

    def get_weak_concept_for_review(self, profile_id, exclude=frozenset()):
        self.last_exclude = exclude
        return self._weak_concept


class FakeLLMClient:
    def __init__(self, fresh_question="What is 6/8 in simplest form?"):
        self._fresh_question = fresh_question
        self.calls = []

    def generate_review_question(self, concept, previous_question):
        self.calls.append((concept, previous_question))
        return self._fresh_question


class TestGetReview:
    def test_returns_none_when_nothing_to_review(self):
        scheduler = ReviewScheduler(FakeProfileManager(weak_concept=None), FakeLLMClient())
        assert scheduler.get_review(profile_id=1) is None

    def test_returns_concept_with_a_freshly_generated_question(self):
        weak = {"concept": "equivalent_fractions", "question": "What is 2/4 equal to?"}
        llm = FakeLLMClient(fresh_question="What is 3/6 equal to?")
        scheduler = ReviewScheduler(FakeProfileManager(weak_concept=weak), llm)

        review = scheduler.get_review(profile_id=1)

        assert review == {"concept": "equivalent_fractions", "question": "What is 3/6 equal to?"}

    def test_asks_the_llm_using_the_stored_question_as_context(self):
        weak = {"concept": "area_rectangle", "question": "Area of a rectangle 4 by 3?"}
        llm = FakeLLMClient()
        scheduler = ReviewScheduler(FakeProfileManager(weak_concept=weak), llm)

        scheduler.get_review(profile_id=1)

        assert llm.calls == [("area_rectangle", "Area of a rectangle 4 by 3?")]

    def test_forwards_exclude_set_to_profile_manager(self):
        pm = FakeProfileManager(weak_concept=None)
        scheduler = ReviewScheduler(pm, FakeLLMClient())

        scheduler.get_review(profile_id=1, exclude={"area_rectangle"})

        assert pm.last_exclude == {"area_rectangle"}
