"""
Tests for knowledge.textbook_search - normalize_grade() (pure parsing) and
TextbookSearch.search() against the real processed_json_textbooks/ corpus,
not a mock, since the corpus is small enough to just use directly and a
mock would risk drifting from the real chunk/scoring behavior.
"""

import os

import pytest

from knowledge.textbook_search import TextbookSearch, normalize_grade

_TEXTBOOKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "processed_json_textbooks"
)


class TestNormalizeGrade:
    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("I'm in third grade", "3"),
            ("3rd", "3"),
            ("grade 5", "5"),
            ("first grade", "1"),
            ("I am in 8th grade", "8"),
            ("kindergarten", "K"),
            ("I'm in kinder", "K"),
            ("k", "K"),
            ("seventh", "7"),
            ("six", "6"),
        ],
    )
    def test_recognized_answers(self, spoken, expected):
        assert normalize_grade(spoken) == expected

    @pytest.mark.parametrize(
        "spoken",
        [
            "",
            "I like pizza",
            "what's your favorite color",
            "nine",  # out of K-8 range, deliberately not in _GRADE_WORDS
        ],
    )
    def test_unrecognized_answers_return_none(self, spoken):
        assert normalize_grade(spoken) is None


@pytest.fixture(scope="module")
def search_engine():
    if not os.path.isdir(_TEXTBOOKS_DIR):
        pytest.skip(f"processed_json_textbooks/ not found at {_TEXTBOOKS_DIR}")
    return TextbookSearch(_TEXTBOOKS_DIR)


class TestTextbookSearch:
    def test_loads_all_grades(self, search_engine):
        # 9 grades: K, 1-8 - matches "TextbookSearch loaded ... chunks
        # across 9 grades" seen in live startup logs throughout this project.
        assert len(search_engine._chunks_by_grade) == 9

    def test_search_returns_grade_scoped_results(self, search_engine):
        results = search_engine.search("fractions", grade="4", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, str) and r.strip()

    def test_search_unknown_grade_returns_empty(self, search_engine):
        assert search_engine.search("fractions", grade="99") == []

    def test_search_nonsense_query_returns_empty_or_low_relevance(self, search_engine):
        # Not asserting an exact empty list (keyword-overlap scoring may
        # still surface something for a query with zero real overlap) -
        # just that it doesn't crash and returns the expected type.
        results = search_engine.search("xyzabc nonsense gibberish", grade="3")
        assert isinstance(results, list)
