"""
Tests for privacy.privacy_manager - regex-based PII anonymization.
Mirrors the module's own __main__ self-test cases as real pytest cases,
plus additional live-conversation-shaped cases (profile names, grade
answers, volume commands, math content) that must NOT be touched, since a
false positive here would corrupt what actually gets sent to the LLM.
"""

import pytest

from privacy.privacy_manager import PrivacyManager


@pytest.fixture
def pm():
    return PrivacyManager()


class TestDetectsPII:
    @pytest.mark.parametrize(
        "text,expected_token",
        [
            ("my email is kid@gmail.com", "[EMAIL]"),
            ("call me on 98765-43210", "[PHONE]"),
            ("call me at +1 987-654-3210", "[PHONE]"),
            ("my roll number is 20230456", "[STUDENT_ID]"),
            ("my student id is STU007", "[STUDENT_ID]"),
            ("my admission no: RN2023", "[STUDENT_ID]"),
            ("my teacher is Mr. Ramesh Kumar", "[TEACHER_NAME]"),
            ("taught by Dr Singh", "[TEACHER_NAME]"),
            ("my mom Sarah picks me up", "[PARENT_NAME]"),
            ("my dad's name is Ravi Sharma", "[PARENT_NAME]"),
            ("i go to Greenwood Elementary School", "[SCHOOL_NAME]"),
            ("i study at Springfield Academy", "[SCHOOL_NAME]"),
            ("i live at 42 MG Road", "[ADDRESS]"),
            ("my address is 12 Oak Avenue", "[ADDRESS]"),
            ("my pin code is 400001", "[POSTAL_CODE]"),
            ("zip is 90210", "[POSTAL_CODE]"),
        ],
    )
    def test_replaces_pii_with_token(self, pm, text, expected_token):
        assert expected_token in pm.anonymize(text)


class TestLeavesNonPIIUnchanged:
    @pytest.mark.parametrize(
        "text",
        [
            "two plus two is four",
            "i need help with high school math",
            "what is the area of a rectangle",
            # Live-conversation-shaped cases specific to this project - must
            # not be touched or the LLM/profile-matching pipeline breaks.
            "Voice Recognition Ryan Lewis",
            "third grade",
            "the answer is 42",
            "pi is about 3.14159",
            "please talk more loudly",
            "what is the area of a triangle with base 10 and height 5",
            "can you help me with fractions like 1/2 plus 1/4",
        ],
    )
    def test_no_change(self, pm, text):
        assert pm.anonymize(text) == text


class TestApiCompatibility:
    """anonymize() is the only method main.py actually calls, but the
    others are part of the public interface the placeholder it replaced
    also had - keep them working the same shape."""

    def test_deanonymize_is_a_noop(self, pm):
        text = "some text with [EMAIL] in it"
        assert pm.deanonymize(text) == text

    def test_clear_mappings_does_not_raise(self, pm):
        pm.clear_mappings()

    def test_get_anonymization_count_returns_int(self, pm):
        assert isinstance(pm.get_anonymization_count(), int)
