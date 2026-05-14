"""
Privacy Manager for PII anonymization.
Regex-based detection tuned for tutoring speech-to-text output.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns — applied in order, more specific first.
# Speech-to-text output is often lowercase, so most patterns use IGNORECASE.
# Patterns that need to preserve surrounding context words use a capture group
# in the replacement (e.g. r'\1[TOKEN]').
# ---------------------------------------------------------------------------
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Email addresses
    (
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        '[EMAIL]',
    ),
    # Indian mobile numbers: 10 digits starting with 6-9, optional separator after 5th digit
    # Placed before the generic phone pattern so "98765-43210" isn't split into two postal codes
    (
        re.compile(r'\b[6-9]\d{4}[-.\s]?\d{5}\b'),
        '[PHONE]',
    ),
    # Phone numbers — US (###-###-####) and international (+1 987-654-3210)
    # Uses (?<![A-Za-z0-9]) instead of \b so the leading + is consumed in the match
    (
        re.compile(r'(?<![A-Za-z0-9])\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        '[PHONE]',
    ),
    # Roll / student ID when explicitly labelled
    # "is" allowed as separator: "roll number is 12345", "student id A001", "admission no: RN2023"
    (
        re.compile(
            r'\b(?:roll|student|admission|enrollment|reg(?:istration)?)\s*'
            r'(?:number|no|num|#|id)?\s*(?:is\s+|[:\-]\s*)?[A-Z0-9][A-Z0-9\-]{2,11}\b',
            re.IGNORECASE,
        ),
        '[STUDENT_ID]',
    ),
    # Standalone alphanumeric IDs: STU001, RN12345, 2023-0456
    # Requires ≥2 letter prefix OR year-slash format to avoid matching plain numbers
    (
        re.compile(r'\b(?:[A-Z]{2,4}\d{3,8}|\d{4}[-/]\d{3,6})\b'),
        '[STUDENT_ID]',
    ),
    # Teacher / staff names preceded by an honorific
    # Catches both "Mr. Smith" and speech-to-text "mr smith"
    (
        re.compile(
            r'\b(Mr|Mrs|Ms|Miss|Dr|Prof|Sir)\.?\s+[A-Za-z]+(?:\s+[A-Za-z]+)?\b',
            re.IGNORECASE,
        ),
        '[TEACHER_NAME]',
    ),
    # Parent names following a relationship word
    # Captures exactly one name word to avoid eating the next verb ("picks", "said", etc.)
    # e.g. "my mom Sarah", "my dad's name is Ravi"
    (
        re.compile(
            r'(\b(?:my\s+)?(?:mom|dad|mother|father|parent|guardian|mama|papa|mum|daddy|mommy)(?:\'s)?\s+(?:name\s+is\s+)?)([A-Za-z]{2,20})',
            re.IGNORECASE,
        ),
        r'\1[PARENT_NAME]',
    ),
    # School names: a proper-looking name followed by a compound school-type suffix.
    # Negative lookahead blocks common function words (go, at, help, with, …) from being
    # treated as the start of a school name, preventing "help with high school math" matching.
    (
        re.compile(
            r'\b(?!(?:go|at|my|the|a|an|in|on|to|from|of|and|or|but|for|with|help|about|do|did|does|make|need|learn|study)\b)'
            r'[A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20}){0,2}\s+'
            r'(?:High\s+School|Middle\s+School|Elementary\s+School|'
            r'Public\s+School|Primary\s+School|International\s+School|'
            r'Academy|Institute|University|Kindergarten|Preschool)\b',
            re.IGNORECASE,
        ),
        '[SCHOOL_NAME]',
    ),
    # Street addresses: house number + street name + street-type word
    # Covers common US types and Indian suffixes (Nagar, Colony, Layout, etc.)
    (
        re.compile(
            r'\b\d+[,\s]+[A-Za-z\s]{3,30}'
            r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|'
            r'Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir|'
            r'Nagar|Colony|Layout|Extension|Ext)\.?\b',
            re.IGNORECASE,
        ),
        '[ADDRESS]',
    ),
    # Postal / ZIP codes
    # Context-required patterns prevent false positives on math numbers.
    # Match "zip/postal/pin (code) (is) 12345" or the distinctive ZIP+4 hyphen format.
    (
        re.compile(
            r'\b(?:zip|postal|pin)\s*(?:code)?\s*(?:is\s*)?\d{5,6}(?:-\d{4})?\b'
            r'|\b\d{5}-\d{4}\b',  # ZIP+4 hyphen format is distinctive enough without context
            re.IGNORECASE,
        ),
        '[POSTAL_CODE]',
    ),
]


class PrivacyManager:
    """Anonymizes PII in student speech before it is sent to the LLM."""

    def __init__(self):
        logger.info("PrivacyManager initialized")

    def anonymize(self, text: str) -> str:
        """Replace detected PII with labelled tokens. Student name is not touched."""
        result = text
        for pattern, replacement in _PATTERNS:
            result = pattern.sub(replacement, result)
        if result != text:
            logger.debug("PII anonymized | original=%r anonymized=%r", text, result)
        return result

    def deanonymize(self, text: str) -> str:
        """No-op: LLM responses do not echo tokens that need restoring."""
        return text

    def clear_mappings(self) -> None:
        pass

    def get_anonymization_count(self) -> int:
        return 0


if __name__ == "__main__":
    manager = PrivacyManager()

    cases = [
        # (input, expected_token)
        ("my email is kid@gmail.com",                    "[EMAIL]"),
        ("call me on 98765-43210",                       "[PHONE]"),
        ("call me at +1 987-654-3210",                   "[PHONE]"),
        ("my roll number is 20230456",                   "[STUDENT_ID]"),
        ("my student id is STU007",                      "[STUDENT_ID]"),
        ("my admission no: RN2023",                      "[STUDENT_ID]"),
        ("my teacher is Mr. Ramesh Kumar",               "[TEACHER_NAME]"),
        ("taught by Dr Singh",                           "[TEACHER_NAME]"),
        ("my mom Sarah picks me up",                     "[PARENT_NAME]"),
        ("my dad's name is Ravi Sharma",                 "[PARENT_NAME]"),
        ("i go to Greenwood Elementary School",          "[SCHOOL_NAME]"),
        ("i study at Springfield Academy",               "[SCHOOL_NAME]"),
        ("i live at 42 MG Road",                         "[ADDRESS]"),
        ("my address is 12 Oak Avenue",                  "[ADDRESS]"),
        ("my pin code is 400001",                        "[POSTAL_CODE]"),
        ("zip is 90210",                                 "[POSTAL_CODE]"),
        # Should NOT change
        ("two plus two is four",                         "(no change)"),
        ("i need help with high school math",            "(no change)"),
        ("what is the area of a rectangle",              "(no change)"),
    ]

    print(f"{'Input':<45}  {'Expected':<15}  {'Got'}")
    print("-" * 100)
    for text, expected in cases:
        result = manager.anonymize(text)
        token_hit = expected in result if expected != "(no change)" else result == text
        status = "OK" if token_hit else "FAIL"
        print(f"{text:<45}  {expected:<15}  {result}  [{status}]")
