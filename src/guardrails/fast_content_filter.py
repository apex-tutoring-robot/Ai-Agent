"""
FastContentFilter - a synchronous, near-zero-latency keyword/pattern
check run BEFORE text is queued for TTS, layered in FRONT of
GuardrailsManager's existing async NeMo check (see JarvisBot.
_run_output_safety_check).

This exists because the async check is a real ~8s LLM round-trip that
intentionally runs AFTER audio has already started playing (to avoid
delaying time-to-first-audio) - meaning a child can hear most or all of
an unsafe response before that check can act. This fast layer can't
replace the async check's semantic judgment (it's a blocklist, not an
LLM), but it closes the worst of that exposure window for anything crude
enough to match a keyword, at effectively zero added latency.

Deliberately narrow today: wired into _run_teaching_turn's per-step
speech (already discrete, complete text units - not token-streamed) and
into the comprehension-check evaluation response. NOT wired into the
token-streamed general chat path (generate_response_with_tools) - safely
intercepting a live token stream without reintroducing the latency this
architecture was built to avoid is a larger, separate change, not
something to bolt on here.
"""

import re
from typing import Optional, Tuple

# Deliberately crude and conservative - a fast net for the worst cases,
# not a replacement for NeMo's semantic judgment. A false positive here
# just means an occasional unnecessary refusal on benign content that
# happens to contain one of these words; a false negative is still
# caught by the slower async check afterward, same as before this filter
# existed.
_BLOCKED_PATTERNS = [
    re.compile(r"\b(kill (yourself|myself)|suicide|self[\s-]harm)\b", re.IGNORECASE),
    # Weapon-building instructions - checked in both word orders ("how to
    # build a bomb" and "bomb-building instructions") since a single
    # ordered pattern missed the first live test case that came up.
    re.compile(r"\b(gun|weapon|bomb)s?\b.{0,30}\b(build|make|instructions)\b", re.IGNORECASE),
    re.compile(r"\b(build|make|instructions)\b.{0,30}\b(gun|weapon|bomb)s?\b", re.IGNORECASE),
    re.compile(r"\b(sex|sexual|porn|naked)\b", re.IGNORECASE),
    re.compile(r"\b(fuck|shit|bullshit|bitch|asshole)\b", re.IGNORECASE),
]

REFUSAL_MESSAGE = "I'm not sure how to answer that safely. Let's get back to learning! Do you have a math question?"


class FastContentFilter:
    """Synchronous keyword/pattern check - see module docstring."""

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Returns (safe, refusal_message). safe=False means the caller
        should skip synthesizing/speaking `text` and speak
        refusal_message instead.
        """
        if not text:
            return True, None
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(text):
                return False, REFUSAL_MESSAGE
        return True, None
