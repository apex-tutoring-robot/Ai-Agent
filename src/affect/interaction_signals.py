"""
InteractionSignalEngine - lightweight TEXT-based signals about how a
student seems to be doing, computed from what's already available (the
utterance text, attempt count) rather than real audio affect analysis.

Real prosody/hesitation-from-audio (pause timing, pitch, speaking rate)
would be a much larger, separate investment - Azure STT doesn't expose
that today, and building a real affect model is its own project. These
are heuristic SIGNALS meant to nudge a decision (see ExpressionController.
for_action's repeated_struggle parameter), not a verdict about the
student's actual emotional state - don't treat them as more certain than
they are.
"""

import re
from dataclasses import dataclass

# Deliberately simple - filler words and hedging language a K-8 student
# might actually say when unsure, not a sentiment model.
_UNCERTAIN_PATTERN = re.compile(r"\b(i don'?t know|not sure|maybe|i guess|um+|uh+)\b", re.IGNORECASE)
_SHORT_ANSWER_WORDS = 4


@dataclass
class InteractionSignals:
    hesitation: bool        # hedging/filler language in this utterance
    low_confidence: bool    # short AND hedging - "um, maybe?" vs. a full guess
    repeated_struggle: bool  # this is the 2nd+ attempt at the same question


class InteractionSignalEngine:
    def analyze(self, user_text: str, attempts_so_far: int) -> InteractionSignals:
        hesitation = bool(_UNCERTAIN_PATTERN.search(user_text))
        word_count = len(user_text.split())
        return InteractionSignals(
            hesitation=hesitation,
            low_confidence=hesitation and word_count <= _SHORT_ANSWER_WORDS,
            # attempts_so_far is 0 on a student's very first try at a
            # question and 1 once a hint has already been given and
            # they're trying again (the teaching loop moves to "reteach"
            # - which clears the pending question entirely - on a second
            # wrong attempt, so 1 is the highest value this ever actually
            # reaches in practice, not just a nominal threshold).
            repeated_struggle=attempts_so_far >= 1,
        )
