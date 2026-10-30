"""
Deterministic fraction-bar layout for the whiteboard's fraction_bar draw
action - the same "LLM picks WHAT (numerator/denominator), Python
computes HOW (exact segment pixel boundaries)" pattern as
visuals.arithmetic and visuals.number_line. The LLM never computes a
pixel boundary for an individual segment - it only supplies the
numerator(s) and denominator(s), and every segment's exact left/right
edge is derived here by dividing the bar's pixel width evenly.

Scope: one or more horizontal bars, each divided into `denominator`
equal segments with `numerator` of them shaded, stacked vertically -
covers plotting a single fraction, comparing two or more fractions, and
equivalent-fraction pairs (same bar width, different denominators).
Improper fractions/mixed numbers (numerator > denominator, needing more
than one "whole" bar) are out of scope for this first slice.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIN_DENOMINATOR = 1
MAX_DENOMINATOR = 12
MAX_BARS = 4


@dataclass
class FractionBarSegment:
    x0: float  # pixel offset from the bar's own left edge
    x1: float
    shaded: bool


@dataclass
class FractionBar:
    numerator: int
    denominator: int
    label: str
    segments: List[FractionBarSegment] = field(default_factory=list)


@dataclass
class FractionBarLayout:
    """Everything the renderer needs, already computed - no further math
    or coordinate decisions left for teaching_canvas.py to make. Bars are
    meant to be stacked vertically by the renderer, one per row."""
    width: float
    bar_height: float
    bars: List[FractionBar] = field(default_factory=list)


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_fraction_bar_spec(fractions: List[Dict], width, bar_height) -> Optional[str]:
    """Returns an error message if the spec is invalid, or None. Checked
    BEFORE any computation - mathematical validation, not just a
    JSON-shape check."""
    if not isinstance(width, (int, float)) or isinstance(width, bool) or width <= 0:
        return "width must be a positive number."
    if not isinstance(bar_height, (int, float)) or isinstance(bar_height, bool) or bar_height <= 0:
        return "bar_height must be a positive number."
    if not isinstance(fractions, list) or len(fractions) == 0:
        return "fraction_bar needs at least 1 fraction."
    if len(fractions) > MAX_BARS:
        return f"fraction_bar supports at most {MAX_BARS} bars at once (for legibility)."
    for f in fractions:
        if not isinstance(f, dict) or "numerator" not in f or "denominator" not in f:
            return "Each fraction needs a 'numerator' and 'denominator'."
        n, d = f["numerator"], f["denominator"]
        if not _is_plain_int(n) or not _is_plain_int(d):
            return "numerator and denominator must be plain integers."
        if n < 0:
            return "numerator must be non-negative."
        if not (MIN_DENOMINATOR <= d <= MAX_DENOMINATOR):
            return f"denominator must be between {MIN_DENOMINATOR} and {MAX_DENOMINATOR}."
        if n > d:
            return (
                f"numerator ({n}) can't be greater than denominator ({d}) - "
                "fraction_bar only supports proper fractions (or exactly 1 whole)."
            )
    return None


def build_fraction_bar_layout(fractions: List[Dict], width, bar_height=50) -> FractionBarLayout:
    """Raises ValueError if the spec fails validate_fraction_bar_spec() -
    callers should validate first and treat a raised error as "don't
    render this", same as any other malformed draw action."""
    error = validate_fraction_bar_spec(fractions, width, bar_height)
    if error:
        raise ValueError(error)

    bars = []
    for f in fractions:
        n, d = f["numerator"], f["denominator"]
        seg_w = width / d
        segments = [
            FractionBarSegment(x0=i * seg_w, x1=(i + 1) * seg_w, shaded=(i < n))
            for i in range(d)
        ]
        label = str(f["label"]) if f.get("label") is not None else f"{n}/{d}"
        bars.append(FractionBar(numerator=n, denominator=d, label=label, segments=segments))

    return FractionBarLayout(width=width, bar_height=bar_height, bars=bars)
