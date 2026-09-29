"""
Deterministic number-line layout for the whiteboard's number_line draw
action - the same "LLM picks WHAT (values), Python computes HOW (exact
tick/point/arc pixel positions)" pattern as visuals.arithmetic. The LLM
never computes a pixel position for a tick mark, point, or jump arc
itself - it only supplies the integer range and the integer values to
plot, and every pixel coordinate is derived here by linear interpolation.

Scope: a single horizontal number line over a small integer range
(2-30 units, keeping tick marks legibly spaced on a fixed-width canvas),
with optional labeled points (e.g. "where is 7") and optional "jump"
arcs - the classic "start here, hop N, land there" way addition/
subtraction is taught on a number line.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIN_RANGE_SPAN = 2
MAX_RANGE_SPAN = 30


@dataclass
class NumberLinePoint:
    value: int
    label: str
    px: float  # pixel x-offset from the line's own left edge (0..width)


@dataclass
class NumberLineJump:
    from_value: int
    to_value: int
    label: str
    from_px: float
    to_px: float
    direction: int  # +1 (rightward/increasing) or -1 (leftward/decreasing)


@dataclass
class NumberLineLayout:
    """Everything the renderer needs, already computed - no further math
    or coordinate decisions left for teaching_canvas.py to make.

    tick_values/tick_px: one entry per integer from min_value to
    max_value inclusive, in matching order. px values are offsets from
    the line's own left edge (add the line's own x anchor when drawing).
    """
    min_value: int
    max_value: int
    width: float
    tick_values: List[int] = field(default_factory=list)
    tick_px: List[float] = field(default_factory=list)
    points: List[NumberLinePoint] = field(default_factory=list)
    jumps: List[NumberLineJump] = field(default_factory=list)


def _value_to_px(value: int, min_value: int, max_value: int, width: float) -> float:
    return (value - min_value) / (max_value - min_value) * width


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_number_line_spec(min_value, max_value, width, points: List[Dict], jumps: List[Dict]) -> Optional[str]:
    """Returns an error message if the spec is invalid, or None. Checked
    BEFORE any computation - mathematical/range validation, not just a
    JSON-shape check."""
    if not _is_plain_int(min_value):
        return "min must be a plain integer."
    if not _is_plain_int(max_value):
        return "max must be a plain integer."
    if min_value >= max_value:
        return "min must be less than max."
    span = max_value - min_value
    if span < MIN_RANGE_SPAN or span > MAX_RANGE_SPAN:
        return f"Range span must be between {MIN_RANGE_SPAN} and {MAX_RANGE_SPAN} (got {span})."
    if not isinstance(width, (int, float)) or isinstance(width, bool) or width <= 0:
        return "width must be a positive number."

    if not isinstance(points, list):
        return "points must be a list."
    for p in points:
        if not isinstance(p, dict) or "value" not in p:
            return "Each point needs a 'value'."
        if not _is_plain_int(p["value"]):
            return "Each point's value must be a plain integer."
        if not (min_value <= p["value"] <= max_value):
            return f"Point value {p['value']} is outside the range [{min_value}, {max_value}]."

    if not isinstance(jumps, list):
        return "jumps must be a list."
    for j in jumps:
        if not isinstance(j, dict) or "from" not in j or "to" not in j:
            return "Each jump needs 'from' and 'to'."
        fv, tv = j["from"], j["to"]
        if not _is_plain_int(fv) or not _is_plain_int(tv):
            return "Jump 'from'/'to' must be plain integers."
        if not (min_value <= fv <= max_value) or not (min_value <= tv <= max_value):
            return f"Jump values must be within the range [{min_value}, {max_value}]."
        if fv == tv:
            return "A jump needs 'from' and 'to' to be different values."
    return None


def build_number_line_layout(min_value, max_value, width, points: List[Dict] = None,
                              jumps: List[Dict] = None) -> NumberLineLayout:
    """Raises ValueError if the spec fails validate_number_line_spec() -
    callers should validate first and treat a raised error as "don't
    render this", same as any other malformed draw action."""
    points = points or []
    jumps = jumps or []
    error = validate_number_line_spec(min_value, max_value, width, points, jumps)
    if error:
        raise ValueError(error)

    tick_values = list(range(min_value, max_value + 1))
    tick_px = [_value_to_px(v, min_value, max_value, width) for v in tick_values]

    built_points = [
        NumberLinePoint(
            value=p["value"],
            label=str(p["label"]) if p.get("label") is not None else str(p["value"]),
            px=_value_to_px(p["value"], min_value, max_value, width),
        )
        for p in points
    ]

    built_jumps = []
    for j in jumps:
        fv, tv = j["from"], j["to"]
        delta = tv - fv
        default_label = f"+{delta}" if delta > 0 else str(delta)
        built_jumps.append(NumberLineJump(
            from_value=fv,
            to_value=tv,
            label=str(j["label"]) if j.get("label") is not None else default_label,
            from_px=_value_to_px(fv, min_value, max_value, width),
            to_px=_value_to_px(tv, min_value, max_value, width),
            direction=1 if tv > fv else -1,
        ))

    return NumberLineLayout(
        min_value=min_value,
        max_value=max_value,
        width=width,
        tick_values=tick_values,
        tick_px=tick_px,
        points=built_points,
        jumps=built_jumps,
    )
