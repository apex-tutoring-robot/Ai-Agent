"""
Deterministic column-arithmetic computation for the whiteboard's
vertical_arithmetic draw action - the first slice of "the LLM decides
WHAT to visualize, Python decides HOW" (validate mathematically, then
compute, rather than trusting the LLM to lay out columns/carries/borrows
or compute the result itself).

The LLM only ever supplies (operation, operands) - see generate_teaching_
plan's prompt. Every digit, carry mark, borrow mark, and the result
itself are computed here in plain Python, so the class of bug where an
LLM occasionally miscalculates arithmetic (already seen live in
evaluate_answer) simply can't happen for this whiteboard element.

Scope: addition and subtraction of any two non-negative integers, and
multiplication by a single-digit (0-9) multiplier - the standard grade
3-5 column-arithmetic algorithms (vertical_arithmetic / ArithmeticLayout
below). Long division by a single-digit (1-9) divisor (the "bring down
each digit" algorithm, see DivisionLayout / build_division_layout below)
is a further, structurally different layout of its own. Multi-digit x
multi-digit long multiplication and multi-digit-divisor long division
(which needs trial-and-adjust quotient digit guessing, not a direct
digit-by-digit computation) are both out of scope here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

VALID_OPERATIONS = ("add", "subtract", "multiply")


@dataclass
class ArithmeticLayout:
    """Everything the renderer needs, already computed - no further math
    or layout decisions left for teaching_canvas.py to make.

    top_digits / bottom_digits / result_digits: strings, right-aligned,
    all the same width, space-padded on the left. bottom_digits is the
    number being added/subtracted, or the single-digit multiplier
    (right-aligned, so it's naturally blank-padded to just the ones
    column).

    carry_marks[i] / borrow_marks[i]: the small annotation (if any) to
    draw above column i of top_digits - '' means no annotation there.
    borrow_struck[i]: True if column i's original top digit should be
    drawn with a strikethrough (it was reduced by a borrow), in which
    case borrow_marks[i] holds the reduced digit to show above it.
    """
    operation: str
    operator_symbol: str
    top_digits: str
    bottom_digits: str
    result_digits: str
    carry_marks: List[str] = field(default_factory=list)
    borrow_marks: List[str] = field(default_factory=list)
    borrow_struck: List[bool] = field(default_factory=list)


def validate_operands(operation: str, operands: List[int]) -> Optional[str]:
    """Returns an error message if the spec is invalid for rendering as
    vertical arithmetic, or None if it's fine. Checked BEFORE any
    computation - mathematical validation, not just a JSON-shape check."""
    if operation not in VALID_OPERATIONS:
        return f"Unknown operation '{operation}' - expected one of {VALID_OPERATIONS}."
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return "vertical_arithmetic needs exactly 2 operands."
    a, b = operands
    if not (isinstance(a, int) and isinstance(b, int)) or isinstance(a, bool) or isinstance(b, bool):
        return "Both operands must be plain integers."
    if a < 0 or b < 0:
        return "Negative operands aren't supported (grade 3-5 column arithmetic stays non-negative)."
    if operation == "subtract" and a < b:
        return f"Subtracting {b} from {a} would go negative - not supported for this visual."
    if operation == "multiply" and not (0 <= b <= 9):
        return "Vertical multiplication only supports a single-digit (0-9) multiplier as the second operand."
    return None


def _addition_carries(a: int, b: int, width: int) -> List[str]:
    a_digits = str(a).rjust(width, " ")
    b_digits = str(b).rjust(width, " ")
    carries = [""] * width
    carry = 0
    for i in range(width - 1, -1, -1):
        da = int(a_digits[i]) if a_digits[i] != " " else 0
        db = int(b_digits[i]) if b_digits[i] != " " else 0
        carry = 1 if (da + db + carry) >= 10 else 0
        if carry and i > 0:
            carries[i - 1] = "1"
    return carries


def _multiplication_carries(a: int, b: int, width: int) -> List[str]:
    a_digits = str(a).rjust(width, " ")
    carries = [""] * width
    carry = 0
    for i in range(width - 1, -1, -1):
        da = int(a_digits[i]) if a_digits[i] != " " else 0
        carry = (da * b + carry) // 10
        if carry and i > 0:
            carries[i - 1] = str(carry)
    return carries


def _subtraction_borrows(a: int, b: int, width: int) -> Tuple[List[str], List[bool]]:
    """Simulates right-to-left borrowing on a's own digits. Returns
    (marks, struck) - marks[i] is the regrouped value to show above
    column i (if any annotation is needed there at all), struck[i] is
    whether the ORIGINAL digit at column i should be shown crossed out
    (it lent 1 to its right neighbor). A column that only *received* a
    borrow (and didn't itself lend further) gets an annotation but no
    strikethrough, since its own written digit doesn't change - see the
    module docstring's example in the accompanying tests."""
    working = [int(c) for c in str(a).rjust(width, " ").replace(" ", "0")]
    b_digits = str(b).rjust(width, " ")
    struck = [False] * width
    received = [False] * width

    for i in range(width - 1, -1, -1):
        db = int(b_digits[i]) if b_digits[i] != " " else 0
        if working[i] < db:
            j = i - 1
            while j >= 0 and working[j] == 0:
                working[j] = 9
                struck[j] = True
                j -= 1
            working[j] -= 1
            struck[j] = True
            working[i] += 10
            received[i] = True

    marks = [""] * width
    for i in range(width):
        if struck[i] or received[i]:
            marks[i] = str(working[i])
    return marks, struck


_OPERATOR_SYMBOLS = {"add": "+", "subtract": "−", "multiply": "×"}


def build_layout(operation: str, operands: List[int]) -> ArithmeticLayout:
    """Raises ValueError if the spec fails validate_operands() - callers
    should validate first and treat a raised error as "don't render
    this", same as any other malformed draw action in this codebase."""
    error = validate_operands(operation, operands)
    if error:
        raise ValueError(error)

    a, b = operands

    if operation == "add":
        result = a + b
        width = max(len(str(a)), len(str(b)), len(str(result)))
        return ArithmeticLayout(
            operation=operation,
            operator_symbol=_OPERATOR_SYMBOLS[operation],
            top_digits=str(a).rjust(width, " "),
            bottom_digits=str(b).rjust(width, " "),
            result_digits=str(result).rjust(width, " "),
            carry_marks=_addition_carries(a, b, width),
        )

    if operation == "subtract":
        result = a - b
        width = max(len(str(a)), len(str(b)), len(str(result)))
        marks, struck = _subtraction_borrows(a, b, width)
        return ArithmeticLayout(
            operation=operation,
            operator_symbol=_OPERATOR_SYMBOLS[operation],
            top_digits=str(a).rjust(width, " "),
            bottom_digits=str(b).rjust(width, " "),
            result_digits=str(result).rjust(width, " "),
            borrow_marks=marks,
            borrow_struck=struck,
        )

    # multiply (b is validated to be a single digit 0-9)
    result = a * b
    width = max(len(str(a)), len(str(result)))
    return ArithmeticLayout(
        operation=operation,
        operator_symbol=_OPERATOR_SYMBOLS[operation],
        top_digits=str(a).rjust(width, " "),
        bottom_digits=str(b).rjust(width, " "),
        result_digits=str(result).rjust(width, " "),
        carry_marks=_multiplication_carries(a, b, width),
    )


@dataclass
class DivisionStep:
    """One digit-position's worth of the "bring down" algorithm.
    start_col/end_col are dividend-digit column indices (0-based, left to
    right) - brought_value/product/remainder_after are all right-aligned
    strings spanning exactly those columns (width 1 if nothing was
    carried into this step, width 2 if the previous step's remainder was
    carried in - never more, since a single-digit divisor's remainder is
    always itself a single digit)."""
    start_col: int
    end_col: int
    brought_value: str
    product: str
    remainder_after: str


@dataclass
class DivisionLayout:
    """Everything the renderer needs for long division, already computed.

    dividend_digits: plain digit string, e.g. "402" - unpadded, since
    division has no separate "result width" the way add/subtract/multiply
    do; every column position is just an index into this string.
    quotient_row: same length as dividend_digits, one character per
    column - a space at any column that comes before the first non-zero
    quotient digit (division "skips" a leading digit that's smaller than
    the divisor, same as real long division never writes a leading-zero
    quotient digit).
    steps: one entry per column that actually got a subtraction step
    (the skipped leading column, if any, has no entry).
    quotient/remainder: the ground-truth answer, computed directly via
    Python's own // and % - never re-derived from quotient_row/steps, so
    a rendering bug in the row layout can never silently change what the
    "real" answer is understood to be in tests or elsewhere.
    """
    dividend_digits: str
    divisor_digits: str
    quotient_row: str
    quotient: int
    remainder: int
    steps: List[DivisionStep] = field(default_factory=list)


def validate_division_operands(operands: List[int]) -> Optional[str]:
    """Returns an error message if the spec is invalid, or None. Scoped
    to a single-digit (1-9) divisor - the standard grade 4-5 "bring down"
    algorithm computes one quotient digit directly per dividend digit;
    a multi-digit divisor needs trial-and-adjust quotient guessing
    instead, a different algorithm, out of scope here."""
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return "long_division needs exactly 2 operands."
    a, b = operands
    if not (isinstance(a, int) and isinstance(b, int)) or isinstance(a, bool) or isinstance(b, bool):
        return "Both operands must be plain integers."
    if a < 0:
        return "The dividend must be non-negative."
    if not (1 <= b <= 9):
        return "long_division only supports a single-digit (1-9) divisor as the second operand."
    if a < b:
        return f"Dividing {a} by {b} gives a quotient of 0 - not supported for this visual."
    return None


def build_division_layout(operands: List[int]) -> DivisionLayout:
    """Raises ValueError if the spec fails validate_division_operands() -
    callers should validate first and treat a raised error as "don't
    render this", same as any other malformed draw action."""
    error = validate_division_operands(operands)
    if error:
        raise ValueError(error)

    dividend, divisor = operands
    digits = str(dividend)
    n = len(digits)
    quotient_chars = [" "] * n
    steps: List[DivisionStep] = []
    remainder = 0
    started = False

    for i, ch in enumerate(digits):
        digit = int(ch)
        remainder_before = remainder
        current = remainder_before * 10 + digit
        q, r = divmod(current, divisor)

        if not started and q == 0 and i < n - 1:
            # Dividend's leading digit(s) are smaller than the divisor -
            # skip writing a step/quotient digit here, but the remainder
            # still carries into the next column (a single-digit divisor
            # can only ever skip this one leading column - see the
            # module docstring's reasoning).
            remainder = r
            continue

        started = True
        start_col = i if remainder_before == 0 else i - 1
        width = i - start_col + 1
        steps.append(DivisionStep(
            start_col=start_col,
            end_col=i,
            brought_value=str(current).rjust(width, " "),
            product=str(q * divisor).rjust(width, " "),
            remainder_after=str(r).rjust(width, " "),
        ))
        quotient_chars[i] = str(q)
        remainder = r

    return DivisionLayout(
        dividend_digits=digits,
        divisor_digits=str(divisor),
        quotient_row="".join(quotient_chars),
        quotient=dividend // divisor,
        remainder=dividend % divisor,
        steps=steps,
    )
