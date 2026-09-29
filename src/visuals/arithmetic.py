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

Scope, deliberately: addition and subtraction of any two non-negative
integers, and multiplication by a single-digit (0-9) multiplier - the
standard grade 3-5 column-arithmetic algorithms. Multi-digit x multi-digit
long multiplication (partial-product rows) and long division (quotient
bar, remainder) are structurally different layouts, not just bigger
numbers - deliberately out of scope for this first primitive rather than
half-implemented.
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
