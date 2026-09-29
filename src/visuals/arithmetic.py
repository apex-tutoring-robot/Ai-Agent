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
below). Multi-digit x multi-digit long multiplication (partial-product
rows, see LongMultiplicationLayout / build_long_multiplication_layout
below) is a structurally different layout - its own action name and
dataclass, rather than overloading vertical_arithmetic's "multiply" with
a second shape. Long division (quotient bar, remainder) is a further,
separate layout still not implemented here.
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
class LongMultiplicationLayout:
    """Multi-digit x multi-digit long multiplication: one partial-product
    row per digit of the multiplier (bottom_digits), right-aligned and
    already left-shifted with explicit zero placeholders (e.g. 23 x 14's
    second partial product is written "230", not "23" shifted - the
    beginner-friendly "placeholder zero" convention, since it keeps every
    row a plain right-aligned digit string with no shift for the renderer
    to reinterpret), followed by a final summation row.

    partial_carries[row] holds that row's own multiplication carry marks
    (the same per-digit carry annotation used by ArithmeticLayout's single-
    digit multiply, since computing one partial product IS a single-digit
    multiplication - it's unrelated to the shift applied afterward).
    result_carries holds the addition carry marks for summing all the
    partial product rows together into result_digits.

    All digit strings are right-aligned, space-padded, and the same width.
    """
    top_digits: str
    bottom_digits: str
    partial_products: List[str] = field(default_factory=list)
    partial_carries: List[List[str]] = field(default_factory=list)
    result_digits: str = ""
    result_carries: List[str] = field(default_factory=list)


def validate_long_multiplication_operands(operands: List[int]) -> Optional[str]:
    """Returns an error message if the spec is invalid, or None. Long
    multiplication is specifically for a multi-digit multiplier - a
    single-digit multiplier should use vertical_arithmetic's "multiply"
    operation instead, so there's only ever one correct action for a
    given problem."""
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return "long_multiplication needs exactly 2 operands."
    a, b = operands
    if not (isinstance(a, int) and isinstance(b, int)) or isinstance(a, bool) or isinstance(b, bool):
        return "Both operands must be plain integers."
    if a < 0 or b < 0:
        return "Negative operands aren't supported (grade-school column arithmetic stays non-negative)."
    if a == 0:
        return "The first operand (the number being multiplied) must be positive."
    if b < 10:
        return (
            "The second operand has only one digit - use vertical_arithmetic's "
            "'multiply' operation instead of long_multiplication for a "
            "single-digit multiplier."
        )
    return None


def _sum_carries(rows: List[str], width: int) -> List[str]:
    """Right-to-left addition-carry simulation for summing N same-width,
    right-aligned digit-strings (the partial product rows) - a
    generalization of _addition_carries beyond exactly 2 addends. A carry
    can exceed 1 digit at wide columns with several rows, so marks are
    stored as their full string value (e.g. "12"), same convention as
    ArithmeticLayout.borrow_marks."""
    carries = [""] * width
    carry = 0
    for i in range(width - 1, -1, -1):
        col_sum = carry
        for row in rows:
            ch = row[i]
            if ch != " ":
                col_sum += int(ch)
        carry = col_sum // 10
        if carry and i > 0:
            carries[i - 1] = str(carry)
    return carries


def build_long_multiplication_layout(operands: List[int]) -> LongMultiplicationLayout:
    """Raises ValueError if the spec fails validate_long_multiplication_
    operands() - callers should validate first and treat a raised error
    as "don't render this", same as any other malformed draw action."""
    error = validate_long_multiplication_operands(operands)
    if error:
        raise ValueError(error)

    a, b = operands
    result = a * b
    b_str = str(b)
    width = len(str(result))

    partial_products: List[str] = []
    partial_carries: List[List[str]] = []
    for shift, digit_char in enumerate(reversed(b_str)):
        d = int(digit_char)
        shifted_value = (a * d) * (10 ** shift)
        partial_products.append(str(shifted_value).rjust(width, " "))
        partial_carries.append(_multiplication_carries(a, d, width))

    return LongMultiplicationLayout(
        top_digits=str(a).rjust(width, " "),
        bottom_digits=str(b).rjust(width, " "),
        partial_products=partial_products,
        partial_carries=partial_carries,
        result_digits=str(result).rjust(width, " "),
        result_carries=_sum_carries(partial_products, width),
    )
