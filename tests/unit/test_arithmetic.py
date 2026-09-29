"""
Tests for visuals.arithmetic - the deterministic column-arithmetic
computation behind the whiteboard's vertical_arithmetic draw action.

This is the safety-critical core of the "LLM picks WHAT, Python computes
HOW" pattern: the LLM only ever supplies (operation, operands), so every
one of these computations needs to be independently, exhaustively correct
- there's no LLM judgment left to catch a bug here.
"""

import pytest

from visuals.arithmetic import (
    build_layout,
    validate_operands,
    build_division_layout,
    validate_division_operands,
)


class TestValidation:
    def test_rejects_unknown_operation(self):
        assert validate_operands("divide", [10, 2]) is not None

    def test_rejects_wrong_operand_count(self):
        assert validate_operands("add", [10]) is not None
        assert validate_operands("add", [10, 2, 3]) is not None

    def test_rejects_non_integer_operands(self):
        assert validate_operands("add", [10.5, 2]) is not None
        assert validate_operands("add", ["10", 2]) is not None

    def test_rejects_booleans_masquerading_as_ints(self):
        # isinstance(True, int) is True in Python - must be excluded explicitly.
        assert validate_operands("add", [True, 2]) is not None

    def test_rejects_negative_operands(self):
        assert validate_operands("add", [-5, 2]) is not None
        assert validate_operands("subtract", [10, -2]) is not None

    def test_rejects_subtraction_that_would_go_negative(self):
        assert validate_operands("subtract", [5, 10]) is not None

    def test_accepts_subtraction_with_equal_operands(self):
        assert validate_operands("subtract", [7, 7]) is None

    def test_rejects_multi_digit_multiplier(self):
        assert validate_operands("multiply", [234, 12]) is not None

    def test_accepts_single_digit_multiplier(self):
        assert validate_operands("multiply", [234, 9]) is None
        assert validate_operands("multiply", [234, 0]) is None

    def test_accepts_valid_addition(self):
        assert validate_operands("add", [347, 58]) is None


class TestAddition:
    def test_simple_no_carry(self):
        layout = build_layout("add", [12, 3])
        assert layout.result_digits.strip() == "15"
        assert all(c == "" for c in layout.carry_marks)

    def test_single_carry(self):
        # 347 + 58 = 405: carry into tens (7+8=15) and hundreds (4+5+1=10)
        layout = build_layout("add", [347, 58])
        assert layout.result_digits.strip() == "405"
        assert layout.top_digits.strip() == "347"
        assert layout.bottom_digits.strip() == "58"
        # width = 4 ("_347" style), carries land on the two columns left
        # of where each carry was generated.
        non_empty = [i for i, c in enumerate(layout.carry_marks) if c]
        assert layout.carry_marks[non_empty[0]] == "1"
        assert len(non_empty) == 2

    def test_carry_all_the_way_through(self):
        # 999 + 1 = 1000 - carries cascade through every column.
        layout = build_layout("add", [999, 1])
        assert layout.result_digits.strip() == "1000"

    def test_adding_zero(self):
        layout = build_layout("add", [123, 0])
        assert layout.result_digits.strip() == "123"

    def test_result_matches_plain_python_addition_for_many_cases(self):
        import random
        random.seed(42)
        for _ in range(200):
            a, b = random.randint(0, 99999), random.randint(0, 99999)
            layout = build_layout("add", [a, b])
            assert int(layout.result_digits) == a + b


class TestSubtraction:
    def test_simple_no_borrow(self):
        layout = build_layout("subtract", [78, 23])
        assert layout.result_digits.strip() == "55"
        assert all(m == "" for m in layout.borrow_marks)
        assert all(s is False for s in layout.borrow_struck)

    def test_single_borrow(self):
        # 52 - 27 = 25: ones borrows from tens (5 -> 4, struck; ones becomes 12)
        layout = build_layout("subtract", [52, 27])
        assert layout.result_digits.strip() == "25"
        # tens column (index 0 of a 2-wide layout) lent 1 -> struck, shows "4"
        assert layout.borrow_struck[0] is True
        assert layout.borrow_marks[0] == "4"
        # ones column (index 1) received the borrow -> shows "12", not struck
        assert layout.borrow_struck[1] is False
        assert layout.borrow_marks[1] == "12"

    def test_cascading_borrow_through_a_zero(self):
        # 405 - 58 = 347: ones borrows, tens is 0 so it cascades to hundreds.
        layout = build_layout("subtract", [405, 58])
        assert layout.result_digits.strip() == "347"
        # hundreds (index 0): lent 1, 4 -> 3
        assert layout.borrow_struck[0] is True
        assert layout.borrow_marks[0] == "3"
        # tens (index 1): received from hundreds (became 10), then lent 1 to
        # ones, net displayed value 9, struck because its written digit (0)
        # effectively changed for the computation.
        assert layout.borrow_struck[1] is True
        assert layout.borrow_marks[1] == "9"
        # ones (index 2): received a borrow, becomes 15, not struck.
        assert layout.borrow_struck[2] is False
        assert layout.borrow_marks[2] == "15"

    def test_subtracting_to_zero(self):
        layout = build_layout("subtract", [42, 42])
        assert layout.result_digits.strip() == "0"

    def test_subtracting_zero(self):
        layout = build_layout("subtract", [42, 0])
        assert layout.result_digits.strip() == "42"

    def test_result_matches_plain_python_subtraction_for_many_cases(self):
        import random
        random.seed(7)
        for _ in range(200):
            a = random.randint(0, 99999)
            b = random.randint(0, a)  # keep it non-negative, per validation
            layout = build_layout("subtract", [a, b])
            assert int(layout.result_digits) == a - b


class TestMultiplication:
    def test_simple_no_carry(self):
        layout = build_layout("multiply", [21, 3])
        assert layout.result_digits.strip() == "63"
        assert all(c == "" for c in layout.carry_marks)

    def test_single_digit_multiplier_with_carry(self):
        # 234 x 6 = 1404
        layout = build_layout("multiply", [234, 6])
        assert layout.result_digits.strip() == "1404"
        assert layout.top_digits.strip() == "234"
        assert layout.bottom_digits.strip() == "6"

    def test_multiplying_by_zero(self):
        layout = build_layout("multiply", [1234, 0])
        assert layout.result_digits.strip() == "0"

    def test_multiplying_by_one(self):
        layout = build_layout("multiply", [1234, 1])
        assert layout.result_digits.strip() == "1234"
        assert all(c == "" for c in layout.carry_marks)

    def test_result_matches_plain_python_multiplication_for_many_cases(self):
        import random
        random.seed(99)
        for _ in range(200):
            a, b = random.randint(0, 99999), random.randint(0, 9)
            layout = build_layout("multiply", [a, b])
            assert int(layout.result_digits) == a * b


class TestBuildLayoutRaisesOnInvalidSpec:
    def test_raises_on_negative_result_subtraction(self):
        with pytest.raises(ValueError):
            build_layout("subtract", [5, 10])

    def test_raises_on_multi_digit_multiplier(self):
        with pytest.raises(ValueError):
            build_layout("multiply", [12, 34])

    def test_raises_on_unknown_operation(self):
        with pytest.raises(ValueError):
            build_layout("divide", [10, 2])


class TestDivisionValidation:
    def test_rejects_wrong_operand_count(self):
        assert validate_division_operands([10]) is not None

    def test_rejects_non_integer_operands(self):
        assert validate_division_operands([10.5, 2]) is not None

    def test_rejects_negative_dividend(self):
        assert validate_division_operands([-10, 2]) is not None

    def test_rejects_zero_divisor(self):
        assert validate_division_operands([10, 0]) is not None

    def test_rejects_multi_digit_divisor(self):
        assert validate_division_operands([100, 12]) is not None

    def test_rejects_dividend_smaller_than_divisor(self):
        assert validate_division_operands([2, 3]) is not None

    def test_accepts_dividend_equal_to_divisor(self):
        assert validate_division_operands([5, 5]) is None

    def test_accepts_valid_division(self):
        assert validate_division_operands([84, 3]) is None


class TestDivision:
    def test_simple_no_remainder_no_leading_skip(self):
        # 84 / 3 = 28: first digit (8) >= divisor, so every dividend
        # digit gets its own step, no leading skip.
        layout = build_division_layout([84, 3])
        assert layout.quotient == 28
        assert layout.remainder == 0
        assert layout.quotient_row == "28"
        assert len(layout.steps) == 2
        assert (layout.steps[0].start_col, layout.steps[0].end_col) == (0, 0)
        assert layout.steps[0].brought_value == "8"
        assert layout.steps[0].product == "6"
        assert layout.steps[0].remainder_after == "2"
        assert (layout.steps[1].start_col, layout.steps[1].end_col) == (0, 1)
        assert layout.steps[1].brought_value == "24"
        assert layout.steps[1].product == "24"
        # Right-aligned to the step's 2-column width - the remainder (a
        # single digit, since it's always < the divisor) sits under the
        # last column, blank under the first.
        assert layout.steps[1].remainder_after == " 0"

    def test_leading_digit_smaller_than_divisor_is_skipped(self):
        # 105 / 7 = 15: the first digit (1) is smaller than 7, so it's
        # skipped (no step, no quotient digit) and its remainder carries
        # into the next column, same as real long division never writing
        # a leading-zero quotient digit.
        layout = build_division_layout([105, 7])
        assert layout.quotient == 15
        assert layout.remainder == 0
        assert layout.quotient_row == " 15"
        assert len(layout.steps) == 2
        assert (layout.steps[0].start_col, layout.steps[0].end_col) == (0, 1)
        assert layout.steps[0].brought_value == "10"

    def test_internal_zero_quotient_digit_still_gets_a_step(self):
        # 402 / 4 = 100 remainder 2: once started, every remaining digit
        # gets a full step even when that digit's own quotient is 0 -
        # otherwise place value would be lost.
        layout = build_division_layout([402, 4])
        assert layout.quotient == 100
        assert layout.remainder == 2
        assert layout.quotient_row == "100"
        assert len(layout.steps) == 3
        assert layout.steps[1].brought_value == "0"
        assert layout.steps[1].product == "0"

    def test_internal_zero_quotient_digit_after_a_carried_remainder(self):
        # 181 / 9 = 20 remainder 1: the middle step lands on a carried-in
        # remainder of 0 (from the previous step's exact division), so
        # its own span collapses back to width 1.
        layout = build_division_layout([181, 9])
        assert layout.quotient == 20
        assert layout.remainder == 1
        assert layout.quotient_row == " 20"
        assert len(layout.steps) == 2
        last = layout.steps[-1]
        assert (last.start_col, last.end_col) == (2, 2)
        assert last.brought_value == "1"
        assert last.product == "0"
        assert last.remainder_after == "1"

    def test_exact_division_with_single_digit_dividend(self):
        layout = build_division_layout([9, 3])
        assert layout.quotient == 3
        assert layout.remainder == 0
        assert layout.quotient_row == "3"

    def test_dividend_equal_to_divisor(self):
        layout = build_division_layout([5, 5])
        assert layout.quotient == 1
        assert layout.remainder == 0

    def test_result_matches_plain_python_division_for_many_cases(self):
        import random
        random.seed(2024)
        for _ in range(200):
            divisor = random.randint(1, 9)
            dividend = random.randint(divisor, 999999)
            layout = build_division_layout([dividend, divisor])
            assert layout.quotient == dividend // divisor
            assert layout.remainder == dividend % divisor
            assert int(layout.quotient_row) == dividend // divisor
            if layout.steps:
                assert int(layout.steps[-1].remainder_after) == dividend % divisor

    def test_raises_on_invalid_spec(self):
        with pytest.raises(ValueError):
            build_division_layout([2, 3])
