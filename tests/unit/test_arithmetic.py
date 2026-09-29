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
    build_long_multiplication_layout,
    validate_long_multiplication_operands,
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


class TestLongMultiplicationValidation:
    def test_rejects_wrong_operand_count(self):
        assert validate_long_multiplication_operands([23]) is not None

    def test_rejects_non_integer_operands(self):
        assert validate_long_multiplication_operands([23.5, 14]) is not None

    def test_rejects_negative_operands(self):
        assert validate_long_multiplication_operands([-23, 14]) is not None

    def test_rejects_zero_multiplicand(self):
        assert validate_long_multiplication_operands([0, 14]) is not None

    def test_rejects_single_digit_multiplier(self):
        # Should use vertical_arithmetic's multiply instead.
        assert validate_long_multiplication_operands([234, 6]) is not None

    def test_accepts_two_digit_multiplier(self):
        assert validate_long_multiplication_operands([23, 14]) is None


class TestLongMultiplication:
    def test_two_digit_by_two_digit_with_carry(self):
        # 23 x 14 = 322: first partial product (23x4=92) carries 1 into
        # the tens column; summing 92 + 230 carries 1 into the hundreds.
        layout = build_long_multiplication_layout([23, 14])
        assert layout.top_digits.strip() == "23"
        assert layout.bottom_digits.strip() == "14"
        assert layout.result_digits.strip() == "322"
        assert [p.strip() for p in layout.partial_products] == ["92", "230"]
        assert layout.partial_carries[0] == ["", "1", ""]
        assert layout.partial_carries[1] == ["", "", ""]
        assert layout.result_carries == ["1", "", ""]

    def test_three_digit_by_two_digit_no_summation_carry(self):
        # 234 x 16 = 3744: partial products 1404 + 2340 sum cleanly
        # (no column exceeds 9), so result_carries is all blank.
        layout = build_long_multiplication_layout([234, 16])
        assert layout.result_digits.strip() == "3744"
        assert [p.strip() for p in layout.partial_products] == ["1404", "2340"]
        assert all(c == "" for c in layout.result_carries)

    def test_three_digit_multiplier_three_partial_rows(self):
        layout = build_long_multiplication_layout([47, 123])
        assert len(layout.partial_products) == 3
        assert int("".join(layout.result_digits.split())) == 47 * 123

    def test_multiplier_with_internal_zero_digit(self):
        # 23 x 104: the tens-digit partial product is 23 x 0 = 0, still
        # produces a valid (all-zero, correctly shifted) row.
        layout = build_long_multiplication_layout([23, 104])
        assert layout.result_digits.strip() == "2392"
        assert [p.strip() for p in layout.partial_products] == ["92", "0", "2300"]

    def test_result_matches_plain_python_multiplication_for_many_cases(self):
        import random
        random.seed(123)
        for _ in range(200):
            a = random.randint(1, 9999)
            b = random.randint(10, 999)
            layout = build_long_multiplication_layout([a, b])
            assert int(layout.result_digits) == a * b
            assert sum(int(p) for p in layout.partial_products) == a * b

    def test_raises_on_invalid_spec(self):
        with pytest.raises(ValueError):
            build_long_multiplication_layout([234, 6])
