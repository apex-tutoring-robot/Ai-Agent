"""
Tests for visuals.number_line - the deterministic tick/point/jump pixel
computation behind the whiteboard's number_line draw action.

Same safety-critical role as tests/unit/test_arithmetic.py: the LLM only
ever supplies the integer range and integer values to plot, never a pixel
coordinate, so every pixel position computed here needs to be
independently correct.
"""

import pytest

from visuals.number_line import build_number_line_layout, validate_number_line_spec


class TestValidation:
    def test_rejects_non_integer_min(self):
        assert validate_number_line_spec(0.5, 10, 500, [], []) is not None

    def test_rejects_non_integer_max(self):
        assert validate_number_line_spec(0, 10.5, 500, [], []) is not None

    def test_rejects_booleans_masquerading_as_ints(self):
        assert validate_number_line_spec(False, 10, 500, [], []) is not None

    def test_rejects_min_not_less_than_max(self):
        assert validate_number_line_spec(10, 10, 500, [], []) is not None
        assert validate_number_line_spec(10, 5, 500, [], []) is not None

    def test_rejects_span_too_small(self):
        assert validate_number_line_spec(0, 1, 500, [], []) is not None

    def test_rejects_span_too_large(self):
        assert validate_number_line_spec(0, 1000, 500, [], []) is not None

    def test_rejects_non_positive_width(self):
        assert validate_number_line_spec(0, 10, 0, [], []) is not None
        assert validate_number_line_spec(0, 10, -100, [], []) is not None

    def test_rejects_point_missing_value(self):
        assert validate_number_line_spec(0, 10, 500, [{}], []) is not None

    def test_rejects_point_out_of_range(self):
        assert validate_number_line_spec(0, 10, 500, [{"value": 15}], []) is not None

    def test_rejects_jump_missing_from_or_to(self):
        assert validate_number_line_spec(0, 10, 500, [], [{"from": 2}]) is not None

    def test_rejects_jump_out_of_range(self):
        assert validate_number_line_spec(0, 10, 500, [], [{"from": 2, "to": 15}]) is not None

    def test_rejects_jump_with_equal_from_and_to(self):
        assert validate_number_line_spec(0, 10, 500, [], [{"from": 5, "to": 5}]) is not None

    def test_accepts_valid_spec(self):
        assert validate_number_line_spec(0, 10, 500, [{"value": 7}], [{"from": 5, "to": 8}]) is None

    def test_accepts_empty_points_and_jumps(self):
        assert validate_number_line_spec(0, 10, 500, [], []) is None


class TestBuildNumberLineLayout:
    def test_ticks_span_the_full_width(self):
        layout = build_number_line_layout(0, 10, 500)
        assert layout.tick_values == list(range(0, 11))
        assert layout.tick_px[0] == 0
        assert layout.tick_px[-1] == 500
        assert layout.tick_px[5] == 250  # value 5 is halfway across 0-10

    def test_ticks_with_a_negative_range(self):
        layout = build_number_line_layout(-5, 5, 500)
        assert layout.tick_values == list(range(-5, 6))
        assert layout.tick_px[layout.tick_values.index(0)] == 250  # 0 is the midpoint

    def test_point_pixel_position_is_linearly_interpolated(self):
        layout = build_number_line_layout(0, 10, 500, points=[{"value": 7}])
        assert layout.points[0].px == pytest.approx(350)
        assert layout.points[0].label == "7"

    def test_point_uses_explicit_label_when_given(self):
        layout = build_number_line_layout(0, 10, 500, points=[{"value": 7, "label": "seven"}])
        assert layout.points[0].label == "seven"

    def test_jump_direction_and_default_label_increasing(self):
        layout = build_number_line_layout(0, 10, 500, jumps=[{"from": 5, "to": 8}])
        jump = layout.jumps[0]
        assert jump.direction == 1
        assert jump.label == "+3"
        assert jump.from_px == pytest.approx(250)
        assert jump.to_px == pytest.approx(400)

    def test_jump_direction_and_default_label_decreasing(self):
        layout = build_number_line_layout(0, 10, 500, jumps=[{"from": 8, "to": 5}])
        jump = layout.jumps[0]
        assert jump.direction == -1
        assert jump.label == "-3"

    def test_jump_uses_explicit_label_when_given(self):
        layout = build_number_line_layout(0, 10, 500, jumps=[{"from": 5, "to": 8, "label": "jump!"}])
        assert layout.jumps[0].label == "jump!"

    def test_multiple_points_and_jumps(self):
        layout = build_number_line_layout(
            0, 20, 800,
            points=[{"value": 3}, {"value": 17}],
            jumps=[{"from": 3, "to": 10}, {"from": 10, "to": 6}],
        )
        assert len(layout.points) == 2
        assert len(layout.jumps) == 2

    def test_raises_on_invalid_spec(self):
        with pytest.raises(ValueError):
            build_number_line_layout(10, 0, 500)

    def test_raises_on_out_of_range_point(self):
        with pytest.raises(ValueError):
            build_number_line_layout(0, 10, 500, points=[{"value": 99}])
