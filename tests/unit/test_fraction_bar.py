"""
Tests for visuals.fraction_bar - the deterministic segment-boundary
computation behind the whiteboard's fraction_bar draw action.

Same safety-critical role as test_arithmetic.py/test_number_line.py: the
LLM only ever supplies numerator/denominator values, never a pixel
coordinate, so every segment boundary computed here needs to be
independently correct.
"""

import pytest

from visuals.fraction_bar import build_fraction_bar_layout, validate_fraction_bar_spec


class TestValidation:
    def test_rejects_non_positive_width(self):
        assert validate_fraction_bar_spec([{"numerator": 1, "denominator": 2}], 0, 50) is not None
        assert validate_fraction_bar_spec([{"numerator": 1, "denominator": 2}], -100, 50) is not None

    def test_rejects_non_positive_bar_height(self):
        assert validate_fraction_bar_spec([{"numerator": 1, "denominator": 2}], 600, 0) is not None

    def test_rejects_empty_fraction_list(self):
        assert validate_fraction_bar_spec([], 600, 50) is not None

    def test_rejects_too_many_bars(self):
        fractions = [{"numerator": 1, "denominator": 2}] * 5
        assert validate_fraction_bar_spec(fractions, 600, 50) is not None

    def test_rejects_missing_numerator_or_denominator(self):
        assert validate_fraction_bar_spec([{"numerator": 1}], 600, 50) is not None
        assert validate_fraction_bar_spec([{"denominator": 2}], 600, 50) is not None

    def test_rejects_non_integer_values(self):
        assert validate_fraction_bar_spec([{"numerator": 1.5, "denominator": 2}], 600, 50) is not None

    def test_rejects_booleans_masquerading_as_ints(self):
        assert validate_fraction_bar_spec([{"numerator": True, "denominator": 2}], 600, 50) is not None

    def test_rejects_negative_numerator(self):
        assert validate_fraction_bar_spec([{"numerator": -1, "denominator": 2}], 600, 50) is not None

    def test_rejects_denominator_out_of_range(self):
        assert validate_fraction_bar_spec([{"numerator": 0, "denominator": 0}], 600, 50) is not None
        assert validate_fraction_bar_spec([{"numerator": 0, "denominator": 13}], 600, 50) is not None

    def test_rejects_improper_fraction(self):
        assert validate_fraction_bar_spec([{"numerator": 5, "denominator": 4}], 600, 50) is not None

    def test_accepts_whole_number_fraction(self):
        assert validate_fraction_bar_spec([{"numerator": 4, "denominator": 4}], 600, 50) is None

    def test_accepts_valid_spec(self):
        assert validate_fraction_bar_spec([{"numerator": 3, "denominator": 4}], 600, 50) is None

    def test_accepts_max_bars(self):
        fractions = [{"numerator": 1, "denominator": 2}] * 4
        assert validate_fraction_bar_spec(fractions, 600, 50) is None


class TestBuildFractionBarLayout:
    def test_single_bar_segment_count_and_shading(self):
        layout = build_fraction_bar_layout([{"numerator": 3, "denominator": 4}], 600)
        assert len(layout.bars) == 1
        bar = layout.bars[0]
        assert len(bar.segments) == 4
        assert [s.shaded for s in bar.segments] == [True, True, True, False]

    def test_segment_boundaries_evenly_divide_the_width(self):
        layout = build_fraction_bar_layout([{"numerator": 1, "denominator": 4}], 800)
        bar = layout.bars[0]
        assert bar.segments[0].x0 == 0
        assert bar.segments[0].x1 == 200
        assert bar.segments[1].x0 == 200
        assert bar.segments[3].x1 == 800

    def test_default_label_is_the_fraction(self):
        layout = build_fraction_bar_layout([{"numerator": 3, "denominator": 4}], 600)
        assert layout.bars[0].label == "3/4"

    def test_custom_label_is_used_when_given(self):
        layout = build_fraction_bar_layout([{"numerator": 3, "denominator": 4, "label": "pizza eaten"}], 600)
        assert layout.bars[0].label == "pizza eaten"

    def test_whole_number_fraction_shades_every_segment(self):
        layout = build_fraction_bar_layout([{"numerator": 4, "denominator": 4}], 600)
        assert all(s.shaded for s in layout.bars[0].segments)

    def test_zero_numerator_shades_nothing(self):
        layout = build_fraction_bar_layout([{"numerator": 0, "denominator": 4}], 600)
        assert not any(s.shaded for s in layout.bars[0].segments)

    def test_multiple_bars_for_comparison(self):
        layout = build_fraction_bar_layout(
            [{"numerator": 1, "denominator": 2}, {"numerator": 3, "denominator": 4}], 600
        )
        assert len(layout.bars) == 2
        assert layout.bars[0].label == "1/2"
        assert layout.bars[1].label == "3/4"

    def test_equivalent_fractions_same_width_different_denominators(self):
        # 1/2 and 2/4 should shade the same physical span even though
        # they have a different number of segments.
        layout = build_fraction_bar_layout(
            [{"numerator": 1, "denominator": 2}, {"numerator": 2, "denominator": 4}], 800
        )
        half_bar, quarter_bar = layout.bars
        half_shaded_end = max(s.x1 for s in half_bar.segments if s.shaded)
        quarter_shaded_end = max(s.x1 for s in quarter_bar.segments if s.shaded)
        assert half_shaded_end == quarter_shaded_end == 400

    def test_raises_on_invalid_spec(self):
        with pytest.raises(ValueError):
            build_fraction_bar_layout([{"numerator": 5, "denominator": 4}], 600)
