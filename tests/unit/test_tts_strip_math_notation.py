r"""
Tests for azure_services.tts_client._strip_math_notation() - a defensive
cleanup pass for markdown/LaTeX artifacts that occasionally slip into an
LLM response despite every prompt's "no formatting" instructions, applied
right before synthesis in synthesize_stream() so it catches text from any
caller.

Regression test for a real bug found via a production log review: a
generic chat response about circle area came back containing literal
LaTeX ("\[ \text{Area} = \pi r^2 \]"), which would have been read aloud
as gibberish instead of plain spoken math.

A module-level pure function, so it's tested directly with no client
construction needed.
"""

from azure_services.tts_client import _strip_math_notation


class TestLatexDelimitersAndCommands:
    def test_strips_bracket_delimiters(self):
        assert _strip_math_notation(r"\[ \pi r^2 \]") == "pi r squared"

    def test_strips_paren_delimiters(self):
        assert _strip_math_notation(r"\( x + 1 \)") == "x + 1"

    def test_unwraps_text_command(self):
        assert _strip_math_notation(r"\text{Area} = 5") == "Area = 5"

    def test_converts_frac_to_spoken_form(self):
        assert _strip_math_notation(r"\frac{1}{2}") == "1 over 2"

    def test_strips_double_dollar_delimiters(self):
        assert _strip_math_notation(r"$$x^2$$") == "x squared"


class TestLatexSymbols:
    def test_pi_symbol(self):
        assert _strip_math_notation(r"\pi r^2") == "pi r squared"

    def test_times_symbol(self):
        assert _strip_math_notation(r"3 \times 4") == "3 times 4"

    def test_div_symbol(self):
        assert _strip_math_notation(r"8 \div 2") == "8 divided by 2"

    def test_cdot_symbol(self):
        assert _strip_math_notation(r"3 \cdot 4") == "3 times 4"


class TestExponents:
    def test_squared(self):
        assert _strip_math_notation("r^2") == "r squared"

    def test_cubed(self):
        assert _strip_math_notation("x^3") == "x cubed"

    def test_other_power_uses_generic_phrasing(self):
        assert _strip_math_notation("x^5") == "x to the power of 5"

    def test_does_not_double_convert_squared(self):
        # The specific ^2/^3 patterns must run BEFORE the generic one,
        # or this would come out as "r to the power of 2" instead.
        assert "power" not in _strip_math_notation("r^2")


class TestMarkdown:
    def test_strips_bold_markers(self):
        assert _strip_math_notation("**numerator**") == "numerator"

    def test_strips_leading_bullet_dash(self):
        assert _strip_math_notation("- The numerator is 1") == "The numerator is 1"

    def test_strips_leading_bullet_asterisk(self):
        assert _strip_math_notation("* The denominator is 4") == "The denominator is 4"

    def test_does_not_touch_multiplication_asterisk(self):
        # Single "*" used for multiplication (not a markdown bullet or
        # bold marker) must be left alone - only a line-leading "* " or
        # a "**...**" pair are touched.
        assert _strip_math_notation("3 * 4 = 12") == "3 * 4 = 12"


class TestRealWorldExample:
    def test_the_actual_production_case(self):
        # Exact text from the log that motivated this fix.
        text = r"\[ \text{Area} = \pi r^2 \]"
        result = _strip_math_notation(text)
        assert "\\" not in result
        assert "pi" in result
        assert "squared" in result

    def test_plain_text_is_unchanged(self):
        text = "So you just multiply the width by the height - that's the area!"
        assert _strip_math_notation(text) == text
