"""
Tests for JarvisBot.is_math_query() - the keyword heuristic that routes a
turn to the teaching-plan pipeline (whiteboard diagram) vs. a normal
conversational answer.

is_math_query() never touches `self`, so it's called unbound
(JarvisBot.is_math_query(None, text)) rather than constructing a real
JarvisBot, which would need live Azure credentials, PyAudio devices, etc.
"""

import pytest

from main import JarvisBot


def is_math_query(text: str) -> bool:
    return JarvisBot.is_math_query(None, text)


class TestVolumeKeywordCollision:
    """
    Regression test for a real bug found live: "volume" is both a genuine
    3D-geometry term (volume of a cube) and the word used for loudness (see
    the set_volume tool - "please talk more loudly", "maximum volume").
    A bare substring match on "volume" sent "Please tell me in maximum
    volume" to the teaching-plan pipeline instead of set_volume - the
    JSON-only prompt got a conversational reply back, failed to parse, and
    Jarvis apologized instead of changing the volume.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Please tell me in maximum volume.",
            "please talk more loudly",
            "can you be quieter",
            "turn the volume up",
            "please talk in a lower volume",
        ],
    )
    def test_audio_volume_phrasing_is_not_math(self, text):
        assert is_math_query(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "what is the volume of a cube with side 3",
            "find the volume of this cylinder",
            "how do I calculate the volume of a sphere",
        ],
    )
    def test_geometry_volume_phrasing_is_math(self, text):
        assert is_math_query(text) is True


class TestOtherMathKeywords:
    @pytest.mark.parametrize(
        "text",
        [
            "what is the area of a triangle",
            "help me with fractions",
            "solve this equation for x",
            "what is the perimeter of a square",
            "2 + 2 = ?",
        ],
    )
    def test_recognized_as_math(self, text):
        assert is_math_query(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "what grade are you in",
            "third grade",
            "tell me about yourself",
        ],
    )
    def test_not_recognized_as_math(self, text):
        assert is_math_query(text) is False
