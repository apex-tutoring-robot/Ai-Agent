"""
Tests for JarvisBot._tool_set_volume() - the implementation behind the
set_volume LLM tool (see VOLUME_CONTROL_TOOL's schema in main.py).

_tool_set_volume touches self.audio_player and self.ui_signals but nothing
else on JarvisBot, so these tests use a bare instance (JarvisBot.__new__)
with just those two attributes stubbed, rather than constructing a real
bot (which needs live Azure credentials, PyAudio devices, etc.) - same
reasoning as test_is_math_query.py's unbound-call approach, just via a
lightweight instance instead since this method needs real instance state
(current volume) to update.
"""

import pytest

from main import JarvisBot


class _FakeAudioPlayer:
    def __init__(self, volume=1.0):
        self.volume = volume

    def set_volume(self, value):
        self.volume = value


def _bot_with_volume(volume=1.0):
    bot = JarvisBot.__new__(JarvisBot)
    bot.audio_player = _FakeAudioPlayer(volume)
    bot.ui_signals = None
    return bot


class TestVolumeMagnitude:
    def test_small_louder_step_is_015(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("louder", "small")
        assert bot.audio_player.volume == pytest.approx(1.15)

    def test_medium_louder_step_is_025(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("louder", "medium")
        assert bot.audio_player.volume == pytest.approx(1.25)

    def test_large_louder_step_is_05(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("louder", "large")
        assert bot.audio_player.volume == pytest.approx(1.5)

    def test_small_quieter_step_is_015(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("quieter", "small")
        assert bot.audio_player.volume == pytest.approx(0.85)

    def test_large_quieter_step_is_05(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("quieter", "large")
        assert bot.audio_player.volume == pytest.approx(0.5)

    def test_missing_amount_defaults_to_medium(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("louder")
        assert bot.audio_player.volume == pytest.approx(1.25)

    def test_unrecognized_amount_falls_back_to_medium(self):
        bot = _bot_with_volume(1.0)
        bot._tool_set_volume("louder", "extremely-loud-please")
        assert bot.audio_player.volume == pytest.approx(1.25)

    def test_normal_ignores_amount_and_resets_to_one(self):
        bot = _bot_with_volume(1.7)
        bot._tool_set_volume("normal", "large")
        assert bot.audio_player.volume == pytest.approx(1.0)

    def test_normal_with_no_amount_still_resets_to_one(self):
        bot = _bot_with_volume(0.4)
        bot._tool_set_volume("normal")
        assert bot.audio_player.volume == pytest.approx(1.0)


class TestVolumeClamping:
    def test_large_step_clamps_at_max(self):
        bot = _bot_with_volume(1.8)
        bot._tool_set_volume("louder", "large")
        assert bot.audio_player.volume == pytest.approx(2.0)

    def test_large_step_clamps_at_min(self):
        bot = _bot_with_volume(0.4)
        bot._tool_set_volume("quieter", "large")
        assert bot.audio_player.volume == pytest.approx(0.25)

    def test_small_step_still_clamps_at_min(self):
        bot = _bot_with_volume(0.3)
        bot._tool_set_volume("quieter", "small")
        assert bot.audio_player.volume == pytest.approx(0.25)


class TestVolumeResponseText:
    def test_response_reports_actual_percentage(self):
        bot = _bot_with_volume(1.0)
        response = bot._tool_set_volume("louder", "large")
        assert "150%" in response

    def test_response_reports_maximum_wording_at_ceiling(self):
        bot = _bot_with_volume(1.8)
        response = bot._tool_set_volume("louder", "large")
        assert "maximum" in response.lower()

    def test_response_reports_minimum_wording_at_floor(self):
        bot = _bot_with_volume(0.4)
        response = bot._tool_set_volume("quieter", "large")
        assert "minimum" in response.lower()

    def test_response_tells_the_model_it_already_happened(self):
        bot = _bot_with_volume(1.0)
        response = bot._tool_set_volume("louder", "small")
        assert "already happened" in response.lower()

    def test_unknown_level_is_rejected(self):
        bot = _bot_with_volume(1.0)
        response = bot._tool_set_volume("sideways", "medium")
        assert "unknown volume level" in response.lower()
        assert bot.audio_player.volume == pytest.approx(1.0)  # unchanged
