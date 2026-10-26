"""
Tests for MicLevelIndicator.show_level() - the RMS normalization +
smoothing logic behind the live "is the mic actually picking anything
up" meter (see ui_signals.mic_level_changed). Pure numeric logic, no
paintEvent exercised - same convention as test_audio_device_indicator.py
(verified live via rendered screenshots instead).
"""

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.mic_level_indicator import MicLevelIndicator


class TestShowLevel:
    def test_silence_keeps_the_level_near_zero(self):
        widget = MicLevelIndicator()
        widget.show_level(0.0)
        assert widget._level < 0.01

    def test_loud_rms_moves_the_level_up(self):
        widget = MicLevelIndicator()
        widget.show_level(MicLevelIndicator._MAX_RMS)
        assert widget._level > 0.0

    def test_rms_above_max_is_clamped_to_one(self):
        widget = MicLevelIndicator()
        for _ in range(50):  # enough smoothing steps to settle near the ceiling
            widget.show_level(MicLevelIndicator._MAX_RMS * 10)
        assert widget._level <= 1.0

    def test_negative_rms_is_clamped_to_zero(self):
        # Shouldn't happen from a real RMS calculation, but the normalize
        # step shouldn't produce a negative level either way.
        widget = MicLevelIndicator()
        widget.show_level(-500.0)
        assert widget._level >= 0.0

    def test_repeated_loud_input_converges_toward_full(self):
        widget = MicLevelIndicator()
        for _ in range(50):
            widget.show_level(MicLevelIndicator._MAX_RMS)
        assert widget._level > 0.95

    def test_first_call_shows_the_widget(self):
        widget = MicLevelIndicator()
        assert not widget.isVisible()
        widget.show_level(1000.0)
        assert widget.isVisible()

    def test_a_single_loud_chunk_does_not_instantly_max_the_bar(self):
        # Smoothing should mean one chunk nudges the level, not snaps it -
        # a flickery all-or-nothing bar would be harder to read at a
        # glance than a smoothly moving one.
        widget = MicLevelIndicator()
        widget.show_level(MicLevelIndicator._MAX_RMS)
        assert widget._level < 1.0
