"""
Tests for AudioDeviceIndicator._truncate() - the pure string logic behind
the on-screen mic/speaker health-check HUD (see ui_signals.
audio_devices_resolved and JarvisBot._announce_resolved_audio_devices).

Pure string logic, no Qt widget construction/QApplication needed - only
actually showing the widget (paintEvent) requires a real window, which
this doesn't exercise (verified live via rendered screenshots instead,
same as VolumeIndicator's own untested painting code).
"""

from visuals.ui.audio_device_indicator import AudioDeviceIndicator


class TestTruncate:
    def test_short_name_is_unchanged(self):
        assert AudioDeviceIndicator._truncate("Headset Mic") == "Headset Mic"

    def test_name_at_exactly_the_limit_is_unchanged(self):
        name = "x" * AudioDeviceIndicator._MAX_NAME_CHARS
        assert AudioDeviceIndicator._truncate(name) == name

    def test_long_name_is_truncated_with_ellipsis(self):
        name = "A Very Long Audio Device Name That Goes On And On"
        result = AudioDeviceIndicator._truncate(name)
        assert len(result) <= AudioDeviceIndicator._MAX_NAME_CHARS
        assert result.endswith("…")

    def test_truncation_does_not_leave_trailing_whitespace_before_ellipsis(self):
        # A name that happens to truncate right after a space shouldn't
        # leave "Foo Bar …" with an awkward gap before the ellipsis.
        name = "Foo Bar " + "x" * 40
        result = AudioDeviceIndicator._truncate(name)
        assert "  " not in result
        assert not result[:-1].endswith(" ")

    def test_empty_name_shows_a_placeholder(self):
        assert AudioDeviceIndicator._truncate("") == "(none found)"

    def test_none_shows_a_placeholder(self):
        assert AudioDeviceIndicator._truncate(None) == "(none found)"
