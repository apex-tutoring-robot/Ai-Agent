"""
Tests for StatusBar - the thin, phone-style strip across the top of the
screen showing mic/speaker and the running commit (see status_bar.py's
own module docstring for why this replaced AudioDeviceIndicator's and
CommitIndicator's floating pill widgets).

Pure logic (visibility + text layout), no paintEvent exercised - same
convention as test_error_banner.py/test_mic_level_indicator.py
(verified live via rendered screenshots instead). _left_text/_right_text
are pulled out of paintEvent specifically so this is possible, same
reasoning as ErrorBanner._relayout.
"""

from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.status_bar import StatusBar


class TestVisibility:
    def test_starts_hidden(self):
        widget = StatusBar()
        assert not widget.isVisible()

    def test_show_devices_makes_it_visible(self):
        widget = StatusBar()
        widget.show_devices("USB Mic", "pulse")
        assert widget.isVisible()

    def test_show_commit_makes_it_visible(self):
        widget = StatusBar()
        widget.show_commit("a5c2570")
        assert widget.isVisible()

    def test_set_width_before_anything_shown_does_not_show_it(self):
        widget = StatusBar()
        widget.set_width(800)
        assert not widget.isVisible()


class TestRightText:
    def test_no_commit_is_empty(self):
        widget = StatusBar()
        assert widget._right_text() == ""

    def test_commit_is_prefixed_with_v(self):
        widget = StatusBar()
        widget.show_commit("a5c2570")
        assert widget._right_text() == "va5c2570"

    def test_falsy_commit_falls_back_to_unknown(self):
        widget = StatusBar()
        widget.show_commit("")
        assert widget._right_text() == "vunknown"


class TestLeftText:
    def _metrics(self, widget):
        return QFontMetrics(widget._font())

    def test_no_devices_is_empty(self):
        widget = StatusBar()
        widget.set_width(800)
        assert widget._left_text(self._metrics(widget), right_width=0) == ""

    def test_includes_both_mic_and_speaker(self):
        widget = StatusBar()
        widget.set_width(800)
        widget.show_devices("USB Mic", "pulse")
        text = widget._left_text(self._metrics(widget), right_width=0)
        assert "USB Mic" in text
        assert "pulse" in text

    def test_missing_name_falls_back_to_none_found(self):
        widget = StatusBar()
        widget.set_width(800)
        widget.show_devices("", "")
        text = widget._left_text(self._metrics(widget), right_width=0)
        assert "(none found)" in text

    def test_elides_when_too_narrow_to_fit(self):
        widget = StatusBar()
        widget.set_width(120)  # deliberately too narrow for the full text
        widget.show_devices(
            "A Fairly Long USB Audio Device Name",
            "Another Fairly Long Output Device Name",
        )
        metrics = self._metrics(widget)
        text = widget._left_text(metrics, right_width=0)
        full = "Mic: A Fairly Long USB Audio Device Name    Spk: Another Fairly Long Output Device Name"
        assert text != full
        assert len(text) < len(full)

    def test_reserves_space_for_the_right_text(self):
        # A wider right_width must leave less room for the left text -
        # otherwise the two would overlap instead of sharing the one line.
        widget = StatusBar()
        widget.set_width(300)
        widget.show_devices("A Fairly Long USB Audio Device Name", "pulse")
        metrics = self._metrics(widget)

        narrow_reserved = widget._left_text(metrics, right_width=10)
        wide_reserved = widget._left_text(metrics, right_width=150)

        assert len(wide_reserved) <= len(narrow_reserved)

    def test_does_not_crash_on_zero_width(self):
        widget = StatusBar()
        widget.set_width(200)  # set_width() already clamps to a minimum
        widget.show_devices("USB Mic", "pulse")
        metrics = self._metrics(widget)
        # A right_width larger than the whole bar shouldn't raise -
        # available space clamps to zero instead of going negative.
        text = widget._left_text(metrics, right_width=10_000)
        assert isinstance(text, str)


class TestSizing:
    def test_set_width_is_clamped_to_a_minimum(self):
        widget = StatusBar()
        widget.set_width(10)
        assert widget.width() == 200

    def test_set_width_sets_exact_width_above_the_minimum(self):
        widget = StatusBar()
        widget.set_width(1024)
        assert widget.width() == 1024

    def test_height_is_fixed_and_thin(self):
        widget = StatusBar()
        widget.set_width(1024)
        assert widget.height() == StatusBar._HEIGHT
        # "Thin" relative to the old pill widgets (56-70px) - the whole
        # point of this redesign (see module docstring).
        assert widget.height() < 40
