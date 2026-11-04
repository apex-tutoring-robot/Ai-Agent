"""
Tests for ErrorBanner - the full-width "something is fundamentally
broken" overlay driven by ui_signals.fatal_error (see that signal's own
docstring and error_banner.py's module docstring for the real silent-
startup-failure case this exists to make visible).

Pure logic (visibility + sizing), no paintEvent exercised - same
convention as test_mic_level_indicator.py/test_status_bar.py (verified
live via rendered screenshots instead).
"""

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.error_banner import ErrorBanner


class TestShowError:
    def test_starts_hidden(self):
        widget = ErrorBanner()
        assert not widget.isVisible()

    def test_show_error_makes_it_visible(self):
        widget = ErrorBanner()
        widget.show_error("Startup failed: something broke")
        assert widget.isVisible()

    def test_message_is_prefixed_with_a_warning_glyph(self):
        # A plain log-style message should read as unmistakably an
        # alert at a glance, not blend in with the other HUD widgets.
        widget = ErrorBanner()
        widget.show_error("Startup failed: boom")
        assert widget._message.startswith("⚠")
        assert "Startup failed: boom" in widget._message


class TestSizing:
    def test_width_matches_set_width(self):
        widget = ErrorBanner()
        widget.set_width(600)
        widget.show_error("short message")
        assert widget.width() == 600

    def test_longer_message_is_taller(self):
        widget = ErrorBanner()
        widget.set_width(300)
        widget.show_error("short")
        short_height = widget.height()

        widget.show_error(
            "A much longer error message that should wrap across "
            "several lines once it no longer fits on just one, given "
            "a fairly narrow fixed width to wrap against."
        )
        long_height = widget.height()

        assert long_height > short_height

    def test_height_is_capped_for_a_runaway_long_message(self):
        widget = ErrorBanner()
        widget.set_width(300)
        widget.show_error("line " * 500)
        assert widget.height() <= widget._MAX_HEIGHT

    def test_set_width_while_visible_relayouts_immediately(self):
        widget = ErrorBanner()
        widget.show_error("some message")
        widget.set_width(250)
        assert widget.width() == 250

    def test_set_width_before_any_error_shown_does_not_show_it(self):
        # set_width() is called on every resize, including before any
        # error has ever happened - it must not make the (empty, never
        # shown) banner pop up on its own.
        widget = ErrorBanner()
        widget.set_width(500)
        assert not widget.isVisible()
