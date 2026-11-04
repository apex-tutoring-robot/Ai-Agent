"""
Tests for MainWindow's StatusBar wiring - the thin, phone-style strip
across the top of the screen (see visuals/ui/status_bar.py) that
replaced the old AudioDeviceIndicator/CommitIndicator floating pills,
and the repositioning of the widgets that used to share that top-right
corner with them.

Same real-MainWindow-with-fullscreen=False pattern as
test_main_window_show_expression.py. get_current_commit is patched
where main_window.py looks it up, rather than relying on git actually
being on PATH during the test.
"""
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

import visuals.ui.main_window as main_window_module
from visuals.ui.main_window import MainWindow
from visuals.ui.status_bar import StatusBar


def _make_window(monkeypatch, commit="deadbee"):
    monkeypatch.setattr(main_window_module, "get_current_commit", lambda: commit)
    return MainWindow(signals=None, faces_dir=r"src\visuals\faces", fullscreen=False)


class TestStatusBarWiring:
    def test_shows_the_current_commit_on_construction(self, monkeypatch):
        window = _make_window(monkeypatch, commit="deadbee")
        assert isinstance(window.status_bar, StatusBar)
        assert window.status_bar._commit == "deadbee"

    def test_spans_the_full_window_width_at_the_very_top(self, monkeypatch):
        window = _make_window(monkeypatch)
        assert window.status_bar.x() == 0
        assert window.status_bar.y() == 0
        assert window.status_bar.width() == window.width()

    def test_position_status_bar_matches_the_current_window_width(self, monkeypatch):
        # Calling the positioning method directly (same as resizeEvent
        # does) rather than relying on resize() to actually deliver a
        # native resize event in a headless, never-shown test window.
        window = _make_window(monkeypatch)
        window.resize(1600, 900)
        window._position_status_bar()
        assert window.status_bar.width() == window.width() == 1600


class TestWidgetsRepositionedBelowTheStatusBar:
    def test_volume_indicator_clears_the_status_bar(self, monkeypatch):
        window = _make_window(monkeypatch)
        assert window.volume_indicator.y() >= window.status_bar.height()

    def test_error_banner_clears_the_status_bar(self, monkeypatch):
        window = _make_window(monkeypatch)
        assert window.error_banner.y() >= window.status_bar.height()
