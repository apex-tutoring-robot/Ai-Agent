"""
Tests for MainWindow's CommitIndicator wiring - the "what commit is
actually running" HUD (see visuals/ui/commit_indicator.py and
utils/version_info.py), added after a remote deployment confusion where
"it's not working" reports couldn't be told apart from "the device
never got the new code".

Same real-MainWindow-with-fullscreen=False pattern as
test_main_window_show_expression.py. get_current_commit is patched
where main_window.py looks it up, rather than relying on git actually
being on PATH during the test.
"""
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

import visuals.ui.main_window as main_window_module
from visuals.ui.commit_indicator import CommitIndicator
from visuals.ui.main_window import MainWindow


class TestCommitIndicatorWiring:
    def test_shows_the_current_commit_on_construction(self, monkeypatch):
        monkeypatch.setattr(main_window_module, "get_current_commit", lambda: "deadbee")
        window = MainWindow(signals=None, faces_dir=r"src\visuals\faces", fullscreen=False)
        assert isinstance(window.commit_indicator, CommitIndicator)
        assert window.commit_indicator._commit == "deadbee"

    def test_positioned_in_the_top_right_below_the_volume_indicator(self, monkeypatch):
        monkeypatch.setattr(main_window_module, "get_current_commit", lambda: "deadbee")
        window = MainWindow(signals=None, faces_dir=r"src\visuals\faces", fullscreen=False)
        assert window.commit_indicator.x() == window.volume_indicator.x()
        assert window.commit_indicator.y() > window.volume_indicator.y() + window.volume_indicator.height()
