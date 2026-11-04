"""
Small persistent HUD showing the git commit Jarvis is actually running
(see utils.version_info.get_current_commit) - lets someone standing in
front of the physical robot tell at a glance whether it picked up a
pushed fix, instead of "it's not working" reports being ambiguous
between "the new code has a bug" and "the device never got the new
code" (confirmed live as a real back-and-forth with a remote tester).
Same always-visible pattern as AudioDeviceIndicator, not a transient
popup like VolumeIndicator.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath
from PyQt5.QtWidgets import QWidget


class CommitIndicator(QWidget):
    """Self-contained HUD widget - call show_commit(commit_hash) once
    resolved at startup (see JarvisBot.__init__)."""

    _WIDTH = 190
    _HEIGHT = 40
    _PADDING = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._commit = ""
        self.hide()

    def show_commit(self, commit: str) -> None:
        self._commit = commit or "unknown"
        self.update()
        self.show()
        self.raise_()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self._WIDTH, self._HEIGHT, 16, 16)
        painter.fillPath(bg_path, QColor(20, 20, 25, 210))

        font = QFont("Comic Sans MS", 12)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        metrics = QFontMetrics(font)

        text_y = (self._HEIGHT + metrics.ascent() - metrics.descent()) / 2
        painter.drawText(self._PADDING, int(text_y), f"\U0001f527 {self._commit}")
