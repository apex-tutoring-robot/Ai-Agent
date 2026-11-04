"""
Full-width, persistent error banner - the "something is fundamentally
broken" counterpart to the status bar and small corner HUD widgets
(StatusBar, MicLevelIndicator), which all assume the robot is
basically working and just offer extra detail. This is for the opposite
case: a fatal, otherwise-silent failure (see ui_signals.fatal_error) that
would otherwise leave the robot looking "on" but doing nothing - no boot
greeting, no response to the wake word, no visible symptom at all - with
the real reason only ever reaching a log file nobody at a remote
deployment can read.

Built for a real, live case: a startup failure inside
WakeWordDetector.start() (model load or mic-stream-open) is caught by
JarvisBot.run()'s outer except and used to just log-and-stop - the robot
went completely silent with nothing on screen to show for it, and nobody
on site could read the log to say why.

Never auto-hides once shown - unlike VolumeIndicator's transient pop-up,
whoever is physically at the robot needs to be able to read (or
photograph, to relay to the team) the message at their own pace, not
whatever a timer decides.
"""

from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath
from PyQt5.QtWidgets import QWidget


class ErrorBanner(QWidget):
    """Self-contained HUD widget - call show_error(message) any time a
    fatal, otherwise-silent failure happens (see ui_signals.fatal_error).
    Call set_width() whenever the host window resizes, same as the other
    corner widgets' own _position_*() methods, so wrapping/height are
    computed against the real screen width instead of a guess."""

    _PADDING = 16
    _MAX_HEIGHT = 220  # a runaway-long message still can't eat the whole screen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._message = ""
        self._width = 400  # placeholder - real value arrives via set_width() before first show
        self.hide()

    def set_width(self, width: int) -> None:
        self._width = max(200, width)
        if self.isVisible():
            self._relayout()

    def show_error(self, message: str) -> None:
        self._message = f"⚠ {message}"
        self._relayout()
        self.update()
        self.show()
        self.raise_()

    def _relayout(self) -> None:
        metrics = QFontMetrics(self._font())
        text_rect = metrics.boundingRect(
            QRect(0, 0, self._width - 2 * self._PADDING, 0),
            Qt.TextWordWrap,
            self._message,
        )
        height = min(self._MAX_HEIGHT, text_rect.height() + 2 * self._PADDING)
        self.setFixedSize(self._width, height)

    @staticmethod
    def _font() -> QFont:
        font = QFont("Comic Sans MS", 13)
        font.setBold(True)
        return font

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        painter.fillPath(bg_path, QColor(180, 30, 30, 235))

        painter.setFont(self._font())
        painter.setPen(QColor(255, 255, 255))
        text_rect = QRect(
            self._PADDING, self._PADDING,
            self.width() - 2 * self._PADDING, self.height() - 2 * self._PADDING
        )
        painter.drawText(text_rect, Qt.TextWordWrap, self._message)
