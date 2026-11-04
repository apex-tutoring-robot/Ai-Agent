"""
Thin, persistent status bar across the very top of the screen - phone-
style (a slim strip with small status text, not a floating card sitting
on top of the face) - replacing AudioDeviceIndicator's and
CommitIndicator's floating pill widgets.

Built from direct customer-facing feedback (Discord): a teammate asked
whether mic/speaker names and the running commit were meant for
customers or developers, since they read as unpolished "developer UI"
sitting on top of a tutoring robot's face. The information itself is
still real diagnostic value for debugging a remote device - the fix
isn't removing it, it's moving it to where a phone puts exactly this
kind of small persistent status text (carrier/battery/wifi), which
reads as normal device chrome rather than a debug overlay.

Always visible once populated (same as the widgets it replaces) - not
a transient popup like VolumeIndicator, which stays separate (a real
phone's status bar doesn't show a volume change either - that's its
own momentary overlay elsewhere on screen).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget


class StatusBar(QWidget):
    """Self-contained HUD widget - call show_devices(mic, speaker) and/or
    show_commit(commit) as each becomes known (see JarvisBot.__init__/
    _recreate_audio_system and ui_signals.audio_devices_resolved). Call
    set_width() whenever the host window resizes, so text eliding is
    computed against the real screen width."""

    _HEIGHT = 26
    _PADDING = 14
    _GAP = 20  # between the mic/speaker text and the commit text

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._mic_name = ""
        self._speaker_name = ""
        self._commit = ""
        self.setFixedSize(400, self._HEIGHT)  # placeholder - set_width() gives the real value
        self.hide()

    def set_width(self, width: int) -> None:
        self.setFixedSize(max(200, width), self._HEIGHT)
        self.update()

    def show_devices(self, mic_name: str, speaker_name: str) -> None:
        self._mic_name = mic_name or "(none found)"
        self._speaker_name = speaker_name or "(none found)"
        self.update()
        self.show()
        self.raise_()

    def show_commit(self, commit: str) -> None:
        self._commit = commit or "unknown"
        self.update()
        self.show()
        self.raise_()

    @staticmethod
    def _font() -> QFont:
        return QFont("Comic Sans MS", 10)

    def _right_text(self) -> str:
        return f"v{self._commit}" if self._commit else ""

    def _left_text(self, metrics: QFontMetrics, right_width: int) -> str:
        """
        The mic/speaker text, elided to whatever's left after the right-
        aligned commit text - pulled out of paintEvent so it's testable
        on its own (same convention as ErrorBanner._relayout), since
        paintEvent itself isn't exercised in unit tests project-wide.
        """
        if not self._mic_name and not self._speaker_name:
            return ""
        full = f"Mic: {self._mic_name}    Spk: {self._speaker_name}"
        reserved = right_width + (self._GAP if right_width else 0)
        available = max(0, self.width() - 2 * self._PADDING - reserved)
        return metrics.elidedText(full, Qt.ElideRight, available)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(15, 15, 20, 235))

        font = self._font()
        painter.setFont(font)
        painter.setPen(QColor(225, 225, 230))
        metrics = QFontMetrics(font)
        text_y = int((self._HEIGHT + metrics.ascent() - metrics.descent()) / 2)

        right_text = self._right_text()
        right_width = metrics.horizontalAdvance(right_text)
        if right_text:
            painter.drawText(self.width() - self._PADDING - right_width, text_y, right_text)

        left_text = self._left_text(metrics, right_width)
        if left_text:
            painter.drawText(self._PADDING, text_y, left_text)
