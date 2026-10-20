"""
Small persistent HUD showing which mic/speaker Jarvis actually picked
(see audio.device_detection.resolve_device_index) - a "health check" so
someone standing in front of the physical robot can glance at the
screen and confirm the right devices were selected, instead of needing
to find and read the startup logs (which may not even be visible on a
deployed/headless Pi). Unlike VolumeIndicator (which pops up and fades
for a transient event), this stays visible once populated - the whole
point is being able to check it at any time, not just right at boot.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath
from PyQt5.QtWidgets import QWidget


class AudioDeviceIndicator(QWidget):
    """Self-contained HUD widget - call show_devices(mic_name, speaker_name)
    once resolved (see JarvisBot.__init__/_recreate_audio_system)."""

    _WIDTH = 320
    _HEIGHT = 70
    _PADDING = 14
    _MAX_NAME_CHARS = 28  # keeps a long device name from overflowing the pill

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._mic_name = ""
        self._speaker_name = ""
        self.hide()

    def show_devices(self, mic_name: str, speaker_name: str) -> None:
        self._mic_name = self._truncate(mic_name)
        self._speaker_name = self._truncate(speaker_name)
        self.update()
        self.show()
        self.raise_()

    @classmethod
    def _truncate(cls, name: str) -> str:
        if not name:
            return "(none found)"
        if len(name) <= cls._MAX_NAME_CHARS:
            return name
        return name[: cls._MAX_NAME_CHARS - 1].rstrip() + "…"

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

        mic_y = self._PADDING + metrics.ascent()
        speaker_y = mic_y + metrics.height() + 4

        painter.drawText(self._PADDING, int(mic_y), f"\U0001f3a4 {self._mic_name}")
        painter.drawText(self._PADDING, int(speaker_y), f"\U0001f50a {self._speaker_name}")
