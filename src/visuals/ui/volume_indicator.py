"""
Mobile-style transient volume HUD: a small pill that pops up in the corner
of the screen whenever Jarvis's speaking volume changes (see
JarvisBot._tool_set_volume in main.py), then fades out on its own - the same
pattern as a phone's on-screen volume overlay, so someone watching can see
that "please talk louder" actually did something, not just hear it.
"""

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QPointF
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QWidget, QGraphicsOpacityEffect


class VolumeIndicator(QWidget):
    """Self-contained HUD widget - just call show_level(volume) to pop it up."""

    _WIDTH = 190
    _HEIGHT = 56
    _SEGMENTS = 10
    _AUTO_HIDE_MS = 1600
    _MAX_VOLUME = 2.0  # keep in sync with AudioPlayer's clamp ceiling

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._level = 0.5  # 0.0-1.0, fraction of _MAX_VOLUME

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(250)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        self.hide()

    def show_level(self, volume: float) -> None:
        """volume: the raw AudioPlayer.volume value (0.0-2.0)."""
        self._level = max(0.0, min(volume / self._MAX_VOLUME, 1.0))
        self.update()
        self.show()
        self.raise_()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        self._hide_timer.start(self._AUTO_HIDE_MS)

    def _fade_out(self) -> None:
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self._WIDTH, self._HEIGHT, 16, 16)
        painter.fillPath(bg_path, QColor(20, 20, 25, 210))

        # Speaker icon: box + cone
        icon_x, icon_y = 14, self._HEIGHT / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        cone = QPolygonF([
            QPointF(icon_x, icon_y - 6),
            QPointF(icon_x + 8, icon_y - 6),
            QPointF(icon_x + 16, icon_y - 14),
            QPointF(icon_x + 16, icon_y + 14),
            QPointF(icon_x + 8, icon_y + 6),
            QPointF(icon_x, icon_y + 6),
        ])
        painter.drawPolygon(cone)

        # Segmented level bar
        bar_x = icon_x + 30
        bar_w = self._WIDTH - bar_x - 14
        seg_gap = 3
        seg_w = (bar_w - seg_gap * (self._SEGMENTS - 1)) / self._SEGMENTS
        filled = round(self._level * self._SEGMENTS)

        for i in range(self._SEGMENTS):
            seg_x = bar_x + i * (seg_w + seg_gap)
            color = QColor(90, 200, 250) if i < filled else QColor(70, 70, 78)
            painter.setBrush(color)
            painter.drawRoundedRect(int(seg_x), int(self._HEIGHT / 2 - 10), int(seg_w), 20, 3, 3)
