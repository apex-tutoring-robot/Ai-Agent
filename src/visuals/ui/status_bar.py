"""
Thin, persistent status bar across the very top of the screen - phone-
style (a slim translucent strip with small status content, not a
floating card sitting on top of the face) - replacing
AudioDeviceIndicator's, CommitIndicator's, and MicLevelIndicator's
floating pill widgets.

Built from direct customer-facing feedback (Discord): a teammate asked
whether mic/speaker names and the running commit were meant for
customers or developers, since they read as unpolished "developer UI"
sitting on top of a tutoring robot's face. The information itself is
still real diagnostic value for debugging a remote device - the fix
isn't removing it, it's moving it (and the live mic-level meter/barge-in
indicator that joined it next) to where a phone puts exactly this kind
of small persistent status content (carrier/battery/wifi), which reads
as normal translucent device chrome rather than a solid debug overlay.

Always visible once populated (same as the widgets it replaces) - not
a transient popup like VolumeIndicator, which stays separate (a real
phone's status bar doesn't show a volume change either - that's its
own momentary overlay elsewhere on screen).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget


class StatusBar(QWidget):
    """Self-contained HUD widget - call show_devices(mic, speaker),
    show_commit(commit), show_level(rms), and/or show_barge_in_state
    (active) as each becomes known (see JarvisBot.__init__/
    _recreate_audio_system, ui_signals.audio_devices_resolved/
    mic_level_changed/barge_in_state_changed). Call set_width() whenever
    the host window resizes, so text eliding is computed against the
    real screen width."""

    _HEIGHT = 30
    _PADDING = 14
    _GAP = 16  # between adjacent sections (mic/speaker | meter | commit)

    # Live mic-level meter - same normalization/smoothing as the
    # MicLevelIndicator this absorbed, so the feel on screen is
    # unchanged for anyone already used to reading it.
    _MAX_RMS = 3000.0
    _SMOOTHING = 0.3
    _METER_WIDTH = 60
    _METER_HEIGHT = 10
    _METER_SEGMENTS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._mic_name = ""
        self._speaker_name = ""
        self._commit = ""
        self._level = 0.0  # 0.0-1.0, smoothed
        self._barge_in_active = False
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

    def show_level(self, rms: float) -> None:
        normalized = max(0.0, min(rms / self._MAX_RMS, 1.0))
        self._level += (normalized - self._level) * self._SMOOTHING
        self.update()
        if not self.isVisible():
            self.show()
            self.raise_()

    def show_barge_in_state(self, active: bool) -> None:
        self._barge_in_active = active
        self.update()
        if not self.isVisible():
            self.show()
            self.raise_()

    @staticmethod
    def _font(bold: bool = False) -> QFont:
        font = QFont("Comic Sans MS", 10)
        font.setBold(bold)
        return font

    def _right_text(self) -> str:
        return f"v{self._commit}" if self._commit else ""

    def _meter_left_x(self, right_width: int) -> int:
        """x-coordinate where the live level meter starts - immediately
        left of the commit text (if any), which stays right-aligned to
        the bar's own right edge."""
        edge = self.width() - self._PADDING
        if right_width:
            edge -= right_width + self._GAP
        return edge - self._METER_WIDTH

    def _left_text(self, metrics: QFontMetrics, meter_left_x: int) -> str:
        """
        The mic/speaker text, elided to whatever's left before the
        meter/commit cluster on the right - pulled out of paintEvent so
        it's testable on its own (same convention as
        ErrorBanner._relayout), since paintEvent itself isn't exercised
        in unit tests project-wide.
        """
        if not self._mic_name and not self._speaker_name:
            return ""
        full = f"Mic: {self._mic_name}    Spk: {self._speaker_name}"
        available = max(0, meter_left_x - self._GAP - self._PADDING)
        return metrics.elidedText(full, Qt.ElideRight, available)

    def _draw_level_meter(self, painter: QPainter, x: int) -> None:
        meter_y = (self._HEIGHT - self._METER_HEIGHT) / 2
        seg_gap = 2
        seg_w = (self._METER_WIDTH - seg_gap * (self._METER_SEGMENTS - 1)) / self._METER_SEGMENTS
        filled = round(self._level * self._METER_SEGMENTS)

        for i in range(self._METER_SEGMENTS):
            seg_x = x + i * (seg_w + seg_gap)
            color = QColor(90, 220, 140) if i < filled else QColor(120, 120, 128)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(seg_x), int(meter_y), int(seg_w), self._METER_HEIGHT, 2, 2)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._barge_in_active:
            # Gate is open right now - the state itself is the signal,
            # unmissable across the full width instead of the small
            # corner pill this took over from (see module docstring).
            painter.fillRect(self.rect(), QColor(200, 90, 20, 235))
            font = self._font(bold=True)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            metrics = QFontMetrics(font)
            text_y = int((self._HEIGHT + metrics.ascent() - metrics.descent()) / 2)
            painter.drawText(self._PADDING, text_y, "INTERRUPTED!")
            return

        # Translucent, not a solid block - this is meant to read as
        # overlay chrome (like a phone's status bar) rather than an
        # opaque debug panel sitting on the face. Light tint + dark text
        # (not the inverse) because this app's background is always
        # light (TeachingCanvas/FaceWidget both run light) - a dark
        # translucent fill blended with that backdrop washes out to a
        # pale grey, which a light-grey text color then has almost no
        # contrast against (confirmed live via a rendered screenshot:
        # the mic/speaker text was essentially invisible).
        painter.fillRect(self.rect(), QColor(255, 255, 255, 165))

        font = self._font()
        painter.setFont(font)
        painter.setPen(QColor(40, 40, 46))
        metrics = QFontMetrics(font)
        text_y = int((self._HEIGHT + metrics.ascent() - metrics.descent()) / 2)

        right_text = self._right_text()
        right_width = metrics.horizontalAdvance(right_text)
        if right_text:
            painter.drawText(self.width() - self._PADDING - right_width, text_y, right_text)

        meter_x = self._meter_left_x(right_width)
        self._draw_level_meter(painter, meter_x)

        # _draw_level_meter sets Qt.NoPen for the meter's own fills and
        # never had a reason to restore it - without this, drawText
        # below silently draws with no pen at all (found live: the
        # mic/speaker text rendered as nothing).
        painter.setPen(QColor(40, 40, 46))
        left_text = self._left_text(metrics, meter_x)
        if left_text:
            painter.drawText(self._PADDING, text_y, left_text)
