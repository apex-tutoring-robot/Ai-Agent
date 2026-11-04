"""
Live "is the mic actually picking anything up" meter - a segmented bar
that moves in real time with however loud whatever the mic is currently
capturing is, fed by WakeWordDetector.on_level and ContinuousVADCapture.
on_input_level (see ui_signals.mic_level_changed).

Built for a specific, real situation: a remote tester on a physical
device with a new USB mic reported the wake word never firing, but had
no way to run a terminal diagnostic (tests/test_microphone.py) and
couldn't be walked through one either. This needs zero technical
steps - just look at the screen while talking. A bar that visibly
moves when spoken to proves real audio is reaching the app (so any
remaining problem is a tunable threshold, not broken hardware); a bar
that stays flat while clearly audible speech is happening points
straight at the microphone/driver/gain instead, no further guessing
needed.

Always visible once started (same as AudioDeviceIndicator/
CommitIndicator), not a transient popup like VolumeIndicator - the
whole point is being able to watch it continuously while testing.

Also shows barge-in GATE state (show_barge_in_state, fed by
ContinuousVADCapture.on_barge_in_state via ui_signals.
barge_in_state_changed) as a distinct, unmistakable visual - a bright
orange pill reading "INTERRUPTED!" instead of the normal level bar.
Added after a remote report that the raw level bar alone was
ambiguous for diagnosing a "robot never gets interrupted" complaint:
the mic picks up the bot's OWN voice continuously through imperfect
echo cancellation, so the level bar reads "loud" throughout ordinary
playback regardless of whether a real interruption is happening, with
no way to tell the two apart by eye. This reflects the actual
post-AEC-and-VAD gate decision instead, which isn't fooled by that
ambient echo - it only lights up when the barge-in logic itself
actually opens the gate.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath
from PyQt5.QtWidgets import QWidget


class MicLevelIndicator(QWidget):
    """Self-contained HUD widget - call show_level(rms) continuously as
    audio is captured (see ui_signals.mic_level_changed)."""

    _WIDTH = 190
    _HEIGHT = 56
    _SEGMENTS = 10
    _PADDING = 14
    # Raw int16 RMS that counts as "the bar is full" - background noise
    # typically sits well under 500, clearly-audible speech commonly
    # lands in the low thousands, so this leaves headroom without
    # needing a quiet room to see the bar move at all.
    _MAX_RMS = 3000.0
    # Exponential smoothing so the bar reads as a level, not a flicker -
    # raw per-chunk RMS updates every 20-80ms, far faster than a human
    # glancing at the screen needs to resolve.
    _SMOOTHING = 0.3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._level = 0.0  # 0.0-1.0, smoothed
        self._barge_in_active = False
        self.hide()

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

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_path = QPainterPath()
        bg_color = QColor(200, 90, 20, 230) if self._barge_in_active else QColor(20, 20, 25, 210)
        bg_path.addRoundedRect(0, 0, self._WIDTH, self._HEIGHT, 16, 16)
        painter.fillPath(bg_path, bg_color)

        font = QFont("Comic Sans MS", 11)
        font.setBold(self._barge_in_active)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        metrics = QFontMetrics(font)
        label = "INTERRUPTED!" if self._barge_in_active else "Mic:"
        label_y = self._PADDING + metrics.ascent()
        painter.drawText(self._PADDING, int(label_y), label)

        if self._barge_in_active:
            # Gate is open right now - the state itself is the signal,
            # no need for the ordinary level bar underneath it too.
            return

        bar_y = label_y + 6
        bar_h = self._HEIGHT - bar_y - 10
        bar_w = self._WIDTH - 2 * self._PADDING
        seg_gap = 3
        seg_w = (bar_w - seg_gap * (self._SEGMENTS - 1)) / self._SEGMENTS
        filled = round(self._level * self._SEGMENTS)

        for i in range(self._SEGMENTS):
            seg_x = self._PADDING + i * (seg_w + seg_gap)
            color = QColor(90, 220, 140) if i < filled else QColor(70, 70, 78)
            painter.setBrush(color)
            painter.drawRoundedRect(int(seg_x), int(bar_y), int(seg_w), int(bar_h), 3, 3)
