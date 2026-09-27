"""
Qt-native port of FaceAnimator. Same blink/mouth/emotion state machine and
asset set, but rendered into a QLabel on a QTimer instead of a blocking
cv2.imshow loop - needed because this widget lives inside the same
QGraphicsScene as TeachingCanvas (see tutor_scene.py), and a blocking cv2
window can't be embedded alongside it.

Public method names/semantics (start_idle/start_thinking/start_talking/
stop_talking/push_mouth_level) are kept identical to FaceAnimator so nothing
about the animation behavior itself changes - only how it's driven and
rendered.
"""

import logging
import os
import random
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QRegion
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

logger = logging.getLogger(__name__)

# Expressions the product spec (AGENT Face Animation doc) calls for beyond
# the art that actually exists today (neutral/thinking/happy/blinking) -
# see _load_optional(). Each falls back to the closest existing face
# until real art is provided, rather than crashing on a missing file.
_EXPANSION_FALLBACKS = {
    "excited": "happy",
    "sad": "neutral",
    "confused": "neutral",
    "listening": "neutral",
}


class FaceWidget(QWidget):
    def __init__(self, face_dir, parent=None):
        super().__init__(parent)
        self.face_dir = face_dir

        self.emotion = "neutral"
        self.is_talking = False

        # Mouth state
        self.mouth_open = 0.0
        self.external_mouth_level = 0.0

        # Blink state
        self.blink = 0.0

        # Whether to clip this widget to a circle inscribed in its current
        # bounds - the small "webcam bubble" look used for teaching layout
        # (see TutorScene.show_teaching_layout), as opposed to the
        # full-screen square face shown during normal chat. Applied/cleared
        # via set_circular_mask(), kept in sync across resizes.
        self._circular_mask = False

        # Load images - the 4 originals must exist (a missing one here is a
        # real setup error), the expansion set degrades gracefully instead
        # (see _EXPANSION_FALLBACKS/_load_optional).
        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load(name + ".png")
        for name, fallback in _EXPANSION_FALLBACKS.items():
            self.faces[name] = self._load_optional(name + ".png", fallback)

        self.talk_frames = [self._load(f"talk{i}.png") for i in range(1, 6)]

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        # 60 FPS, matching FaceAnimator.render_forever()'s original rate
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // 60)

        print("[FACE] Ready")

    def set_circular_mask(self, enabled: bool) -> None:
        """Clips this widget to a circle inscribed in its current bounds
        (a square widget -> a perfect circle) - see _circular_mask."""
        self._circular_mask = enabled
        self._update_mask()

    def _update_mask(self) -> None:
        if self._circular_mask:
            self.setMask(QRegion(self.rect(), QRegion.Ellipse))
        else:
            self.clearMask()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_mask()

    # --------------------------------------------------
    # Asset loading
    # --------------------------------------------------

    def _load(self, name):
        path = os.path.join(self.face_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.resize(img, (600, 600))

    def _load_optional(self, name, fallback_emotion):
        """Like _load(), but returns the fallback emotion's already-loaded
        image instead of raising when `name` doesn't exist yet - see
        _EXPANSION_FALLBACKS."""
        path = os.path.join(self.face_dir, name)
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"No art for {name} yet - using '{fallback_emotion}' face instead")
            return self.faces[fallback_emotion]
        return cv2.resize(img, (600, 600))

    # --------------------------------------------------
    # Public API (same as FaceAnimator)
    # --------------------------------------------------

    def start_idle(self):
        self.emotion = "neutral"
        self.is_talking = False

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_happy(self):
        self.emotion = "happy"

    def start_excited(self):
        self.emotion = "excited"

    def start_confused(self):
        self.emotion = "confused"
        self.is_talking = False

    def start_sad(self):
        self.emotion = "sad"
        self.is_talking = False

    def start_listening(self):
        self.emotion = "listening"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True

    def stop_talking(self):
        self.is_talking = False
        self.external_mouth_level = 0.0  # immediately decay toward closed

    def push_mouth_level(self, level: float):
        # May be invoked via a queued Qt signal connection from the audio
        # thread (see ui_signals.mouth_level) - safe to just store the value.
        self.external_mouth_level = float(level)

    # --------------------------------------------------
    # Animation updates (identical logic to FaceAnimator)
    # --------------------------------------------------

    def _blink_update(self):
        if self.blink < 0.05 and random.random() < 0.01:
            self.blink = 1.0
        self.blink *= 0.85  # smooth decay

    def _update_mouth(self):
        attack = 0.60   # opens faster
        release = 0.22  # closes smoother

        target = self.external_mouth_level
        if target < 0.03:
            target = 0.0

        k = attack if target > self.mouth_open else release
        self.mouth_open += (target - self.mouth_open) * k

        if self.mouth_open < 0.01:
            self.mouth_open = 0.0

    def _frame(self):
        self._update_mouth()

        talking_now = self.is_talking or (self.mouth_open > 0.02)

        if talking_now and self.mouth_open > 0.05:
            idx = int(self.mouth_open * (len(self.talk_frames) - 1))
            idx = np.clip(idx, 0, len(self.talk_frames) - 1)
            frame = self.talk_frames[idx].copy()
        else:
            frame = self.faces[self.emotion].copy()

        self._blink_update()
        if self.blink > 0.7:
            frame = self.faces["blinking"].copy()

        return frame

    # --------------------------------------------------
    # Render tick (replaces FaceAnimator.render_forever()'s blocking loop)
    # --------------------------------------------------

    def _tick(self):
        frame = self._frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # QPixmap.fromImage() deep-copies the pixel data, so it's safe even
        # though `rgb` (the numpy buffer QImage was built on) goes out of
        # scope right after this line.
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width() or w, self.height() or h,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pixmap)
