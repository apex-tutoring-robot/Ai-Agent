import os
import random
import numpy as np
import cv2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout


class FaceWidget(QWidget):
    def __init__(self, face_dir, parent=None):
        super().__init__(parent)

        self.face_dir = face_dir
        self.emotion = "neutral"
        self.is_talking = False
        self.mouth_open = 0.0
        self.external_mouth_level = 0.0
        self.blink = 0.0

        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking",
                     "encouraging", "surprised", "explaining"]:
            path = os.path.join(self.face_dir, name + ".png")
            if os.path.exists(path):
                self.faces[name] = self._load(name + ".png")
            else:
                self.faces[name] = None  # graceful fallback to neutral

        self.talk_frames = [self._load(f"talk{i}.png") for i in range(1, 6)]

        self.label = QLabel()
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent; border: none;")
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_face)
        self.timer.start(16)  # ~60 FPS

    def _load(self, name):
        path = os.path.join(self.face_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.resize(img, (600, 600))

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True
        # Preserve expressive emotions as the base frame while mouth animates on top
        if self.emotion not in ("encouraging", "explaining", "surprised", "happy"):
            self.emotion = "neutral"

    def stop_talking(self):
        self.is_talking = False
        self.external_mouth_level = 0.0
        # Only reset to neutral if not in a deliberate expressive state
        if self.emotion not in ("encouraging", "explaining", "surprised",
                                "thinking", "happy"):
            self.emotion = "neutral"

    def start_encouraging(self):
        self.emotion = "encouraging"
        self.is_talking = False

    def start_surprised(self):
        self.emotion = "surprised"
        self.is_talking = False

    def start_explaining(self):
        self.emotion = "explaining"
        self.is_talking = False

    def push_mouth_level(self, level: float):
        self.external_mouth_level = float(level)

    def _blink_update(self):
        if self.blink < 0.05 and random.random() < 0.01:
            self.blink = 1.0
        self.blink *= 0.85

    def _update_mouth(self):
        attack = 0.60
        release = 0.22

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
            face_img = self.faces.get(self.emotion) or self.faces["neutral"]
            frame = face_img.copy()

        self._blink_update()

        if self.blink > 0.7:
            blink_img = self.faces.get("blinking")
            if blink_img is not None:
                frame = blink_img.copy()

        return frame

    def update_face(self):
        frame = self._frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        self.label.setPixmap(
            pixmap.scaled(
                self.label.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        self.update_face()
        super().resizeEvent(event)
