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
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load(name + ".png")

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

    def stop_talking(self):
        self.emotion = "neutral"
        self.is_talking = False

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True

    def stop_talking(self):
        self.is_talking = False
        self.external_mouth_level = 0.0

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
            frame = self.faces[self.emotion].copy()

        self._blink_update()

        if self.blink > 0.7:
            frame = self.faces["blinking"].copy()

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
