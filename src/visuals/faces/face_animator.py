import cv2
import os
import threading
import time
import random
import numpy as np


class FaceAnimator:
    def __init__(self, face_dir: str):
        self.face_dir = face_dir
        self.lock = threading.Lock()
        self.running = True

        self.emotion = "neutral"
        self.mouth_open = 0.0
        self.blink = 0.0

        # Load faces
        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load_image(f"{name}.png")

        self.talk_frames = [
            self._load_image(f"talk{i}.png") for i in range(1, 6)
        ]

        # Window
        self.window = "CHIPPY"
        cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.window,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        self.audio_delay = 0.5  # seconds (start with 0.1)

    def _load_image(self, filename):
        path = os.path.join(self.face_dir, filename)
        img = cv2.imread(path)

        if img is None:
            img = np.ones((600, 600, 3), dtype=np.uint8) * 255
            cv2.putText(
                img, filename, (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )

        # FORCE consistent size
        img = cv2.resize(img, (600, 600))
        return img


    def set_emotion(self, emotion: str):
        if emotion in self.faces:
            self.emotion = emotion

    def update_mouth(self, rms: float):
        now = time.time()

        if not hasattr(self, "_rms_queue"):
            self._rms_queue = []

        # push rms with timestamp
        self._rms_queue.append((now, rms))

        # apply delayed rms
        while self._rms_queue:
            t, val = self._rms_queue[0]
            if now - t >= self.audio_delay:
                self._rms_queue.pop(0)
                target = np.clip(val * 10.0, 0.0, 1.0)
                self.mouth_open += (target - self.mouth_open) * 0.4
            else:
                break


    def _update_blink(self):
        if self.blink < 0.01 and random.random() < 0.005:
            self.blink = 1.0
        self.blink *= 0.8

    def _compose_frame(self):
        # Choose base face depending on whether speaking
        if self.mouth_open > 0.05:
            idx = int(self.mouth_open * (len(self.talk_frames) - 1))
            frame = self.talk_frames[idx].copy()
        else:
            frame = self.faces[self.emotion].copy()

        # Blink overlay (optional)
        self._update_blink()
        if self.blink > 0.05:
            frame = cv2.addWeighted(
                frame,
                1.0 - self.blink,
                self.faces["blinking"],
                self.blink,
                0
            )
        self.mouth_open *= 0.85

        return cv2.resize(frame, (1280, 720))


    def _render_loop(self):
        while self.running:
            frame = self._compose_frame()
            cv2.imshow(self.window, frame)
            cv2.waitKey(1)
            time.sleep(1 / 60)

    def render_step(self):
        frame = self._compose_frame()
        cv2.imshow(self.window, frame)
        cv2.waitKey(1)
