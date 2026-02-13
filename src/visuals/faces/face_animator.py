import cv2
import os
import time
import random
import numpy as np


class FaceAnimator:
    def __init__(self, face_dir):
        self.face_dir = face_dir
        self.running = True

        self.emotion = "neutral"
        self.is_talking = False

        self.mouth_open = 0.0
        self.blink = 0.0

        # Load images
        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load(name + ".png")

        self.talk_frames = [self._load(f"talk{i}.png") for i in range(1, 6)]

        self.window = "CHIPPY"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        self.last_audio_time = 0
        self.talk_hold_time = 0.4
        self.mouth_open = 0.0
        self.target_mouth = 0.0

        print("[FACE] Ready")

    def _load(self, name):
        path = os.path.join(self.face_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.resize(img, (600, 600))

    # ---------- Public API ----------

    def start_idle(self):
        self.emotion = "neutral"
        self.is_talking = False

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True

    def stop_talking(self):
        self.is_talking = False

    def shutdown(self):
        self.running = False

    def update_mouth(self, pcm_bytes):
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

        # Split into 20ms windows (320 samples at 16kHz)
        window_size = 320
        for i in range(0, len(pcm), window_size):
            window = pcm[i:i + window_size]
            if len(window) == 0:
                continue

            rms = np.sqrt(np.mean(window ** 2)) / 32768.0
            self.target_mouth = np.clip(rms * 10.0, 0.0, 1.0)
            self.last_audio_time = time.time()

            # tiny sleep so frames update while audio plays
            time.sleep(0.02)

    # ---------- Rendering ----------

    def _blink_update(self):
        if self.blink < 0.05 and random.random() < 0.01:
            self.blink = 1.0
        self.blink *= 0.85

    def _frame(self):
        # Smooth mouth motion (very important)
        self.mouth_open += (self.target_mouth - self.mouth_open) * 0.35

        talking_now = (
            self.is_talking or
            (time.time() - self.last_audio_time) < self.talk_hold_time
        )

        if talking_now and self.mouth_open > 0.05:
            idx = int(self.mouth_open * (len(self.talk_frames) - 1))
            frame = self.talk_frames[idx].copy()
        else:
            frame = self.faces[self.emotion].copy()

        self._blink_update()

        if self.blink > 0.05:
            frame = cv2.addWeighted(
                frame, 1 - self.blink,
                self.faces["blinking"], self.blink, 0
            )

        return cv2.resize(frame, (1280, 720))


    def render_forever(self):
        print("[FACE] Render loop started (MAIN THREAD)")
        while self.running:
            cv2.imshow(self.window, self._frame())
            cv2.waitKey(1)
            time.sleep(1 / 60)

        cv2.destroyAllWindows()
