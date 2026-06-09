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

        # Mouth state
        self.mouth_open = 0.0
        self.external_mouth_level = 0.0

        # Blink state
        self.blink = 0.0

        # Load images
        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking",
                     "encouraging", "surprised", "explaining"]:
            path = os.path.join(self.face_dir, name + ".png")
            if os.path.exists(path):
                self.faces[name] = self._load(name + ".png")
            else:
                self.faces[name] = None

        self.talk_frames = [self._load(f"talk{i}.png") for i in range(1, 6)]

        self.window = "CHIPPY"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)

        print("[FACE] Ready")

    # --------------------------------------------------
    # Asset loading
    # --------------------------------------------------

    def _load(self, name):
        path = os.path.join(self.face_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.resize(img, (600, 600))

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True

    def stop_talking(self):
        self.is_talking = False
        self.external_mouth_level = 0.0

    def start_encouraging(self):
        self.emotion = "encouraging"
        self.is_talking = False

    def start_surprised(self):
        self.emotion = "surprised"
        self.is_talking = False

    def start_explaining(self):
        self.emotion = "explaining"
        self.is_talking = False

    def shutdown(self):
        self.running = False

    # --------------------------------------------------
    # Animation Updates
    # --------------------------------------------------

    def _blink_update(self):
        # Random blink trigger
        if self.blink < 0.05 and random.random() < 0.01:
            self.blink = 1.0
        self.blink *= 0.85  # smooth decay
        
    def push_mouth_level(self, level: float):
        # Called from audio thread; keep it dead simple
        self.external_mouth_level = float(level)


    def _update_mouth(self):
        attack = 0.60   # opens faster
        release = 0.22  # closes smoother

        target = self.external_mouth_level

        # Noise gate
        if target < 0.03:
            target = 0.0

        # Attack when rising, release when falling
        k = attack if target > self.mouth_open else release

        self.mouth_open += (target - self.mouth_open) * k

        if self.mouth_open < 0.01:
            self.mouth_open = 0.0


    # --------------------------------------------------
    # Frame Generation
    # --------------------------------------------------

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


        return cv2.resize(frame, (1280, 720))

    # --------------------------------------------------
    # Render Loop
    # --------------------------------------------------

    def render_forever(self):
        print("[FACE] Render loop started (MAIN THREAD)")
        while self.running:
            cv2.imshow(self.window, self._frame())
            cv2.waitKey(1)
            time.sleep(1 / 60)  # 60 FPS

        cv2.destroyAllWindows()
