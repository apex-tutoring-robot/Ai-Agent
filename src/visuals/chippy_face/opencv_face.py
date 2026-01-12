import cv2
import os
import threading
import time
import random
import numpy as np


class FaceAnimator:
    """
    OpenCV-based full-face animator.
    Displays full images only (no parts).
    """

    def __init__(self, face_dir: str):
        self.face_dir = face_dir
        self.lock = threading.Lock()
        self.running = True
        self.state = "neutral"

        # Load static faces
        self.faces = {}
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load_image(f"{name}.png")

        # Load talking frames
        self.talk_frames = []
        for i in range(1, 6):
            self.talk_frames.append(self._load_image(f"talk{i}.png"))

        # Setup fullscreen window
        self.window = "CHIPPY"
        cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.window,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        self.current_image = self.faces["neutral"]
        self._show(self.current_image)

        # Start blinking loop
        threading.Thread(target=self._blink_loop, daemon=True).start()

    def _load_image(self, filename: str):
        path = os.path.join(self.face_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"[WARN] Missing image: {path}")
            img = np.ones((600, 600, 3), dtype=np.uint8) * 255
            cv2.putText(
                img,
                filename,
                (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )
        return img

    def _show(self, img):
        with self.lock:
            try:
                h, w, _ = img.shape
                screen_w, screen_h = 1280, 720
                frame = cv2.resize(img, (screen_w, screen_h))
                cv2.imshow(self.window, frame)
                cv2.waitKey(1)
            except Exception:
                pass

    def set_state(self, state: str):
        with self.lock:
            self.state = state
            if state in self.faces:
                self.current_image = self.faces[state]
                self._show(self.current_image)

    def animate_talking(self, rms: float):
        """
        rms: audio energy between 0.0 – ~0.1
        """
        idx = min(int(rms * 10), len(self.talk_frames) - 1)
        self._show(self.talk_frames[idx])

    def _blink_loop(self):
        while self.running:
            time.sleep(random.uniform(4, 7))
            if self.state == "neutral":
                self._show(self.faces["blinking"])
                time.sleep(0.12)
                self._show(self.faces["neutral"])
