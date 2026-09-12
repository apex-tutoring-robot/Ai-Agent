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
        for name in ["neutral", "thinking", "happy", "blinking"]:
            self.faces[name] = self._load(name + ".png")

        self.talk_frames = [self._load(f"talk{i}.png") for i in range(1, 6)]

        self.window = "CHIPPY"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

    def stop_talking(self):
        self.emotion = "neutral"
        self.is_talking = False

    def start_thinking(self):
        self.emotion = "thinking"
        self.is_talking = False

    def start_scanning(self):
        """Visual feedback while the camera captures an image."""
        self.emotion = "thinking"
        self.is_talking = False

    def start_talking(self):
        self.is_talking = True

    def stop_talking(self):
        self.is_talking = False
        self.external_mouth_level = 0.0 # immediately decay toward closed

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
            frame = self.faces[self.emotion].copy()

        self._blink_update()

        if self.blink > 0.7:
            frame = self.faces["blinking"].copy()

        return cv2.resize(frame, (1024, 600))

    # --------------------------------------------------
    # Render Loop
    # --------------------------------------------------

def render_forever(self):
        print("[FACE] Render loop started (MAIN THREAD)")
        while self.running:
            # 1. Generate the face frame FIRST
            frame = self._frame()

            # 2. Draw the warning over it if Wi-Fi is down
            if not self.is_connected:
                text = "NO WI-FI CONNECTION"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Center the text
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2

                # Draw a black background rectangle for readability
                pad = 10
                cv2.rectangle(frame, 
                              (text_x - pad, text_y - text_size[1] - pad), 
                              (text_x + text_size[0] + pad, text_y + pad), 
                              (0, 0, 0), cv2.FILLED)

                # Draw the red text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                            (0, 0, 255), thickness, cv2.LINE_AA)
            
            # 3. Show the modified frame (don't call self._frame() again here)
            cv2.imshow(self.window, frame)
            
            cv2.waitKey(1)
            time.sleep(1 / 60)  # 60 FPS

        cv2.destroyAllWindows()
                    self.is_connected = False
            except Exception:
                self.is_connected = False
                
            time.sleep(1) # Check once per second
