import pygame
import os
import random
import threading
import time

class ChippyAnimator:
    def __init__(self, face_folder):
        pygame.init()
        self.face_folder = face_folder
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("CHIPPY Face")

        # Load static faces
        self.faces = {
            "neutral": pygame.image.load(os.path.join(face_folder, "neutral.png")),
            "thinking": pygame.image.load(os.path.join(face_folder, "thinking.png")),
            "listening": pygame.image.load(os.path.join(face_folder, "thinking.png")),
            "happy": pygame.image.load(os.path.join(face_folder, "smiling.png")),
            "sleepy": pygame.image.load(os.path.join(face_folder, "sleepy.png")),
            "sad": pygame.image.load(os.path.join(face_folder, "sad.png")),
        }

        # Load talking mouth frames
        self.mouth_frames = [
            pygame.image.load(os.path.join(face_folder, f"talk{i}.png"))
            for i in range(1, 6)
        ]

        self.current_image = self.faces["neutral"]
        self.running = True
        self.state = "neutral"

        # Start a background blinking thread
        threading.Thread(target=self._blink_loop, daemon=True).start()

    def _show_image(self, image):
        self.screen.fill((255, 255, 255))
        self.screen.blit(image, (0, 0))
        pygame.display.flip()

    def set_state(self, state):
        """Switch between listening, thinking, talking, happy, etc."""
        if state == self.state:
            return

        self.state = state
        if state in self.faces:
            self._show_image(self.faces[state])
        elif state == "talking":
            threading.Thread(target=self._talking_loop, daemon=True).start()
        else:
            self._show_image(self.faces["neutral"])

    def _talking_loop(self):
        """Animate mouth movement while TTS is playing."""
        while self.state == "talking":
            frame = random.choice(self.mouth_frames)
            self._show_image(frame)
            time.sleep(random.uniform(0.05, 0.12))  # natural jitter

        # After talking ends
        self._show_image(self.faces["happy"])

    def _blink_loop(self):
        """Runs forever, blinking every few seconds."""
        blink_img = pygame.image.load(os.path.join(self.face_folder, "blinking.png"))
        while self.running:
            time.sleep(random.uniform(4, 7))
            if self.state not in ("talking", "thinking"):
                self._show_image(blink_img)
                time.sleep(0.15)
                self._show_image(self.faces[self.state])
