import time
import random
from face_animator import FaceAnimator

face = FaceAnimator(face_dir=".")

print("Showing thinking face...")
face.set_emotion("thinking")
time.sleep(2)

print("Switching to neutral + fake talking")
face.set_emotion("neutral")

while True:
    fake_rms = random.uniform(0.0, 0.06)
    face.update_mouth(fake_rms)
    time.sleep(0.05)
