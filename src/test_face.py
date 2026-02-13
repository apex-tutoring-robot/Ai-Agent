import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["DISPLAY"] = ":0"

import time
from visuals.faces.face_animator import FaceAnimator

face = FaceAnimator(face_dir="visuals/faces")
face.start_idle()
face.start()

print("Face should be visible now")

while True:
    time.sleep(1)
