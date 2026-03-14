import os
import sys
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["DISPLAY"] = ":0"

import time
from pathlib import Path
# Add the 'src' directory to Python's module search path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from visuals.faces.face_animator import FaceAnimator

face_dir = str(src_dir / "visuals" / "faces")
face = FaceAnimator(face_dir=face_dir)
face.start_idle()
# face.start()

print("Face should be visible now")

while True:
    time.sleep(1)
