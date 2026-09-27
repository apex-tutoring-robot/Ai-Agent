"""
FaceMatcher - a deliberately BASIC face-similarity check built entirely
from what's already a dependency (opencv-python-headless's bundled Haar
cascade face detector + plain numpy) - not a real face-recognition model,
no new dependency added.

This is a starter scaffold for the "remembers kids and other family
members by their names and face" product requirement (AGENT Memory doc).
It can tell "this looks like the same person under similar conditions"
from "this is clearly someone else" via a crude pixel-intensity
similarity on the detected face region - it is NOT robust to lighting,
angle, or expression changes the way a real embedding-based recognizer
(e.g. a trained CNN, or dlib's face_recognition library) would be.
Upgrading this later means swapping this module's implementation, not
anything that calls FaceIdentityProvider.

UNVERIFIED against any real face photo as of this writing - built and
exercised only against synthetic/illustrated test images, since this dev
environment has no camera and no sample face photos at all (picamera2,
which the real capture path depends on, doesn't even import on Windows -
see profiles/camera_capture.py). This needs verification against the
real Arducam camera on the physical Pi before being trusted for anything.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FACE_SIZE = (64, 64)
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    # Loaded once, lazily - a CascadeClassifier load reads and parses an
    # XML model file, worth avoiding on every single call.
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    return _cascade


def _detect_and_crop_face(image: np.ndarray) -> Optional[np.ndarray]:
    """Returns a grayscale, contrast-equalized, fixed-size crop of the
    largest detected face, or None if no face was found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # Largest box - most likely the person closest to the camera, not a
    # false positive or someone in the background.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, _FACE_SIZE)
    return cv2.equalizeHist(face)


def compute_signature(image_path: str) -> Optional[str]:
    """
    Returns a base64-encoded signature for the face in `image_path`, or
    None if the file couldn't be read or no face was found in it - the
    caller should treat None as "face enrollment/matching unavailable for
    this photo", not an error.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not read image for face signature: {image_path}")
        return None
    face = _detect_and_crop_face(image)
    if face is None:
        logger.warning(f"No face detected in {image_path}")
        return None
    vector = face.astype(np.float32).flatten()
    return base64.b64encode(vector.tobytes()).decode("ascii")


def compare_signatures(signature_a: str, signature_b: str) -> float:
    """
    Cosine similarity between two signatures. Meaningful range for real
    face crops is roughly 0.0 (very different) to 1.0 (near-identical) -
    see module docstring for how basic this comparison actually is.
    """
    a = np.frombuffer(base64.b64decode(signature_a), dtype=np.float32)
    b = np.frombuffer(base64.b64decode(signature_b), dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
