import cv2
import base64
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Camera:
    """Captures a single frame from a USB camera and returns it as base64 JPEG."""

    def __init__(self):
        self.device_index = int(os.getenv("CAMERA_DEVICE_INDEX", 0))

    def capture_base64(self) -> str:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera (device {self.device_index})")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Camera capture failed (device {self.device_index})")
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info(f"📷 Image captured ({len(buf)} bytes)")
        return base64.b64encode(buf).decode('utf-8')
