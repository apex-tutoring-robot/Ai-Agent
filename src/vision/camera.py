import base64
import logging
import os
import tempfile
import time

from picamera2 import Picamera2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "captures")


class Camera:
    """Captures a single frame from the Pi CSI camera and returns it as base64 JPEG."""

    def __init__(self):
        # OV5647 max resolution
        self.width = 2592
        self.height = 1944

    def _capture_to_file(self, path: str):
        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError("No cameras detected by libcamera. Check CSI cable and run 'libcamera-hello --list-cameras'.")
        logger.debug(f"📷 Detected cameras: {cameras}")
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (self.width, self.height)}
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)  # allow auto-exposure and AWB to settle
        try:
            cam.capture_file(path)
        finally:
            cam.stop()
            cam.close()

    def capture_base64(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name

        try:
            self._capture_to_file(path)

            with open(path, "rb") as f:
                data = f.read()

            logger.info(f"📷 Image captured ({len(data)} bytes)")
            return base64.b64encode(data).decode("utf-8")
        finally:
            os.unlink(path)

    def capture_and_save(self, filename: str = None) -> str:
        """Capture an image, save it to captures/ and return the file path."""
        os.makedirs(CAPTURES_DIR, exist_ok=True)

        if filename is None:
            filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"

        path = os.path.join(CAPTURES_DIR, filename)
        self._capture_to_file(path)

        size = os.path.getsize(path)
        logger.info(f"📷 Image saved to {path} ({size} bytes)")
        return path
