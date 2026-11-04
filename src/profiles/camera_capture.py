"""
Avatar photo capture for the Arducam 5MP (OV5647) CSI camera module.

Display-only profile pictures, not biometric/facial-recognition data - see
the multi-profile memory plan. Uses picamera2 (the standard library for
CSI camera modules on modern Raspberry Pi OS), not OpenCV VideoCapture,
since this is a CSI ribbon-cable module, not a USB webcam.

picamera2 requires Pi-specific libcamera system bindings and cannot be
installed or imported on Windows - every call here is wrapped so a missing
camera/library degrades this one feature gracefully instead of crashing
the app, the same pattern already used for optional SpeexDSP support.

NOTE: not testable on Windows - needs verification on the real Pi with
this camera physically connected before trusting it works.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


def detect_camera() -> bool:
    """Best-effort presence check for picamera2 + a physical CSI camera,
    without opening a capture session (unlike capture_avatar_photo, which
    does a real capture) - just for a clear startup log line, so a
    missing/disconnected camera is visible immediately instead of only
    discovered later when avatar capture silently returns False."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        logger.info("Camera: picamera2 not installed (expected off-Pi) - avatar capture will be skipped.")
        return False

    try:
        cameras = Picamera2.global_camera_info()
    except Exception as e:
        logger.warning(f"Camera: detection check failed ({e}) - avatar capture will be skipped.")
        return False

    if not cameras:
        logger.warning("Camera: picamera2 is installed but no camera was detected - avatar capture will be skipped.")
        return False

    logger.info(f"Camera: detected ({cameras[0].get('Model', 'unknown model')}).")
    return True


def capture_avatar_photo(output_path: str, warmup_seconds: float = 1.0) -> bool:
    """
    Capture a single still photo to `output_path`. Returns True on success,
    False on any failure (missing library, no camera attached, capture
    error) - callers should treat False as "no avatar this time", not fatal.
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        logger.warning("picamera2 not available - skipping avatar capture (Pi/Linux only)")
        return False

    picam2 = None
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        picam2 = Picamera2()
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        # Let auto-exposure/white-balance settle before capturing, otherwise
        # the first frame is often too dark/off-color.
        time.sleep(warmup_seconds)

        picam2.capture_file(output_path)
        logger.info(f"📸 Captured avatar photo: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Avatar photo capture failed: {e}")
        return False

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
