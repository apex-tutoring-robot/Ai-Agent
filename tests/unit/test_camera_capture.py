"""
Tests for profiles.camera_capture.detect_camera() - the startup presence
check for picamera2 + a physical CSI camera.

picamera2 requires Pi-specific libcamera bindings and isn't installable on
Windows (see camera_capture.py's own docstring), so these tests inject a
fake picamera2 module into sys.modules rather than relying on the real
library - detect_camera() does a deferred `from picamera2 import
Picamera2` inside the function body specifically so this is possible.
"""

import sys
import types

import pytest

from profiles.camera_capture import detect_camera


@pytest.fixture(autouse=True)
def _clean_picamera2_module():
    # detect_camera() imports picamera2 fresh each call - make sure a
    # fake module injected by one test doesn't leak into the next.
    sys.modules.pop("picamera2", None)
    yield
    sys.modules.pop("picamera2", None)


def _install_fake_picamera2(camera_info):
    """camera_info: the list global_camera_info() should return, or an
    exception instance to raise instead."""
    fake_module = types.ModuleType("picamera2")

    class FakePicamera2:
        @staticmethod
        def global_camera_info():
            if isinstance(camera_info, Exception):
                raise camera_info
            return camera_info

    fake_module.Picamera2 = FakePicamera2
    sys.modules["picamera2"] = fake_module


class TestDetectCamera:
    def test_returns_false_when_picamera2_not_installed(self):
        # No fake module installed - the real import fails (expected on
        # Windows/this dev machine), same as production off-Pi.
        assert detect_camera() is False

    def test_returns_true_when_camera_detected(self):
        _install_fake_picamera2([{"Model": "ov5647"}])
        assert detect_camera() is True

    def test_returns_false_when_no_camera_attached(self):
        _install_fake_picamera2([])
        assert detect_camera() is False

    def test_returns_false_when_detection_raises(self):
        _install_fake_picamera2(RuntimeError("libcamera init failed"))
        assert detect_camera() is False
