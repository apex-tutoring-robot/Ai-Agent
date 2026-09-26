"""
Tests for FaceWidget's frame cross-fading - added to fix mouth/eye
animation looking like discrete image swaps instead of smooth motion
(mouth_open and blink were already interpolated every frame, but the old
rendering code threw that away by hard-selecting a single image instead
of blending between neighbors).

Tested via FaceWidget (a real QWidget, needs a QApplication) rather than
FaceAnimator - FaceAnimator's constructor calls cv2.namedWindow(), which
errors against the opencv-python-headless build this project actually
uses. The two classes share identical logic by design (see their
docstrings), so this coverage applies to both.
"""

import numpy as np
import pytest

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.face_widget import FaceWidget

FACE_DIR = "src/visuals/faces"


@pytest.fixture(scope="module")
def widget():
    return FaceWidget(FACE_DIR)


class TestBlendMouthFrames:
    def test_zero_returns_the_first_frame_unblended(self, widget):
        result = widget._blend_mouth_frames(0.0)
        assert np.array_equal(result, widget.talk_frames[0])

    def test_one_returns_the_last_frame_unblended(self, widget):
        result = widget._blend_mouth_frames(1.0)
        assert np.array_equal(result, widget.talk_frames[-1])

    def test_exact_frame_boundary_returns_that_frame_unblended(self, widget):
        # 5 frames -> boundaries at mouth_open 0, 0.25, 0.5, 0.75, 1.0
        result = widget._blend_mouth_frames(0.5)
        assert np.array_equal(result, widget.talk_frames[2])

    def test_midpoint_between_two_frames_is_an_actual_blend(self, widget):
        # 0.125 * 4 = 0.5 -> exactly halfway between frame 0 and frame 1
        result = widget._blend_mouth_frames(0.125)
        assert not np.array_equal(result, widget.talk_frames[0])
        assert not np.array_equal(result, widget.talk_frames[1])
        expected = (widget.talk_frames[0].astype(np.int32) + widget.talk_frames[1].astype(np.int32)) // 2
        assert np.allclose(result.astype(np.int32), expected, atol=2)

    def test_negative_value_clips_to_the_first_frame(self, widget):
        result = widget._blend_mouth_frames(-5.0)
        assert np.array_equal(result, widget.talk_frames[0])

    def test_value_above_one_clips_to_the_last_frame(self, widget):
        result = widget._blend_mouth_frames(5.0)
        assert np.array_equal(result, widget.talk_frames[-1])

    def test_output_shape_matches_input_frames(self, widget):
        result = widget._blend_mouth_frames(0.3)
        assert result.shape == widget.talk_frames[0].shape

    def test_output_dtype_is_uint8(self, widget):
        result = widget._blend_mouth_frames(0.3)
        assert result.dtype == np.uint8


class TestBlendBlink:
    def test_no_blink_returns_the_frame_unchanged(self, widget):
        base = widget.faces["neutral"]
        result = widget._blend_blink(base, blink=0.0)
        assert np.array_equal(result, base)

    def test_tiny_blink_below_threshold_is_unchanged(self, widget):
        base = widget.faces["neutral"]
        result = widget._blend_blink(base, blink=0.01)
        assert np.array_equal(result, base)

    def test_full_blink_is_close_to_pure_blinking_image(self, widget):
        base = widget.faces["neutral"]
        result = widget._blend_blink(base, blink=1.0)
        assert np.array_equal(result, widget.faces["blinking"])

    def test_half_blink_is_an_actual_blend_not_a_hard_switch(self, widget):
        base = widget.faces["neutral"]
        result = widget._blend_blink(base, blink=0.5)
        assert not np.array_equal(result, base)
        assert not np.array_equal(result, widget.faces["blinking"])
        expected = (base.astype(np.int32) + widget.faces["blinking"].astype(np.int32)) // 2
        assert np.allclose(result.astype(np.int32), expected, atol=2)

    def test_blinking_blends_onto_a_talking_frame_not_just_the_base_face(self, widget):
        # Regression check for the old bug: a blink used to replace the
        # frame outright, freezing/resetting mid-word mouth movement.
        talking_frame = widget.talk_frames[3]
        result = widget._blend_blink(talking_frame, blink=0.5)
        assert not np.array_equal(result, talking_frame)
        assert not np.array_equal(result, widget.faces["blinking"])
