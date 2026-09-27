"""
Tests for identity.face_matcher - see that module's docstring for how
basic this similarity check actually is, and why it's unverified against
any real face photo (this dev environment has no camera and no sample
face photos at all). What's genuinely testable here: graceful handling of
"no face found"/missing files, and the compare_signatures math in
isolation (which needs no image at all). The one thing NOT covered by any
test in this file is whether two DIFFERENT real photos of the SAME
person's face actually score above the match threshold - that needs a
real camera and real faces, which is exactly why this is flagged as
needing verification on the physical Pi before being trusted.
"""

import base64

import numpy as np
import pytest

from identity.face_matcher import compare_signatures, compute_signature


def _encode(vector: np.ndarray) -> str:
    """Same encoding compute_signature uses internally - lets tests build
    a signature string directly from a hand-picked vector, with no image
    or face detection involved."""
    return base64.b64encode(vector.astype(np.float32).tobytes()).decode("ascii")


class TestComputeSignatureGracefulFailures:
    def test_returns_none_for_a_nonexistent_file(self):
        assert compute_signature("does/not/exist.png") is None

    def test_returns_none_when_no_face_is_detected(self):
        # The project's own cartoon face art - stylized, not photographic,
        # so the Haar cascade (trained on real faces) correctly finds
        # nothing here. A real regression would be this suddenly matching.
        assert compute_signature("src/visuals/faces/happy.png") is None
        assert compute_signature("src/visuals/faces/neutral.png") is None


class TestCompareSignatures:
    def test_identical_vectors_score_near_one(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        sig = _encode(v)
        assert compare_signatures(sig, sig) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_near_zero(self):
        a = _encode(np.array([1.0, 0.0]))
        b = _encode(np.array([0.0, 1.0]))
        assert compare_signatures(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_score_negative(self):
        a = _encode(np.array([1.0, 1.0]))
        b = _encode(np.array([-1.0, -1.0]))
        assert compare_signatures(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_scaled_vectors_are_still_near_identical(self):
        # Cosine similarity ignores magnitude - a brighter/darker version
        # of the same pattern should still score as very similar.
        a = _encode(np.array([1.0, 2.0, 3.0]))
        b = _encode(np.array([2.0, 4.0, 6.0]))
        assert compare_signatures(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_mismatched_lengths_score_zero_not_an_error(self):
        a = _encode(np.array([1.0, 2.0, 3.0]))
        b = _encode(np.array([1.0, 2.0]))
        assert compare_signatures(a, b) == 0.0

    def test_zero_vector_scores_zero_not_a_division_error(self):
        a = _encode(np.array([0.0, 0.0]))
        b = _encode(np.array([1.0, 1.0]))
        assert compare_signatures(a, b) == 0.0
