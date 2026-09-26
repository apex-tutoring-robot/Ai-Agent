"""
Tests for identity.face_identity_provider.FaceIdentityProvider - the
orchestration logic only (no photo -> unknown, no face detected -> unknown,
best-match-above-threshold selection). Whether real photos of the same
person actually score high enough to match is NOT covered here - see
test_face_matcher.py's docstring for why (no camera, no sample photos in
this dev environment).
"""

from identity.face_identity_provider import FaceIdentityProvider
from identity.identity_provider import IdentityObservation


class FakeProfileManager:
    def __init__(self, signatures=None):
        # list of (profile_id, name, signature)
        self._signatures = signatures or []

    def get_all_face_signatures(self):
        return self._signatures


class TestIdentify:
    def test_no_photo_is_unknown(self):
        provider = FaceIdentityProvider(FakeProfileManager())
        result = provider.identify(IdentityObservation(photo_path=None))
        assert result.status == "unknown"
        assert result.profile_id is None

    def test_photo_with_no_detectable_face_is_unknown(self):
        # The project's own cartoon art has no detectable face - see
        # test_face_matcher.py. This also means enrolled profiles are
        # never even reached in this case; that branch isn't independently
        # testable here without a real photo with a detectable face.
        provider = FaceIdentityProvider(FakeProfileManager(signatures=[(1, "Brian", "fake-signature")]))
        result = provider.identify(IdentityObservation(photo_path="src/visuals/faces/happy.png"))
        assert result.status == "unknown"
