"""
Face-based IdentityProvider - see identity_provider.py's module docstring
for the overall seam design. Wraps face_matcher.py's deliberately basic
similarity check - see that module's docstring for exactly how basic,
and why it's unverified against any real face as of this writing.

Wired into IdentityResolver alongside NameIdentityProvider, but nothing
currently supplies an IdentityObservation.photo_path anywhere in main.py -
that would mean capturing a photo automatically (at wake-word time, most
likely), which is a real privacy/consent decision for a device aimed at
children, not something to bake in silently. Until that decision is made
explicitly, this provider is a harmless no-op: identify() always sees
photo_path=None and returns "unknown", exactly like having no camera at
all.
"""

from typing import List, Tuple

from identity.face_matcher import compare_signatures, compute_signature
from identity.identity_provider import IdentityObservation, IdentityResult

# Never returns "known" outright, same reasoning as NameIdentityProvider -
# this is a much cruder similarity measure than a real face-recognition
# model, so it should be confirmed out loud, not trusted blindly. Set
# higher than the 0.75 used for name matching since a false accept here
# would misattribute a child's answers to someone else's mastery record -
# exactly the "sibling" scenario the identity work was built to avoid.
_MATCH_THRESHOLD = 0.85


class FaceIdentityProvider:
    def __init__(self, profile_manager):
        self._profile_manager = profile_manager

    def identify(self, observation: IdentityObservation) -> IdentityResult:
        if not observation.photo_path:
            return IdentityResult(status="unknown", profile_id=None, name=None, confidence=0.0)

        signature = compute_signature(observation.photo_path)
        if signature is None:
            return IdentityResult(status="unknown", profile_id=None, name=None, confidence=0.0)

        enrolled: List[Tuple[int, str, str]] = self._profile_manager.get_all_face_signatures()
        best_id, best_name, best_score = None, None, 0.0
        for profile_id, name, stored_signature in enrolled:
            score = compare_signatures(signature, stored_signature)
            if score > best_score:
                best_id, best_name, best_score = profile_id, name, score

        if best_id is not None and best_score >= _MATCH_THRESHOLD:
            return IdentityResult(status="uncertain", profile_id=best_id, name=best_name, confidence=best_score)

        return IdentityResult(status="unknown", profile_id=None, name=None, confidence=0.0)
