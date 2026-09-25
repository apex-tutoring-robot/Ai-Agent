"""
Name-based IdentityProvider - the only identity evidence Jarvis has today
(see identity_provider.py's module docstring for why voice biometrics
aren't in scope yet).
"""

from identity.identity_provider import IdentityObservation, IdentityResult

# A fuzzy name match is reported as "uncertain" with this confidence,
# never as "known" outright - two people can share a name (e.g.
# siblings), so the caller always confirms out loud before attaching a
# conversation's learning history to the matched profile. This is a
# fixed value, not a threshold - ProfileManager.find_profile_by_name()
# already enforces its own similarity threshold before returning a match.
_MATCH_CONFIDENCE = 0.8


class NameIdentityProvider:
    """Wraps ProfileManager.find_profile_by_name() as an IdentityProvider."""

    def __init__(self, profile_manager):
        self._profile_manager = profile_manager

    def identify(self, observation: IdentityObservation) -> IdentityResult:
        spoken_name = (observation.spoken_name or "").strip()
        if not spoken_name:
            return IdentityResult(status="unknown", profile_id=None, name=None, confidence=0.0)

        existing_id = self._profile_manager.find_profile_by_name(spoken_name)
        if existing_id is not None and not self._profile_manager.is_default_profile(existing_id):
            stored_name = self._profile_manager.get_profile_name(existing_id)
            return IdentityResult(
                status="uncertain", profile_id=existing_id, name=stored_name, confidence=_MATCH_CONFIDENCE
            )

        return IdentityResult(status="unknown", profile_id=None, name=spoken_name, confidence=0.0)
