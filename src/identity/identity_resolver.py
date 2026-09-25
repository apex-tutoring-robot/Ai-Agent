"""
Combines evidence from one or more IdentityProvider implementations into a
single decision. Today this is a trivial pass-through over the one
provider Jarvis has (NameIdentityProvider) - the seam exists so a future
SpeakerIdentityProvider (voice embeddings) plugs in here, blending voice
and name evidence, without JarvisBot/onboarding/ProfileManager needing to
change at all. See identity_provider.py for the shared types.
"""

from typing import List

from identity.identity_provider import IdentityObservation, IdentityProvider, IdentityResult


class IdentityResolver:
    def __init__(self, providers: List[IdentityProvider]):
        self._providers = providers

    def resolve(self, observation: IdentityObservation) -> IdentityResult:
        for provider in self._providers:
            result = provider.identify(observation)
            if result.status != "unknown":
                return result
        return IdentityResult(status="unknown", profile_id=None, name=observation.spoken_name, confidence=0.0)
