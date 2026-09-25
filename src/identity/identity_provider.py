"""
Identity resolution seam: how Jarvis decides who is talking at the start
of a conversation, before anything gets attached to a specific student's
learning history.

NameIdentityProvider (fuzzy name matching against ProfileManager) is the
only implementation today. Azure Speaker Recognition was investigated and
is very likely discontinued (no capability listing in any current region,
including eastus; live API calls returned generic 500s), and a local
on-device voice-embedding model is unverified on Pi hardware - so voice
biometrics are deliberately out of scope for now, not a placeholder that
was half-built.

This is a Protocol + IdentityResolver specifically so a future
SpeakerIdentityProvider (voice embeddings, once benchmarked on real
hardware) can be added later purely by appending to the resolver's
provider list - JarvisBot, onboarding, and ProfileManager should not need
to change at all when that happens.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable

IdentityStatus = Literal["known", "uncertain", "unknown"]


@dataclass
class IdentityObservation:
    """
    What's available to identify a speaker from, at a given moment.
    spoken_name is the only field populated today - a future
    SpeakerIdentityProvider would add an audio/embedding field here
    without changing this shape for existing providers.
    """
    spoken_name: Optional[str] = None


@dataclass
class IdentityResult:
    """
    status:
      "known"     - confident enough to act on without confirmation
                    (no provider returns this today - see module docstring;
                    a name match is always "uncertain", never "known"
                    outright, since two people can share a name).
      "uncertain" - a candidate profile was found but should be confirmed
                    out loud before attaching this conversation to it.
      "unknown"   - no matching profile; caller should onboard as new.
    """
    status: IdentityStatus
    profile_id: Optional[int]
    name: Optional[str]
    confidence: float


@runtime_checkable
class IdentityProvider(Protocol):
    def identify(self, observation: IdentityObservation) -> IdentityResult:
        ...
