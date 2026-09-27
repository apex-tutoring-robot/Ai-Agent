"""
Tests for identity.identity_resolver - uses simple fake providers rather
than real ProfileManager/NameIdentityProvider, since this class only cares
about iteration/combination behavior over whatever IdentityResult a
provider returns.
"""

from identity.identity_provider import IdentityObservation, IdentityResult
from identity.identity_resolver import IdentityResolver


class FakeProvider:
    def __init__(self, result: IdentityResult):
        self._result = result

    def identify(self, observation: IdentityObservation) -> IdentityResult:
        return self._result


def unknown(name=None):
    return IdentityResult(status="unknown", profile_id=None, name=name, confidence=0.0)


def uncertain(profile_id=1, name="David", confidence=0.8):
    return IdentityResult(status="uncertain", profile_id=profile_id, name=name, confidence=confidence)


class TestIdentityResolver:
    def test_returns_unknown_when_no_providers_configured(self):
        resolver = IdentityResolver([])
        result = resolver.resolve(IdentityObservation(spoken_name="David"))
        assert result.status == "unknown"
        assert result.name == "David"

    def test_passes_through_a_single_providers_unknown_result(self):
        resolver = IdentityResolver([FakeProvider(unknown())])
        result = resolver.resolve(IdentityObservation(spoken_name="David"))
        assert result.status == "unknown"

    def test_passes_through_a_single_providers_uncertain_result(self):
        resolver = IdentityResolver([FakeProvider(uncertain())])
        result = resolver.resolve(IdentityObservation(spoken_name="David"))
        assert result.status == "uncertain"
        assert result.profile_id == 1
        assert result.name == "David"

    def test_falls_through_to_a_later_provider_when_the_first_is_unknown(self):
        resolver = IdentityResolver([FakeProvider(unknown()), FakeProvider(uncertain(profile_id=2, name="Emma"))])
        result = resolver.resolve(IdentityObservation(spoken_name="Emma"))
        assert result.status == "uncertain"
        assert result.profile_id == 2
        assert result.name == "Emma"

    def test_unknown_when_every_provider_is_unknown(self):
        resolver = IdentityResolver([FakeProvider(unknown()), FakeProvider(unknown())])
        result = resolver.resolve(IdentityObservation(spoken_name="Zorblax"))
        assert result.status == "unknown"

    def test_earlier_providers_take_priority_over_later_ones(self):
        resolver = IdentityResolver([FakeProvider(uncertain(profile_id=1, name="David")), FakeProvider(uncertain(profile_id=2, name="Emma"))])
        result = resolver.resolve(IdentityObservation(spoken_name="whoever"))
        assert result.profile_id == 1
        assert result.name == "David"
