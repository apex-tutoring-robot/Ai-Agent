"""
Tests for identity.name_identity_provider - against a real ProfileManager
(temp-file SQLite), not mocked, matching this project's testing style.
"""

import pytest

from profiles.profile_manager import ProfileManager
from identity.identity_provider import IdentityObservation
from identity.name_identity_provider import NameIdentityProvider


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_profiles.db")


@pytest.fixture
def manager(db_path):
    pm = ProfileManager(db_path=db_path, max_history=5)
    yield pm
    pm.close()


@pytest.fixture
def provider(manager):
    return NameIdentityProvider(manager)


class TestNameIdentityProvider:
    def test_unknown_when_no_name_given(self, provider):
        result = provider.identify(IdentityObservation(spoken_name=None))
        assert result.status == "unknown"
        assert result.profile_id is None

    def test_unknown_when_name_is_blank(self, provider):
        result = provider.identify(IdentityObservation(spoken_name="   "))
        assert result.status == "unknown"

    def test_unknown_when_no_matching_profile_exists(self, provider):
        result = provider.identify(IdentityObservation(spoken_name="Zorblax"))
        assert result.status == "unknown"
        assert result.profile_id is None
        assert result.name == "Zorblax"

    def test_uncertain_never_known_on_an_existing_name_match(self, manager, provider):
        profile_id, _ = manager.find_or_create_profile("Brian")
        result = provider.identify(IdentityObservation(spoken_name="Brian"))
        assert result.status == "uncertain"
        assert result.profile_id == profile_id
        assert result.name == "Brian"
        assert result.confidence > 0

    def test_fuzzy_match_still_reported_as_uncertain(self, manager, provider):
        manager.find_or_create_profile("Brian")
        result = provider.identify(IdentityObservation(spoken_name="Bryan"))
        assert result.status == "uncertain"

    def test_default_guest_profile_is_never_matched(self, provider):
        """The auto-created default profile shouldn't be mistaken for a
        real match just because someone happens to say 'Guest'."""
        result = provider.identify(IdentityObservation(spoken_name="Guest"))
        assert result.status == "unknown"
        assert result.profile_id is None
