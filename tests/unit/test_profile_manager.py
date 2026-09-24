"""
Tests for profiles.profile_manager - real SQLite behavior against a
temp-file DB (not mocked), since the whole point of _migrate_schema is a
real schema-evolution bug that only showed up against a real on-disk file.
"""

import os
import tempfile

import pytest

from profiles.profile_manager import ProfileManager, DEFAULT_PROFILE_NAME


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_profiles.db")


@pytest.fixture
def manager(db_path):
    pm = ProfileManager(db_path=db_path, max_history=5)
    yield pm
    pm.close()


class TestDefaultProfile:
    def test_creates_default_guest_profile_on_first_run(self, manager):
        active_id = manager.get_active_profile_id()
        assert active_id is not None
        assert manager.get_profile_name(active_id) == DEFAULT_PROFILE_NAME
        assert manager.is_default_profile(active_id) is True

    def test_reopening_same_db_does_not_duplicate_default_profile(self, db_path):
        pm1 = ProfileManager(db_path=db_path)
        first_id = pm1.get_active_profile_id()
        pm1.close()

        pm2 = ProfileManager(db_path=db_path)
        assert pm2.get_active_profile_id() == first_id
        pm2.close()


class TestProfileCreationAndMatching:
    def test_find_or_create_makes_a_new_profile(self, manager):
        profile_id, created = manager.find_or_create_profile("Ryan Lewis")
        assert created is True
        assert manager.get_profile_name(profile_id) == "Ryan Lewis"
        assert manager.is_default_profile(profile_id) is False

    def test_find_or_create_reuses_existing_profile(self, manager):
        first_id, created = manager.find_or_create_profile("Ryan Lewis")
        assert created is True

        second_id, created_again = manager.find_or_create_profile("Ryan Lewis")
        assert created_again is False
        assert second_id == first_id

    def test_fuzzy_name_matching_tolerates_stt_variation(self, manager):
        # A real name-matching bug this session's work depends on: STT
        # won't transcribe a name identically every time.
        manager.find_or_create_profile("Ryan Lewis")
        matched_id = manager.find_profile_by_name("Ryan Lewis.")  # trailing punctuation, STT artifact
        assert matched_id is not None
        assert manager.get_profile_name(matched_id) == "Ryan Lewis"

    def test_unrelated_name_does_not_force_match(self, manager):
        manager.find_or_create_profile("Ryan Lewis")
        assert manager.find_profile_by_name("Completely Different Person") is None


class TestSwitching:
    def test_switch_to_updates_active_profile(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        manager.switch_to(profile_id)
        assert manager.get_active_profile_id() == profile_id

    def test_go_back_restores_previous_profile(self, manager):
        original_id = manager.get_active_profile_id()
        new_id, _ = manager.find_or_create_profile("Brian")

        manager.switch_to(new_id)
        assert manager.get_active_profile_id() == new_id

        manager.go_back()
        assert manager.get_active_profile_id() == original_id

    def test_go_back_with_no_previous_profile_returns_none(self, manager):
        assert manager.go_back() is None


class TestGrade:
    def test_set_and_get_grade(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        manager.set_grade(profile_id, "3")
        assert manager.get_grade(profile_id) == "3"

    def test_grade_defaults_to_none(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        assert manager.get_grade(profile_id) is None


class TestConceptMastery:
    """Per-student tutoring progress, added for the Jarvis-scaled tutoring
    loop (see JarvisBot._handle_teaching_answer) - tracks per-concept
    attempts/correctness/hints across sessions in the existing SQLite DB,
    deliberately not a separate curriculum-graph service."""

    def test_mastery_defaults_to_all_zero_for_a_new_concept(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        assert manager.get_mastery(profile_id, "equivalent_fractions") == {
            "attempts": 0, "correct_attempts": 0, "hints_used": 0,
        }

    def test_record_attempt_creates_a_row_on_first_attempt(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        manager.record_attempt(profile_id, "equivalent_fractions", correct=True, used_hint=False)
        assert manager.get_mastery(profile_id, "equivalent_fractions") == {
            "attempts": 1, "correct_attempts": 1, "hints_used": 0,
        }

    def test_record_attempt_accumulates_across_multiple_calls(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        manager.record_attempt(profile_id, "equivalent_fractions", correct=False, used_hint=False)
        manager.record_attempt(profile_id, "equivalent_fractions", correct=False, used_hint=True)
        manager.record_attempt(profile_id, "equivalent_fractions", correct=True, used_hint=False)

        assert manager.get_mastery(profile_id, "equivalent_fractions") == {
            "attempts": 3, "correct_attempts": 1, "hints_used": 1,
        }

    def test_different_concepts_are_tracked_independently(self, manager):
        profile_id, _ = manager.find_or_create_profile("Brian")
        manager.record_attempt(profile_id, "equivalent_fractions", correct=True, used_hint=False)
        manager.record_attempt(profile_id, "area_rectangle", correct=False, used_hint=True)

        assert manager.get_mastery(profile_id, "equivalent_fractions") == {
            "attempts": 1, "correct_attempts": 1, "hints_used": 0,
        }
        assert manager.get_mastery(profile_id, "area_rectangle") == {
            "attempts": 1, "correct_attempts": 0, "hints_used": 1,
        }

    def test_different_profiles_are_tracked_independently(self, manager):
        brian_id, _ = manager.find_or_create_profile("Brian")
        maya_id, _ = manager.find_or_create_profile("Maya")

        manager.record_attempt(brian_id, "equivalent_fractions", correct=True, used_hint=False)

        assert manager.get_mastery(brian_id, "equivalent_fractions")["attempts"] == 1
        assert manager.get_mastery(maya_id, "equivalent_fractions")["attempts"] == 0


class TestSchemaMigration:
    def test_migrating_a_pre_grade_column_database_does_not_lose_data(self, db_path):
        """
        Reproduces the exact live bug found this project: a DB created
        before the 'grade' column existed must gain it via ALTER TABLE
        without losing existing rows - CREATE TABLE IF NOT EXISTS alone
        does not add columns to an already-existing table.
        """
        import sqlite3
        from datetime import datetime, timezone

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                avatar_path TEXT,
                voiceprint_id TEXT,
                created_at TEXT NOT NULL,
                last_active_at TEXT
            )
        """)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO profiles (name, created_at, last_active_at) VALUES (?, ?, ?)",
            ("Pre-existing Kid", now, now),
        )
        conn.commit()
        conn.close()

        pm = ProfileManager(db_path=db_path)
        try:
            existing_id = pm.find_profile_by_name("Pre-existing Kid")
            assert existing_id is not None
            # The real bug: this raised "no such column: grade" before
            # _migrate_schema existed, and the uncaught exception took down
            # the whole speaker thread for the rest of the conversation.
            pm.set_grade(existing_id, "4")
            assert pm.get_grade(existing_id) == "4"
        finally:
            pm.close()


class TestConversationHistory:
    def test_conversation_manager_persists_across_reopen(self, db_path):
        pm1 = ProfileManager(db_path=db_path, max_history=5)
        profile_id = pm1.get_active_profile_id()
        conv1 = pm1.get_conversation_manager(profile_id)
        conv1.add_user_message("what is 2 plus 2")
        conv1.add_assistant_message("2 plus 2 is 4")
        pm1.close()

        pm2 = ProfileManager(db_path=db_path, max_history=5)
        conv2 = pm2.get_conversation_manager(profile_id)
        messages = conv2.get_messages()
        assert any(m["content"] == "what is 2 plus 2" for m in messages)
        assert any(m["content"] == "2 plus 2 is 4" for m in messages)
        pm2.close()
