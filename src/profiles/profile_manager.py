"""
Profile management for multi-person households sharing one Jarvis robot.

One household (one subscription) can have multiple profiles - e.g. each
kid in the family - each with their own persistent conversation history,
so tutoring context never bleeds between people. Identification happens
by spoken name (fuzzy-matched, since STT won't transcribe a name
identically every time) plus - in a later stage, not this module - voice
verification against an enrolled voiceprint.

Storage is a single local SQLite file. Deliberately local, not cloud: this
holds real conversation content (and will later hold voice enrollment
data), so it stays on-device unless a separate decision is made to sync it.
"""

import os
import sqlite3
import threading
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable, Optional

from conversation.state_manager import ConversationStateManager

logger = logging.getLogger(__name__)

# Below this similarity ratio, a spoken name is treated as "no match" rather
# than force-matching to the closest existing profile.
NAME_MATCH_THRESHOLD = 0.75

DEFAULT_PROFILE_NAME = "Guest"


class ProfileManager:
    """Owns per-profile persistent history and which profile is active."""

    def __init__(self, db_path: str, max_history: int = 20):
        self.db_path = db_path
        self.max_history = max_history
        self._lock = threading.Lock()
        self._conversation_managers: dict[int, ConversationStateManager] = {}

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        # Ensure there's always an active profile, even before any real
        # enrollment flow exists - keeps the rest of the app working
        # unchanged while the profile system is built up in stages.
        if self.get_active_profile_id() is None:
            default_id = self._find_profile_id_by_exact_name(DEFAULT_PROFILE_NAME)
            if default_id is None:
                default_id = self._create_profile(DEFAULT_PROFILE_NAME)
            self._set_state("active_profile_id", str(default_id))

        logger.info(f"ProfileManager initialized (db: {db_path})")

    # --------------------------------------------------
    # Schema
    # --------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    avatar_path TEXT,
                    voiceprint_id TEXT,
                    grade TEXT,
                    created_at TEXT NOT NULL,
                    last_active_at TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL REFERENCES profiles(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            self._conn.commit()

    # --------------------------------------------------
    # app_state helpers (active/previous profile, survives restarts)
    # --------------------------------------------------

    def _get_state(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM app_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def _set_state(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value)
            )
            self._conn.commit()

    def get_active_profile_id(self) -> Optional[int]:
        value = self._get_state("active_profile_id")
        return int(value) if value is not None else None

    def get_previous_profile_id(self) -> Optional[int]:
        value = self._get_state("previous_profile_id")
        return int(value) if value is not None else None

    # --------------------------------------------------
    # Profile CRUD
    # --------------------------------------------------

    def _find_profile_id_by_exact_name(self, name: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM profiles WHERE name = ?", (name,)
            ).fetchone()
        return row["id"] if row else None

    def is_default_profile(self, profile_id: int) -> bool:
        """True if `profile_id` is the auto-created placeholder, not a real enrolled person."""
        return self.get_profile_name(profile_id) == DEFAULT_PROFILE_NAME

    def _create_profile(self, name: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO profiles (name, created_at, last_active_at) VALUES (?, ?, ?)",
                (name, now, now)
            )
            self._conn.commit()
            profile_id = cursor.lastrowid
        logger.info(f"👤 Created new profile: '{name}' (id={profile_id})")
        return profile_id

    def find_profile_by_name(self, spoken_name: str) -> Optional[int]:
        """Fuzzy-match a spoken name against existing profile names."""
        normalized = spoken_name.strip().lower()
        if not normalized:
            return None

        with self._lock:
            rows = self._conn.execute("SELECT id, name FROM profiles").fetchall()

        best_id, best_ratio = None, 0.0
        for row in rows:
            ratio = SequenceMatcher(None, normalized, row["name"].strip().lower()).ratio()
            if ratio > best_ratio:
                best_id, best_ratio = row["id"], ratio

        if best_ratio >= NAME_MATCH_THRESHOLD:
            return best_id
        return None

    def find_or_create_profile(self, spoken_name: str) -> tuple[int, bool]:
        """Returns (profile_id, created) - created=True if a new profile was made."""
        existing_id = self.find_profile_by_name(spoken_name)
        if existing_id is not None:
            return existing_id, False
        return self._create_profile(spoken_name.strip()), True

    def set_avatar(self, profile_id: int, avatar_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE profiles SET avatar_path = ? WHERE id = ?",
                (avatar_path, profile_id)
            )
            self._conn.commit()

    def get_profile_name(self, profile_id: int) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM profiles WHERE id = ?", (profile_id,)
            ).fetchone()
        return row["name"] if row else None

    def set_grade(self, profile_id: int, grade: str) -> None:
        """`grade` should already be normalized (e.g. 'K', '1'..'8') - see
        knowledge.textbook_search.normalize_grade() for parsing a spoken answer."""
        with self._lock:
            self._conn.execute(
                "UPDATE profiles SET grade = ? WHERE id = ?",
                (grade, profile_id)
            )
            self._conn.commit()

    def get_grade(self, profile_id: int) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT grade FROM profiles WHERE id = ?", (profile_id,)
            ).fetchone()
        return row["grade"] if row else None

    # --------------------------------------------------
    # Switching
    # --------------------------------------------------

    def switch_to(self, profile_id: int) -> ConversationStateManager:
        """Make `profile_id` active, remembering the current one to go back to."""
        current = self.get_active_profile_id()
        if current is not None and current != profile_id:
            self._set_state("previous_profile_id", str(current))
        self._set_state("active_profile_id", str(profile_id))

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE profiles SET last_active_at = ? WHERE id = ?",
                (now, profile_id)
            )
            self._conn.commit()

        name = self.get_profile_name(profile_id)
        logger.info(f"🔀 Switched active profile to '{name}' (id={profile_id})")
        return self.get_conversation_manager(profile_id)

    def go_back(self) -> Optional[ConversationStateManager]:
        """Revert to whichever profile was active before the most recent switch."""
        previous_id = self.get_previous_profile_id()
        if previous_id is None:
            return None
        return self.switch_to(previous_id)

    # --------------------------------------------------
    # Conversation history (persisted, per profile)
    # --------------------------------------------------

    def _persist_message(self, profile_id: int, role: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (profile_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (profile_id, role, content, now)
            )
            self._conn.commit()

    def _load_recent_messages(self, profile_id: int, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE profile_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (profile_id, limit)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_conversation_manager(self, profile_id: Optional[int] = None) -> ConversationStateManager:
        """
        Get the (cached, or freshly hydrated from SQLite) ConversationStateManager
        for a profile, wired to auto-persist every new message. Defaults to
        whichever profile is currently active.
        """
        if profile_id is None:
            profile_id = self.get_active_profile_id()

        if profile_id in self._conversation_managers:
            return self._conversation_managers[profile_id]

        manager = ConversationStateManager(
            max_history=self.max_history,
            on_message=lambda role, content: self._persist_message(profile_id, role, content)
        )
        for msg in self._load_recent_messages(profile_id, self.max_history):
            manager.load_message(msg["role"], msg["content"])

        self._conversation_managers[profile_id] = manager
        return manager

    def close(self) -> None:
        with self._lock:
            self._conn.close()
