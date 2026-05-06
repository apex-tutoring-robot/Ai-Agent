"""
StudyMemoryBackend protocol + JsonFileBackend implementation.

Protocol contract:
  - All methods are synchronous.
  - All methods receive and return plain Python dicts/lists/strings.
  - A future Mem0Backend or ZepBackend only needs to implement these seven methods
    and can be passed to StudySessionManager(backend=...) as a drop-in replacement.
"""

import json
import os
import re
import datetime
import logging
from typing import Optional, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StudyMemoryBackend(Protocol):
    """
    Structural protocol for study session storage backends.

    Any class implementing these seven methods is a valid backend —
    no inheritance required. @runtime_checkable allows isinstance() checks.

    Swap-in contract: implement all seven methods, pass an instance to
    StudySessionManager(backend=YourBackend(...)). Zero other changes needed.
    """

    def load(self) -> Dict[str, Any]:
        """
        Return the full student data dict.
        Shape:
          {
            "processed_syllabi": List[str],
            "study_plan": Optional[Dict]
          }
        Return an empty scaffold dict if no data exists yet.
        """
        ...

    def save(self, data: Dict[str, Any]) -> None:
        """Persist the full student data dict atomically."""
        ...

    def mark_syllabus_processed(self, filename: str) -> None:
        """
        Add filename to processed_syllabi list.
        Idempotent — safe to call multiple times.
        """
        ...

    def save_study_plan(self, plan: Dict[str, Any]) -> None:
        """
        Persist a newly generated study plan, replacing any existing one.
        plan shape:
          {
            "total_sessions": int,
            "source_summary": str,
            "sessions": List[SessionDict]
          }
        """
        ...

    def get_next_session(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent in_progress session if one exists, otherwise
        the first not_started session. Returns None if neither exists.
        Does NOT mutate status — caller marks in_progress via mark_session_in_progress.
        """
        ...

    def has_active_plan(self) -> bool:
        """Return True if a study_plan exists with at least one in_progress or not_started session."""
        ...

    def mark_session_in_progress(self, session_id: str) -> None:
        """Set the session with the given session_id to 'in_progress'."""
        ...

    def complete_session(self, session_id: str, summary_fields: Dict[str, Any]) -> None:
        """
        Mark the session as 'completed' and write the debrief fields onto it.
        summary_fields shape:
          {"date": str, "summary": str, "struggles": List[str], "next_focus": str}
        """
        ...


class JsonFileBackend:
    """
    Local JSON file backend. Satisfies the StudyMemoryBackend protocol.

    Storage layout:
        studyplans/
            index.json                      ← {processed_syllabi, active_plan}
            fractions-grade-5_2026-05-01.json  ← plan sessions + session_history
            algebra-intro_2026-04-10.json   ← older plans remain as a record

    Thread safety: all mutations use a temp file + os.replace() for atomicity.
    Reads parse fresh from disk so a restart always sees consistent state.
    """

    def __init__(self, studyplans_dir: str = "studyplans"):
        self._dir = studyplans_dir
        self._index_path = os.path.join(studyplans_dir, "index.json")
        os.makedirs(studyplans_dir, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: str, data: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as e:
            logger.error("JsonFileBackend: failed to write %s: %s", path, e)
            raise

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self._index_path):
            return {"processed_syllabi": [], "active_plan": None}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            idx.setdefault("processed_syllabi", [])
            idx.setdefault("active_plan", None)
            return idx
        except (json.JSONDecodeError, OSError) as e:
            logger.error("JsonFileBackend: failed to load index: %s", e)
            return {"processed_syllabi": [], "active_plan": None}

    def _save_index(self, index: Dict[str, Any]) -> None:
        self._atomic_write(self._index_path, index)

    def _active_plan_path(self, index: Dict[str, Any]) -> Optional[str]:
        name = index.get("active_plan")
        return os.path.join(self._dir, name) if name else None

    def _load_plan_file(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("JsonFileBackend: failed to load plan file %s: %s", path, e)
            return None

    def _plan_filename(self, source_summary: str) -> str:
        """Generate a unique, readable filename from the plan's source summary."""
        slug = re.sub(r"[^a-z0-9]+", "-", source_summary.lower()).strip("-")[:50]
        date = datetime.date.today().isoformat()
        base = f"{slug}_{date}"
        candidate = f"{base}.json"
        if not os.path.exists(os.path.join(self._dir, candidate)):
            return candidate
        for i in range(2, 1000):
            candidate = f"{base}_{i}.json"
            if not os.path.exists(os.path.join(self._dir, candidate)):
                return candidate
        # Extremely unlikely fallback
        import uuid as _uuid
        return f"{base}_{_uuid.uuid4().hex[:8]}.json"

    # ── Protocol methods ──────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        scaffold: Dict[str, Any] = {
            "processed_syllabi": [],
            "study_plan": None,
        }
        index = self._load_index()
        scaffold["processed_syllabi"] = index.get("processed_syllabi", [])

        plan_path = self._active_plan_path(index)
        if plan_path and os.path.exists(plan_path):
            plan_file = self._load_plan_file(plan_path)
            if plan_file:
                scaffold["study_plan"] = plan_file

        return scaffold

    def save(self, data: Dict[str, Any]) -> None:
        index = self._load_index()
        index["processed_syllabi"] = data.get("processed_syllabi", [])
        self._save_index(index)

        plan_path = self._active_plan_path(index)
        if plan_path and data.get("study_plan") is not None:
            self._atomic_write(plan_path, data["study_plan"])

    def mark_syllabus_processed(self, filename: str) -> None:
        index = self._load_index()
        if filename not in index["processed_syllabi"]:
            index["processed_syllabi"].append(filename)
            self._save_index(index)

    def save_study_plan(self, plan: Dict[str, Any]) -> None:
        filename = self._plan_filename(plan.get("source_summary", "plan"))
        self._atomic_write(os.path.join(self._dir, filename), plan)

        index = self._load_index()
        index["active_plan"] = filename
        self._save_index(index)
        logger.info("JsonFileBackend: new plan saved as %s", filename)

    def get_next_session(self) -> Optional[Dict[str, Any]]:
        data = self.load()
        if not data.get("study_plan"):
            return None
        sessions = data["study_plan"].get("sessions", [])

        # Prefer the most recently started in_progress session (paused/premature exit)
        in_progress = [s for s in sessions if s.get("status") == "in_progress"]
        if in_progress:
            return in_progress[-1]

        for session in sessions:
            if session.get("status") == "not_started":
                return session

        return None

    def has_active_plan(self) -> bool:
        return self.get_next_session() is not None

    def mark_session_in_progress(self, session_id: str) -> None:
        data = self.load()
        if data.get("study_plan"):
            for session in data["study_plan"].get("sessions", []):
                if session.get("session_id") == session_id:
                    session["status"] = "in_progress"
                    break
        self.save(data)

    def complete_session(self, session_id: str, summary_fields: Dict[str, Any]) -> None:
        data = self.load()
        if data.get("study_plan"):
            for session in data["study_plan"].get("sessions", []):
                if session.get("session_id") == session_id:
                    session["status"] = "completed"
                    session.update(summary_fields)
                    break
        self.save(data)
