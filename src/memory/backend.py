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
import logging
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = "config/student_data.json"


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
            "processed_schedules": List[str],
            "study_plan": Optional[Dict],
            "session_history": List[Dict]
          }
        Return an empty scaffold dict if no data exists yet.
        """
        ...

    def save(self, data: Dict[str, Any]) -> None:
        """Persist the full student data dict atomically."""
        ...

    def mark_schedule_processed(self, filename: str) -> None:
        """
        Add filename to processed_schedules list.
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
        Return the first session dict with status == "not_started", or None.
        Does NOT mutate status — caller mutates via complete_session.
        """
        ...

    def has_active_plan(self) -> bool:
        """Return True if a study_plan exists with at least one not_started session."""
        ...

    def complete_session(self, session_number: int, summary: Dict[str, Any]) -> None:
        """
        Mark session_number as 'completed' in study_plan.sessions.
        Append summary to session_history.
        summary shape:
          {"date": str, "session_number": int,
           "summary": str, "struggles": List[str], "next_focus": str}
        """
        ...


class JsonFileBackend:
    """
    Local JSON file backend. Satisfies the StudyMemoryBackend protocol.

    Thread safety: all mutations write to a temp file then os.replace() for
    atomicity. Reads parse fresh from disk so a restart always sees consistent state.
    """

    def __init__(self, data_path: str = _DEFAULT_DATA_PATH):
        self._path = data_path
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)

    def _empty_scaffold(self) -> Dict[str, Any]:
        return {
            "processed_schedules": [],
            "study_plan": None,
            "session_history": []
        }

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self._path):
            return self._empty_scaffold()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scaffold = self._empty_scaffold()
            scaffold.update(data)
            return scaffold
        except (json.JSONDecodeError, OSError) as e:
            logger.error("JsonFileBackend: failed to load %s: %s — returning empty scaffold", self._path, e)
            return self._empty_scaffold()

    def save(self, data: Dict[str, Any]) -> None:
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error("JsonFileBackend: failed to save %s: %s", self._path, e)
            raise

    def mark_schedule_processed(self, filename: str) -> None:
        data = self.load()
        if filename not in data["processed_schedules"]:
            data["processed_schedules"].append(filename)
            self.save(data)

    def save_study_plan(self, plan: Dict[str, Any]) -> None:
        data = self.load()
        data["study_plan"] = plan
        self.save(data)

    def get_next_session(self) -> Optional[Dict[str, Any]]:
        data = self.load()
        if not data.get("study_plan"):
            return None
        for session in data["study_plan"].get("sessions", []):
            if session.get("status") == "not_started":
                return session
        return None

    def has_active_plan(self) -> bool:
        return self.get_next_session() is not None

    def complete_session(self, session_number: int, summary: Dict[str, Any]) -> None:
        data = self.load()
        if data.get("study_plan"):
            for session in data["study_plan"].get("sessions", []):
                if session.get("session_number") == session_number:
                    session["status"] = "completed"
                    break
        data["session_history"].append(summary)
        self.save(data)
