"""
StudySessionManager — orchestrates the study session lifecycle.

Knows nothing about HOW data is stored. Delegates all persistence to the
StudyMemoryBackend. Responsible for:
  - Detecting new syllabus files in syllabus/
  - Extracting text from any supported file format
  - Driving LLM calls for plan generation and session summarization
  - Building the system-prompt context string injected during active sessions
"""

import os
import re
import uuid
import logging
import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from azure_services.llm_client import LLMClient
    from memory.backend import StudyMemoryBackend

logger = logging.getLogger(__name__)

# Sentinels emitted by the LLM — exported so main.py can import them.
SESSION_COMPLETE_TAG = "[SESSION_COMPLETE]"
SESSION_COMPLETE_RE = re.compile(r'\[SESSION_COMPLETE\]', re.IGNORECASE)

OFF_TOPIC_TAG = "[OFF_TOPIC]"
OFF_TOPIC_RE = re.compile(r'\[OFF_TOPIC\]', re.IGNORECASE)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".png", ".jpg", ".jpeg", ".pdf"}

_WORD_TO_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}


class StudySessionManager:
    """
    Orchestration layer for study session memory.

    Args:
        llm_client: Shared LLMClient instance (for plan generation, summarization,
                    and syllabus image extraction).
        backend:    Any object satisfying the StudyMemoryBackend protocol.
                    Defaults to JsonFileBackend when not provided.
        syllabus_dir: Directory where parents drop syllabus files.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        backend: Optional["StudyMemoryBackend"] = None,
        syllabus_dir: str = "syllabus",
    ):
        self.llm_client = llm_client
        self.syllabus_dir = syllabus_dir

        if backend is None:
            from memory.backend import JsonFileBackend
            self.backend = JsonFileBackend()
        else:
            self.backend = backend

        logger.info("StudySessionManager initialized (backend=%s)", type(self.backend).__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # Syllabus detection
    # ──────────────────────────────────────────────────────────────────────────

    def get_new_syllabus_file(self) -> Optional[str]:
        """
        Scan syllabus_dir for any file that has not yet been processed.

        Returns the full path of the first unprocessed file, or None.
        """
        if not os.path.isdir(self.syllabus_dir):
            return None
        data = self.backend.load()
        processed = set(data.get("processed_syllabi", []))
        for filename in sorted(os.listdir(self.syllabus_dir)):
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in _SUPPORTED_EXTENSIONS and filename not in processed:
                return os.path.join(self.syllabus_dir, filename)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Syllabus text extraction
    # ──────────────────────────────────────────────────────────────────────────

    def extract_syllabus_text(self, file_path: str) -> str:
        """
        Extract syllabus content as plain text from any supported format.

        Returns a text string on success, or a string starting with "ERROR:"
        that the caller can speak directly to the student.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in {".txt", ".md"}:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError as e:
                logger.error("extract_syllabus_text: failed to read %s: %s", file_path, e)
                return f"ERROR: Could not read the file {os.path.basename(file_path)}."

        elif ext in {".png", ".jpg", ".jpeg"}:
            try:
                import base64
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                data_url = f"data:{mime};base64,{b64}"
                return self.llm_client.extract_image_content(data_url)
            except Exception as e:
                logger.error("extract_syllabus_text: image extraction failed for %s: %s", file_path, e)
                return "ERROR: Could not read the syllabus image. Please try photographing it with the robot camera instead."

        elif ext == ".pdf":
            try:
                import PyPDF2
                text_parts = []
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                text = "\n".join(text_parts).strip()
                if text:
                    return text
                return "ERROR: The PDF appears to be a scanned image and has no readable text. Please photograph the syllabus with the robot camera instead."
            except ImportError:
                return "ERROR: PDF reading is not available. Please photograph the syllabus or save it as a text file."
            except Exception as e:
                logger.error("extract_syllabus_text: PDF extraction failed for %s: %s", file_path, e)
                return "ERROR: Could not read the PDF. Please photograph the syllabus instead."

        return f"ERROR: Unsupported file format {ext}. Please use a text file, image, or PDF."

    # ──────────────────────────────────────────────────────────────────────────
    # Syllabus processed marking
    # ──────────────────────────────────────────────────────────────────────────

    def mark_syllabus_processed(self, file_path: str) -> None:
        """Record the syllabus file (by basename) as processed and delete it."""
        self.backend.mark_syllabus_processed(os.path.basename(file_path))
        try:
            os.remove(file_path)
            logger.info("mark_syllabus_processed: deleted %s", file_path)
        except OSError as e:
            logger.warning("mark_syllabus_processed: could not delete %s: %s", file_path, e)

    # ──────────────────────────────────────────────────────────────────────────
    # Study plan generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_study_plan(self, syllabus_text: str, num_sessions: int) -> bool:
        """
        Call LLM to produce a structured study plan and persist it via backend.

        Returns True on success, False if the LLM call or schema validation fails.
        """
        import json as _json
        from memory.schemas import StudyPlan, session_plan_llm_schema

        session_schema = _json.dumps(session_plan_llm_schema(), indent=2)

        prompt = (
            f"You are a study planner for a K-8 student. Given the syllabus text below, "
            f"create a study plan with exactly {num_sessions} sessions.\n\n"
            f"Return a JSON object with this exact structure:\n"
            f'{{\n'
            f'  "total_sessions": {num_sessions},\n'
            f'  "source_summary": "<one sentence describing what the full syllabus covers>",\n'
            f'  "sessions": [<array of {num_sessions} session objects matching the schema below>]\n'
            f'}}\n\n'
            f"Each session object must match this schema exactly (no extra fields):\n"
            f"{session_schema}\n\n"
            f"Syllabus text:\n{syllabus_text}\n\n"
            f"Return only valid JSON. No markdown fences. No explanation."
        )

        try:
            raw = self.llm_client.generate_json_response(prompt, max_tokens=2000)
        except Exception as e:
            logger.error("generate_study_plan: LLM call failed: %s", e)
            return False

        try:
            plan = StudyPlan.model_validate_json(raw)
        except Exception as e:
            logger.error("generate_study_plan: schema validation failed: %s | raw=%s", e, raw[:300])
            return False

        self.backend.save_study_plan(plan.model_dump())
        logger.info("Study plan saved: %d sessions", plan.total_sessions)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Session state queries
    # ──────────────────────────────────────────────────────────────────────────

    def has_active_plan(self) -> bool:
        return self.backend.has_active_plan()

    def get_next_session(self) -> Optional[Dict[str, Any]]:
        return self.backend.get_next_session()

    def mark_session_in_progress(self, session_id: str) -> None:
        self.backend.mark_session_in_progress(session_id)

    # ──────────────────────────────────────────────────────────────────────────
    # System prompt context injection
    # ──────────────────────────────────────────────────────────────────────────

    def build_session_context(self, session: Dict[str, Any]) -> str:
        """
        Build the text block appended to the LLM system prompt for an active session.

        Tells Jarvis what to teach and when to emit [SESSION_COMPLETE].
        """
        focus = session.get("focus", "")
        topics = ", ".join(session.get("topics", []))
        concepts = ", ".join(session.get("key_concepts", []))
        practice = session.get("practice", "")

        # Inject carry-forward note from the last completed session if one exists
        data = self.backend.load()
        sessions = (data.get("study_plan") or {}).get("sessions", [])
        completed = [s for s in sessions if s.get("status") == "completed"]
        last_session_note = ""
        if completed:
            last = completed[-1]
            struggles = ", ".join(last.get("struggles", [])) or "none noted"
            next_focus = last.get("next_focus", "")
            last_session_note = (
                f"\nPrevious session note: student struggled with {struggles}. "
                f"Carry-forward focus: {next_focus}."
            )

        return (
            f"\n\n--- ACTIVE STUDY SESSION ---\n"
            f"Focus: {focus}\n"
            f"Topics to cover: {topics}\n"
            f"Key concepts: {concepts}\n"
            f"Practice: {practice}"
            f"{last_session_note}\n"
            f"When the student explicitly says they are done, or when you have covered "
            f"all the topics and key concepts above, end your response with exactly: "
            f"[SESSION_COMPLETE]\n"
            f"If the student asks something clearly unrelated to the session topics above, "
            f"respond with only: [OFF_TOPIC]\n"
            f"--- END SESSION CONTEXT ---"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Session completion
    # ──────────────────────────────────────────────────────────────────────────

    def complete_session(self, session: Dict[str, Any], messages: List[Dict]) -> None:
        """
        Summarize the session via LLM and persist the result.

        Called from a background daemon thread — errors are logged, not re-raised.

        Args:
            session:  The session dict returned by get_next_session().
            messages: Snapshot of conversation_manager.get_messages() taken
                      at the moment the session ended.
        """
        session_id = session.get("session_id", str(uuid.uuid4()))
        focus = session.get("focus", "unknown")

        transcript_lines = []
        for msg in messages[-30:]:
            role = "Student" if msg["role"] == "user" else "Jarvis"
            content = msg["content"]
            if isinstance(content, str):
                transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        prompt = (
            f'Summarize this tutoring session focused on "{focus}".\n\n'
            f"Conversation:\n{transcript}\n\n"
            f"Return JSON with this exact structure:\n"
            f'{{\n'
            f'  "summary": "<2-3 sentence summary of what was covered>",\n'
            f'  "struggles": ["<thing student struggled with>"],\n'
            f'  "next_focus": "<what to prioritise at the start of the next session>"\n'
            f'}}\n\n'
            f"Return only valid JSON. No markdown fences."
        )

        from memory.schemas import SessionPlan

        try:
            raw = self.llm_client.generate_json_response(prompt, max_tokens=400)
            import json as _json
            llm_data = _json.loads(raw)
            # Validate debrief fields by merging into the existing session object
            validated = SessionPlan.model_validate({**session, **llm_data})
            summary_fields = {
                "date": datetime.date.today().isoformat(),
                "summary": validated.summary,
                "struggles": validated.struggles,
                "next_focus": validated.next_focus,
            }
        except Exception as e:
            logger.error("complete_session: summarization failed: %s", e)
            summary_fields = {
                "date": datetime.date.today().isoformat(),
                "summary": f'Session on "{focus}" completed.',
                "struggles": [],
                "next_focus": "",
            }

        try:
            self.backend.complete_session(session_id, summary_fields)
            logger.info("Session %s marked complete and summarized", session_id)
        except Exception as e:
            logger.error("complete_session: backend.complete_session failed: %s", e)

    # ──────────────────────────────────────────────────────────────────────────
    # Input parsing utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_duration_weeks(text: str) -> Optional[float]:
        """
        Parse spoken syllabus duration into weeks.

        Understands: "two weeks", "a month", "3 months", "a semester",
        "one year", bare numbers (treated as weeks). Returns None if not found.
        """
        lowered = text.lower()

        def _word_val(s: str) -> Optional[float]:
            digit = re.search(r'\b(\d+(?:\.\d+)?)\b', s)
            if digit:
                return float(digit.group(1))
            for word, num in _WORD_TO_NUM.items():
                if re.search(rf'\b{word}\b', s):
                    return float(num)
            return None

        # semester / term → ~18 weeks
        if re.search(r'\b(semester|term|quarter)\b', lowered):
            return 18.0

        # year
        m = re.search(r'\b(\w+)\s+year', lowered)
        if m or re.search(r'\byear\b', lowered):
            n = _word_val(m.group(1) if m else "") if m else None
            return (n or 1.0) * 52.0

        # months
        m = re.search(r'(\w+)\s+month', lowered)
        if m or re.search(r'\bmonth\b', lowered):
            n = _word_val(m.group(1) if m else "") if m else None
            return (n or 1.0) * 4.0

        # weeks
        m = re.search(r'(\w+)\s+week', lowered)
        if m or re.search(r'\bweek\b', lowered):
            n = _word_val(m.group(1) if m else "") if m else None
            return n or 1.0

        # bare digit → weeks
        digit = re.search(r'\b(\d+(?:\.\d+)?)\b', lowered)
        if digit:
            return float(digit.group(1))

        return None

    @staticmethod
    def extract_hours_per_week(text: str) -> Optional[float]:
        """
        Parse spoken weekly study hours into a float.

        Understands: "two hours", "an hour", "1.5 hours", "half an hour",
        "thirty minutes". Returns None if not found.
        """
        lowered = text.lower()

        # half an hour / 30 minutes
        if re.search(r'\bhalf\b', lowered) or re.search(r'\b30\s*min', lowered):
            return 0.5

        # X hours and Y minutes  (e.g. "1 hour 30 minutes")
        m = re.search(r'(\d+(?:\.\d+)?)\s*hour[s]?\s*(?:and\s*)?(\d+)\s*min', lowered)
        if m:
            return float(m.group(1)) + float(m.group(2)) / 60.0

        # X minutes only
        m = re.search(r'(\d+(?:\.\d+)?)\s*min', lowered)
        if m:
            return float(m.group(1)) / 60.0

        # decimal digit + hours
        m = re.search(r'(\d+(?:\.\d+)?)\s*hour', lowered)
        if m:
            return float(m.group(1))

        # word number + hours / hour
        if re.search(r'\bhour', lowered):
            for word, num in _WORD_TO_NUM.items():
                if re.search(rf'\b{word}\b', lowered):
                    return float(num)

        # bare digit → hours
        digit = re.search(r'\b(\d+(?:\.\d+)?)\b', lowered)
        if digit:
            return float(digit.group(1))

        return None
