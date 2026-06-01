"""
StudySessionManager — orchestrates the study session lifecycle.

Knows nothing about HOW data is stored. Delegates all persistence to the
StudyMemoryBackend. Responsible for:
  - Detecting new syllabus files in syllabus/
  - Extracting text from any supported file format
  - Generating diagnostic questions to gauge student prior knowledge
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


class StudySessionManager:
    """
    Orchestration layer for study session memory.

    Args:
        llm_client:   Shared LLMClient instance (plan generation, summarization,
                      syllabus image extraction, diagnostic questions).
        backend:      Any object satisfying the StudyMemoryBackend protocol.
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
    # Diagnostic questions
    # ──────────────────────────────────────────────────────────────────────────

    def generate_diagnostic_questions(self, syllabus_text: str) -> List[str]:
        """
        Generate 2 conversational questions to gauge the student's prior knowledge.

        Returns a list of question strings. Falls back to generic questions on failure.
        """
        import json as _json

        prompt = (
            "You are a tutor assessing a K-8 student's prior knowledge. "
            "Based on the syllabus below, write exactly 2 short, conversational questions "
            "to gauge the student's current understanding. "
            "Questions should be open-ended but answerable in a sentence or two. "
            "Do NOT ask about how much time they have or scheduling. "
            "Return a JSON array of exactly 2 question strings. No explanation.\n\n"
            f"Syllabus:\n{syllabus_text}\n\n"
            "Return only valid JSON. Example: "
            "[\"What do you already know about fractions?\", "
            "\"Can you name any fraction types you have learned before?\"]"
        )

        try:
            raw = self.llm_client.generate_json_response(prompt, max_tokens=300)
            questions = _json.loads(raw)
            if isinstance(questions, list) and len(questions) >= 1:
                return [str(q) for q in questions[:2]]
        except Exception as e:
            logger.error("generate_diagnostic_questions: failed: %s", e)

        return [
            "What do you already know about this topic?",
            "What parts feel most confusing or unfamiliar to you?",
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Study plan generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_study_plan(
        self,
        syllabus_text: str,
        diagnostic_qa: List[Dict[str, str]],
    ) -> bool:
        """
        Call LLM to produce a structured study plan and persist it via backend.

        Args:
            syllabus_text:  Extracted syllabus content.
            diagnostic_qa:  List of {"question": ..., "answer": ...} dicts from
                            the pre-plan diagnostic conversation.

        Returns True on success, False if the LLM call or schema validation fails.
        """
        import json as _json
        from memory.schemas import StudyPlan, session_plan_llm_schema

        session_schema = _json.dumps(session_plan_llm_schema(), indent=2)
        qa_text = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in diagnostic_qa
        )

        prompt = (
            "You are a study planner for a K-8 student. "
            "Based on the syllabus and the student's diagnostic responses below, "
            "create a personalised study plan.\n\n"
            "Choose an appropriate number of sessions (typically 3–10) based on the "
            "syllabus scope and the student's apparent level from the diagnostic.\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "total_sessions": <number>,\n'
            '  "source_summary": "<one sentence describing what the syllabus covers>",\n'
            '  "assessment_summary": "<one sentence describing the student\'s prior knowledge level>",\n'
            '  "sessions": [<array of session objects matching the schema below>]\n'
            "}\n\n"
            f"Each session object must match this schema exactly (no extra fields):\n"
            f"{session_schema}\n\n"
            f"Syllabus:\n{syllabus_text}\n\n"
            f"Student diagnostic:\n{qa_text}\n\n"
            "Return only valid JSON. No markdown fences. No explanation."
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
        Includes partial-progress context if this session was previously interrupted.
        """
        focus = session.get("focus", "")
        topics = ", ".join(session.get("topics", []))
        concepts = ", ".join(session.get("key_concepts", []))
        practice = session.get("practice", "")

        # If this session was previously interrupted, surface what was already covered
        partial_note = ""
        if session.get("topics_covered"):
            covered = ", ".join(session["topics_covered"])
            partial_note = f"\nThis session was previously started. Already covered: {covered}."
            if session.get("summary"):
                partial_note += f" Progress so far: {session['summary']}."
            if session.get("performance_notes"):
                partial_note += f" {session['performance_notes']}."
            partial_note += " Resume from where we left off — do not repeat covered material."

        # Carry-forward note from the last fully completed session
        data = self.backend.load()
        sessions = (data.get("study_plan") or {}).get("sessions", [])
        completed = [s for s in sessions if s.get("status") == "completed"]
        last_session_note = ""
        if completed:
            last = completed[-1]
            struggles = ", ".join(last.get("struggles", [])) or "none noted"
            next_focus = last.get("next_focus", "")
            topics_covered = ", ".join(last.get("topics_covered", [])) or "not recorded"
            performance_notes = last.get("performance_notes", "")
            last_session_note = (
                f"\nPrevious session: topics covered: {topics_covered}. "
                f"Student struggled with: {struggles}. "
                + (f"Performance: {performance_notes}. " if performance_notes else "")
                + f"Carry-forward focus: {next_focus}."
            )

        return (
            f"\n\n--- ACTIVE STUDY SESSION ---\n"
            f"Focus: {focus}\n"
            f"Topics to cover: {topics}\n"
            f"Key concepts: {concepts}\n"
            f"Practice: {practice}"
            f"{partial_note}"
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
            "{\n"
            '  "summary": "<2-3 sentence summary of what was covered>",\n'
            '  "struggles": ["<concept or skill the student struggled with>"],\n'
            '  "next_focus": "<what to prioritise at the start of the next session>",\n'
            '  "topics_covered": ["<topic actually covered in this session>"],\n'
            '  "performance_notes": "<brief assessment of student engagement and understanding>"\n'
            "}\n\n"
            "Return only valid JSON. No markdown fences."
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
                "topics_covered": [],
                "performance_notes": "",
            }

        try:
            self.backend.complete_session(session_id, summary_fields)
            logger.info("Session %s marked complete and summarized", session_id)
        except Exception as e:
            logger.error("complete_session: backend.complete_session failed: %s", e)

    def save_session_progress(self, session: Dict[str, Any], messages: List[Dict]) -> None:
        """
        Save partial progress for a session the student is stepping away from.

        Session status stays 'in_progress' so it can be resumed later.
        Called from a background daemon thread — errors are logged, not re-raised.
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
            f'The student is taking a break from a tutoring session on "{focus}".\n\n'
            f"Conversation so far:\n{transcript}\n\n"
            "Summarise what was covered. Return JSON with this exact structure:\n"
            "{\n"
            '  "summary": "<what was covered so far in 1-2 sentences>",\n'
            '  "topics_covered": ["<topic actually covered so far>"],\n'
            '  "performance_notes": "<brief note on student understanding and engagement>"\n'
            "}\n\n"
            "Return only valid JSON. No markdown fences."
        )

        try:
            raw = self.llm_client.generate_json_response(prompt, max_tokens=300)
            import json as _json
            data = _json.loads(raw)
            partial_fields = {
                "summary": str(data.get("summary", "")),
                "topics_covered": list(data.get("topics_covered", [])),
                "performance_notes": str(data.get("performance_notes", "")),
            }
        except Exception as e:
            logger.error("save_session_progress: summarization failed: %s", e)
            partial_fields = {
                "summary": f'Partial session on "{focus}" — student took a break.',
                "topics_covered": [],
                "performance_notes": "",
            }

        try:
            self.backend.save_partial_progress(session_id, partial_fields)
            logger.info("Session %s partial progress saved", session_id)
        except Exception as e:
            logger.error("save_session_progress: backend call failed: %s", e)

