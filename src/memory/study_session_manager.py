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
        self.curriculum_dir = "curriculum"

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
    # Government / standard curriculum loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_curriculum_doc(self, topic: str) -> Optional[str]:
        """
        Find and return the text of a static curriculum document matching *topic*.

        Files live in self.curriculum_dir (default: "curriculum/").  The match is
        fuzzy: both the topic and each filename stem are normalised to lowercase
        with punctuation stripped, then the file whose stem has the highest
        word-overlap with the topic is chosen (must share at least one word).

        Supports .txt and .md files (plain read) and .pdf files (PyPDF2).
        Returns None when no match is found or the directory does not exist.
        """
        if not os.path.isdir(self.curriculum_dir):
            logger.warning("load_curriculum_doc: curriculum dir %r not found", self.curriculum_dir)
            return None

        def _normalise(s: str) -> set:
            import string
            s = s.lower().translate(str.maketrans("", "", string.punctuation))
            return set(s.split())

        topic_words = _normalise(topic)
        if not topic_words:
            return None

        best_path: Optional[str] = None
        best_score: int = 0

        for filename in sorted(os.listdir(self.curriculum_dir)):
            if filename.startswith("."):
                continue
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in {".txt", ".md", ".pdf"}:
                continue
            overlap = len(_normalise(stem) & topic_words)
            if overlap > best_score:
                best_score = overlap
                best_path = os.path.join(self.curriculum_dir, filename)

        if not best_path:
            logger.info("load_curriculum_doc: no match for topic %r", topic)
            return None

        logger.info("load_curriculum_doc: matched %r for topic %r", best_path, topic)
        ext = os.path.splitext(best_path)[1].lower()

        try:
            if ext in {".txt", ".md"}:
                with open(best_path, "r", encoding="utf-8") as f:
                    return f.read().strip() or None

            if ext == ".pdf":
                import PyPDF2
                text_parts = []
                with open(best_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                text = "\n".join(text_parts).strip()
                return text or None

        except Exception as e:
            logger.error("load_curriculum_doc: failed to read %s: %s", best_path, e)

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic questions
    # ──────────────────────────────────────────────────────────────────────────

    def identify_topics(self, syllabus_text: str) -> List[str]:
        """
        Return a list of distinct topics found in the syllabus.
        Returns an empty list on failure.
        """
        from memory.schemas import TopicList, topic_list_schema

        prompt = (
            "You are a tutor reviewing a K-8 syllabus. "
            "List the distinct topics or subject areas covered (typically 2-8 items).\n\n"
            f"Syllabus:\n{syllabus_text}"
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, topic_list_schema(), "topic_list", max_tokens=200
            )
            result = TopicList.model_validate_json(raw)
            if result.topics:
                return result.topics
        except Exception as e:
            logger.error("identify_topics: failed: %s", e)

        return []

    def generate_diagnostic_questions(self, syllabus_text: str, topic: str = "") -> List[str]:
        """
        Generate 2 conversational questions to gauge the student's prior knowledge.
        Returns a list of question strings. Falls back to generic questions on failure.
        """
        from memory.schemas import DiagnosticQuestions, diagnostic_questions_schema

        topic_clause = f' specifically about "{topic}"' if topic else ""
        prompt = (
            "You are a tutor assessing a K-8 student's prior knowledge. "
            f"Based on the syllabus below, write exactly 2 short, conversational questions "
            f"to gauge the student's current understanding{topic_clause}. "
            "Questions should be open-ended but answerable in a sentence or two. "
            "Do NOT ask about how much time they have or scheduling.\n\n"
            f"Syllabus:\n{syllabus_text}"
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, diagnostic_questions_schema(), "diagnostic_questions", max_tokens=300
            )
            result = DiagnosticQuestions.model_validate_json(raw)
            if result.questions:
                return result.questions[:2]
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
        topic: str,
        diagnostic_qa: List[Dict[str, str]],
    ) -> bool:
        """
        Call LLM to produce a single-session study plan and persist it via backend.

        Generates exactly one session. Additional sessions are appended on demand
        via generate_next_session() as the student progresses.

        Args:
            syllabus_text:  Extracted syllabus content.
            topic:          The specific topic the student chose to focus on.
            diagnostic_qa:  List of {"question": ..., "answer": ...} dicts from
                            the pre-plan diagnostic conversation.

        Returns True on success, False if the LLM call or schema validation fails.
        """
        from memory.schemas import StudyPlan, study_plan_llm_schema

        qa_text = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in diagnostic_qa
        )

        prompt = (
            f"You are a study planner for a K-8 student focusing on \"{topic}\".\n\n"
            "Based on the syllabus and the student's diagnostic responses, "
            "generate ONE introductory session tailored to their current level.\n\n"
            f"Syllabus:\n{syllabus_text}\n\n"
            f"Student diagnostic:\n{qa_text}"
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, study_plan_llm_schema(), "study_plan", max_tokens=1000
            )
        except Exception as e:
            logger.error("generate_study_plan: LLM call failed: %s", e)
            return False

        try:
            plan = StudyPlan.model_validate_json(raw)
        except Exception as e:
            logger.error("generate_study_plan: schema validation failed: %s | raw=%s", e, raw[:300])
            return False

        plan_dict = plan.model_dump()
        plan_dict["source_text"] = syllabus_text  # persist full text for future topic re-use
        self.backend.save_study_plan(plan_dict)
        logger.info("Study plan saved for topic '%s'", topic)
        return True

    def generate_next_session(self) -> Optional[Dict[str, Any]]:
        """
        Generate the next deeper session for an existing plan and append it.

        Builds on all completed session history so the LLM knows exactly what
        was covered and where the student struggled.

        Returns the new session dict on success, None on failure.
        """
        from memory.schemas import SessionPlan, session_plan_llm_schema

        data = self.backend.load()
        plan = data.get("study_plan")
        if not plan:
            logger.error("generate_next_session: no active plan found")
            return None

        topic = plan.get("topic", "the subject")
        assessment_summary = plan.get("assessment_summary", "")
        completed = [s for s in plan.get("sessions", []) if s.get("status") == "completed"]

        history = "\n".join(
            f"Session {i + 1}: covered {', '.join(s.get('topics_covered', []) or s.get('topics', []))}. "
            f"Summary: {s.get('summary', '')}. "
            f"Struggles: {', '.join(s.get('struggles', [])) or 'none'}."
            for i, s in enumerate(completed)
        ) or "No sessions completed yet."

        prompt = (
            f"You are a tutor for a K-8 student studying \"{topic}\".\n\n"
            f"Student assessment: {assessment_summary}\n\n"
            f"Sessions completed so far:\n{history}\n\n"
            "Generate ONE next session that goes deeper into the material, "
            "building on what was already covered and addressing any struggles."
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, session_plan_llm_schema(), "session_plan", max_tokens=800
            )
            session = SessionPlan.model_validate_json(raw)
        except Exception as e:
            logger.error("generate_next_session: failed: %s", e)
            return None

        session_dict = session.model_dump()
        self.backend.append_session(session_dict)
        logger.info("Next session generated for topic '%s'", topic)
        return session_dict

    # ──────────────────────────────────────────────────────────────────────────
    # Session state queries
    # ──────────────────────────────────────────────────────────────────────────

    def get_active_source_text(self) -> str:
        """Return the full syllabus text stored in the active plan, or empty string."""
        data = self.backend.load()
        plan = data.get("study_plan") or {}
        return plan.get("source_text", "")

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
            f"Conversation:\n{transcript}"
        )

        from memory.schemas import SessionDebrief, session_debrief_schema

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, session_debrief_schema(), "session_debrief", max_tokens=400
            )
            debrief = SessionDebrief.model_validate_json(raw)
            summary_fields = {
                "date": datetime.date.today().isoformat(),
                "summary": debrief.summary,
                "struggles": debrief.struggles,
                "next_focus": debrief.next_focus,
                "topics_covered": debrief.topics_covered,
                "performance_notes": debrief.performance_notes,
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

    # ──────────────────────────────────────────────────────────────────────────
    # Intent classification
    # ──────────────────────────────────────────────────────────────────────────

    def classify_redirect_intent(self, user_text: str) -> str:
        """
        Classify the student's response in awaiting_session_redirect.

        Returns one of: "continue", "save_and_break", "new_session".
        Falls back to keyword heuristics if the LLM call fails.
        """
        from memory.schemas import RedirectIntent, redirect_intent_schema

        prompt = (
            "A student is in a tutoring session. They were asked whether they want to "
            "continue the session, save their progress and stop for now, or start something "
            "completely new.\n\n"
            f"Student said: \"{user_text}\"\n\n"
            "Classify their intent:\n"
            "- continue: student wants to keep going with the current session\n"
            "- save_and_break: student wants to stop for now and resume the same topic later\n"
            "- new_session: student wants to stop and start something completely different"
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, redirect_intent_schema(), "redirect_intent", max_tokens=30
            )
            result = RedirectIntent.model_validate_json(raw)
            logger.info("classify_redirect_intent: '%s' → %s", user_text[:50], result.intent)
            return result.intent
        except Exception as e:
            logger.error("classify_redirect_intent failed: %s — falling back to keywords", e)
            text = user_text.lower()
            if any(w in text for w in ("new", "different", "else", "something else", "another")):
                return "new_session"
            if any(w in text for w in ("stop", "break", "later", "pause", "done", "end", "finish", "save", "no", "tired")):
                return "save_and_break"
            return "continue"

    def classify_resume_intent(self, user_text: str, topic: str) -> str:
        """
        Classify the student's response in awaiting_resume_choice.

        Returns one of: "continue_topic", "different_topic", "new_syllabus".
        Falls back to keyword heuristics if the LLM call fails.
        """
        from memory.schemas import ResumeIntent, resume_intent_schema

        prompt = (
            f"A student previously studied \"{topic}\" with a tutor. They were asked whether "
            "they want to continue with that same topic, switch to a different topic from the "
            "same syllabus, or upload a completely new syllabus for a different subject.\n\n"
            f"Student said: \"{user_text}\"\n\n"
            "Classify their intent:\n"
            "- continue_topic: student wants to continue with the same topic\n"
            "- different_topic: student wants a different topic from the same syllabus\n"
            "- new_syllabus: student wants to upload a new syllabus for a completely different subject"
        )

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, resume_intent_schema(), "resume_intent", max_tokens=30
            )
            result = ResumeIntent.model_validate_json(raw)
            logger.info("classify_resume_intent: '%s' → %s", user_text[:50], result.intent)
            return result.intent
        except Exception as e:
            logger.error("classify_resume_intent failed: %s — falling back to keywords", e)
            text = user_text.lower()
            if any(p in text for p in ("new syllabus", "upload", "different syllabus", "new file", "new document", "new book")):
                return "new_syllabus"
            if any(p in text for p in ("different topic", "other topic", "something else", "different subject", "another topic")):
                return "different_topic"
            return "continue_topic"

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
            "Summarise what was covered so far."
        )

        from memory.schemas import SessionProgress, session_progress_schema

        try:
            raw = self.llm_client.generate_structured_response(
                prompt, session_progress_schema(), "session_progress", max_tokens=300
            )
            progress = SessionProgress.model_validate_json(raw)
            partial_fields = {
                "summary": progress.summary,
                "topics_covered": progress.topics_covered,
                "performance_notes": progress.performance_notes,
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

