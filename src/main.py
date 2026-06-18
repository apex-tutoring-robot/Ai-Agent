import os
# Allow the OS to use its default display and QT backend, rather than hardcoding.

import threading
import time
import queue
from queue import Queue
from typing import Optional
from dotenv import load_dotenv
from visuals.faces.face_animator import FaceAnimator
import logging

from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient, ToolCall
from azure_services.tts_client import TextToSpeechClient
from conversation.state_manager import ConversationStateManager
from privacy.privacy_manager import PrivacyManager
from vision.camera import Camera
from guardrails.guardrails_manager import GuardrailsManager
from memory.study_session_manager import StudySessionManager, SESSION_COMPLETE_RE, OFF_TOPIC_RE

load_dotenv(".env")


def _setup_logging():
    """Write INFO+ logs to both the console and a timestamped file in logs/."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return log_path


_setup_logging()
logger = logging.getLogger(__name__)


class JarvisBot:
    """Main orchestrator for Jarvis tutoring robot."""
    
    def __init__(self, face: Optional['FaceAnimator'] = None):
        """Initialize Jarvis with all components."""
        logger.info("Initializing Jarvis...")
        self.face = face
        
        # Initialize shared PyAudio instance
        import pyaudio
        self.pa = pyaudio.PyAudio()
        
        # Initialize components with shared PyAudio
        self.wake_word_detector = WakeWordDetector(pa=self.pa)
        self.audio_player = AudioPlayer(
            pa=self.pa,
            on_level=self.face.push_mouth_level if self.face else None
        )
        self.stt_client = SpeechToTextClient()
        self.llm_client = LLMClient()
        # Preserve the original system prompt so session context can be appended/removed cleanly.
        self.llm_client._base_system_prompt = self.llm_client.system_prompt
        self.tts_client = TextToSpeechClient()
        self.privacy_manager = PrivacyManager()
        self.guardrails = GuardrailsManager(
            config_path=os.getenv('GUARDRAILS_CONFIG_PATH', 'config/guardrails')
        )
        self.camera = Camera()
        self.conversation_manager = ConversationStateManager(
            max_history=int(os.getenv('MAX_CONVERSATION_HISTORY', 30))
        )
        
        self._is_running = False
        self._interaction_lock = threading.Lock()
        self._interruption_event = threading.Event()
        self._speaker_busy = threading.Event() # Track if speaker is active
        self._conversation_active = threading.Event()
        self._request_queue = Queue()

        # ── Display sleep state ──
        self._display_on = True
        self._display_timer: Optional[threading.Timer] = None
        self._display_lock = threading.Lock()

        # ── Echo Guard State ──
        self._bot_is_speaking = False          # True while TTS audio is playing
        self._last_bot_response = ""           # Full text of last bot response
        self._playback_ended_time = 0.0        # time.time() when playback stopped
        self._barge_in_detected = threading.Event()  # Set by VAD when real user speech detected

        # ── Study Session State Machine ──
        os.makedirs("syllabus", exist_ok=True)
        os.makedirs("curriculum", exist_ok=True)
        self.study_session_manager = StudySessionManager(llm_client=self.llm_client)
        # States: "normal" | "awaiting_syllabus" | "extracting_syllabus"
        #         | "identifying_topics" | "govt_syllabus" | "awaiting_govt_topic"
        #         | "awaiting_topic_choice" | "generating_diagnostic" | "asking_diagnostic"
        #         | "generating_plan" | "in_session" | "awaiting_session_redirect"
        #         | "awaiting_resume_choice" | "generating_next_session"
        self._study_state: str = "normal"
        self._pending_syllabus_path: Optional[str] = None
        self._pending_syllabus_text: Optional[str] = None
        self._pending_topic: Optional[str] = None
        self._diagnostic_questions: list = []
        self._diagnostic_answers: list = []
        self._diagnostic_index: int = 0
        self._active_session: Optional[dict] = None
        self._pending_session_finalize: bool = False
        self._pending_partial_save: bool = False
        self._pending_new_session: bool = False
        self._study_state_lock = threading.Lock()

        logger.info("Jarvis initialized successfully!")
    
    def _recreate_audio_system(self):
        """Emergency reset of all audio components when ALSA crashes."""
        try:
            # 1. Stop everything
            if hasattr(self, 'wake_word_detector'):
                self.wake_word_detector.stop()
            if hasattr(self, 'audio_player'):
                self.audio_player.stop_streaming(immediate=True)
            
            # 2. Settle period
            time.sleep(1.0)
            
            # 3. Re-initialize everything to Ensure consistency
            self.pa = pyaudio.PyAudio()
            
            # Re-init components with shared callbacks
            self.continuous_vad = ContinuousVADCapture(pa=self.pa)
            self.audio_player = AudioPlayer(
                pa=self.pa,
                on_audio_played=self.continuous_vad.on_audio_played
            )
            self.wake_word_detector = WakeWordDetector(pa=self.pa)
            
            logger.info("✅ Audio system fully recreated and re-wired")
        except Exception as e:
            logger.error(f"Failed to recreate audio system: {e}")

    # ── Display power management ───────────────────────────────────────────────

    def _set_display_power(self, on: bool) -> None:
        if not self.face:
            return
        os.system(f"vcgencmd display_power {'1' if on else '0'}")
        self._display_on = on
        logger.info(f"🖥️  Display power {'on' if on else 'off'}")

    def _on_display_sleep(self) -> None:
        logger.info("💤 Display sleep timeout — turning off display")
        self._set_display_power(False)

    def _schedule_display_sleep(self) -> None:
        if not self.face:
            return
        timeout = int(os.getenv('DISPLAY_SLEEP_TIMEOUT', 60))
        with self._display_lock:
            if self._display_timer:
                self._display_timer.cancel()
            self._display_timer = threading.Timer(timeout, self._on_display_sleep)
            self._display_timer.daemon = True
            self._display_timer.start()
        logger.info(f"💤 Display will sleep in {timeout}s of inactivity")

    def _cancel_display_sleep(self) -> None:
        with self._display_lock:
            if self._display_timer:
                self._display_timer.cancel()
                self._display_timer = None

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Word-overlap ratio between two strings (0.0 – 1.0)."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a), len(words_b))

    _CAMERA_TOOL = {
        "type": "function",
        "function": {
            "name": "capture_photo",
            "description": (
                "Capture a photo using the robot's camera. Call this when the student "
                "wants you to look at something, scan their homework, check a problem "
                "on paper, or refers to something physical in front of them."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    # Phrases that signal the student wants to exit or pause the current session.
    # Intentionally broad — awaiting_session_redirect uses LLM classification
    # to distinguish "break", "new topic", and "continue" after catching one of these.
    _SESSION_EXIT_PHRASES = (
        # New session / topic change
        "new session", "something new", "new topic", "different topic",
        "start over", "start fresh", "start new", "new subject",
        "something else", "different subject", "new syllabus",
        # Break / pause / stop
        "take a break", "need a break", "want a break",
        "stop the session", "end the session", "finish the session",
        "stop for now", "stop for today", "done for today",
        "done for now", "done for the day",
        "pause the session", "pause for now",
        "come back later", "resume later",
        "i'm done", "i am done",
        "let's stop", "let us stop", "want to stop", "need to stop",
        "that's enough", "that is enough", "enough for today",
    )

    _NO_SYLLABUS_PHRASES = (
        "don't have", "do not have", "don't have a syllabus", "no syllabus",
        "i don't have one", "i do not have one", "no file", "nothing to upload",
        "use a standard", "use a default", "use a preset", "use a predefined",
        "standard syllabus", "default syllabus", "preset syllabus",
        "pick one for me", "choose one for me", "you decide",
        "skip the upload", "skip upload",
    )

    _BEGIN_ONBOARDING_TOOL = {
        "type": "function",
        "function": {
            "name": "begin_onboarding",
            "description": (
                "Start or resume a tutoring session. Call this when the student wants to "
                "study, start a lesson, begin or resume a tutoring session, or continue "
                "learning a subject."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    def _handle_study_input(self, user_text: str) -> Optional[str]:
        """
        Study session state machine. Called at the top of every speaker turn,
        before camera and guardrails checks.

        Returns a string to speak directly (bypasses camera/guardrails/LLM),
        or None to let the normal pipeline handle the turn.

        States:
            normal                  → check for study trigger; resume or start new plan
            awaiting_syllabus       → re-check syllabus dir on every user utterance
            extracting_syllabus     → blocking file extraction (synthetic turn)
            generating_diagnostic   → LLM generates diagnostic questions (synthetic turn)
            asking_diagnostic       → iterating through Q&A to gauge prior knowledge
            generating_plan         → LLM builds personalised plan (synthetic turn)
            in_session              → return None (normal LLM handles everything)
            awaiting_session_redirect → off-topic detected; user decides to continue or end
        """
        with self._study_state_lock:
            state = self._study_state

        # ── in_session: intercept session-exit intent before LLM sees it ─────────
        if state == "in_session":
            text = user_text.lower()
            if any(phrase in text for phrase in self._SESSION_EXIT_PHRASES):
                focus = (self._active_session or {}).get("focus", "our current topic")
                with self._study_state_lock:
                    self._study_state = "awaiting_session_redirect"
                logger.info("📚 Session-exit intent detected in in_session — redirecting")
                return (
                    f"It sounds like you want to step away from our session on {focus}. "
                    f"You have three choices: keep going and stay in this session, "
                    f"take a break and save your spot so we can pick up {focus} later, "
                    f"or stop this session and start something on a completely different topic."
                )
            return None

        # ── awaiting_session_redirect: LLM classifies continue / break / new ──────
        if state == "awaiting_session_redirect":
            intent = self.study_session_manager.classify_redirect_intent(user_text)
            if intent == "new_session":
                self._pending_partial_save = True
                self._pending_new_session = True
                return "Of course! I will save your progress here. Let us get you set up with something new."
            if intent == "save_and_break":
                self._pending_partial_save = True
                return "Sure! I will save your progress. Pick up where we left off whenever you are ready."
            # intent == "continue"
            with self._study_state_lock:
                self._study_state = "in_session"
            return "Okay! Let us get back to it."

        # ── extracting_syllabus: blocking text/image/PDF extraction ──
        if state == "extracting_syllabus":
            syllabus_text = self.study_session_manager.extract_syllabus_text(
                self._pending_syllabus_path
            )
            if syllabus_text.startswith("ERROR:"):
                with self._study_state_lock:
                    self._pending_syllabus_path = None
                    self._study_state = "awaiting_syllabus"
                return syllabus_text.replace("ERROR: ", "")
            with self._study_state_lock:
                self._pending_syllabus_text = syllabus_text
                self._study_state = "identifying_topics"
            self._request_queue.put("__IDENTIFY_TOPICS__")
            return "I have read your syllabus!"

        # ── identifying_topics: LLM finds topics, asks user to choose ──
        if state == "identifying_topics":
            topics = self.study_session_manager.identify_topics(self._pending_syllabus_text)
            if not topics:
                with self._study_state_lock:
                    self._study_state = "normal"
                return (
                    "I had trouble reading the topics from your syllabus. "
                    "Please try uploading it again."
                )
            with self._study_state_lock:
                self._study_state = "awaiting_topic_choice"
            topic_list = ", ".join(topics)
            return (
                f"I can see a few topics in this syllabus: {topic_list}. "
                f"Which one would you like to focus on?"
            )

        # ── govt_syllabus: user has no syllabus — ask what topic they want ──
        if state == "govt_syllabus":
            with self._study_state_lock:
                self._study_state = "awaiting_govt_topic"
            return "What subject or topic would you like to study today?"

        # ── awaiting_govt_topic: load static curriculum doc, then identify topics ──
        if state == "awaiting_govt_topic":
            topic = user_text.strip()
            curriculum_text = self.study_session_manager.load_curriculum_doc(topic)
            if not curriculum_text:
                return (
                    f"I do not have a curriculum document for {topic} yet. "
                    "Could you try a different subject, or ask a parent to upload a syllabus file?"
                )
            with self._study_state_lock:
                self._pending_syllabus_text = curriculum_text
                self._study_state = "identifying_topics"
            self._request_queue.put("__IDENTIFY_TOPICS__")
            return f"Great, let me look up what we cover in {topic}!"

        # ── awaiting_topic_choice: store chosen topic, start diagnostic ──
        if state == "awaiting_topic_choice":
            with self._study_state_lock:
                self._pending_topic = user_text.strip()
                self._study_state = "generating_diagnostic"
            self._request_queue.put("__GENERATE_DIAGNOSTIC__")
            return f"Great, let us focus on {user_text.strip()}! Let me think of a couple of questions to see where you are starting from."

        # ── generating_diagnostic: LLM produces 2 questions scoped to chosen topic ──
        if state == "generating_diagnostic":
            questions = self.study_session_manager.generate_diagnostic_questions(
                self._pending_syllabus_text,
                topic=self._pending_topic or "",
            )
            with self._study_state_lock:
                self._diagnostic_questions = questions
                self._diagnostic_answers = []
                self._diagnostic_index = 0
                self._study_state = "asking_diagnostic"
            return questions[0]

        # ── asking_diagnostic: collect answer, advance to next question or plan ──
        if state == "asking_diagnostic":
            self._diagnostic_answers.append(user_text)
            next_index = self._diagnostic_index + 1
            with self._study_state_lock:
                self._diagnostic_index = next_index
            if next_index < len(self._diagnostic_questions):
                return self._diagnostic_questions[next_index]
            # All questions answered — generate plan
            with self._study_state_lock:
                self._study_state = "generating_plan"
            self._request_queue.put("__GENERATE_PLAN__")
            return "Thanks! Let me put together your first session. One moment."

        # ── generating_plan: blocking LLM call to build the single first session ──
        if state == "generating_plan":
            diagnostic_qa = [
                {"question": q, "answer": a}
                for q, a in zip(self._diagnostic_questions, self._diagnostic_answers)
            ]
            success = self.study_session_manager.generate_study_plan(
                self._pending_syllabus_text,
                self._pending_topic or "",
                diagnostic_qa,
            )
            if not success:
                with self._study_state_lock:
                    self._pending_syllabus_path = None
                    self._pending_syllabus_text = None
                    self._pending_topic = None
                    self._diagnostic_questions = []
                    self._diagnostic_answers = []
                    self._diagnostic_index = 0
                    self._study_state = "normal"
                return "Sorry, I had trouble creating your study plan. Please try again."

            if self._pending_syllabus_path:
                self.study_session_manager.mark_syllabus_processed(self._pending_syllabus_path)
            session = self.study_session_manager.get_next_session()
            self.study_session_manager.mark_session_in_progress(session["session_id"])
            with self._study_state_lock:
                self._pending_syllabus_path = None
                self._pending_syllabus_text = None
                self._pending_topic = None
                self._diagnostic_questions = []
                self._diagnostic_answers = []
                self._diagnostic_index = 0
                self._active_session = session
                self._study_state = "in_session"
                context = self.study_session_manager.build_session_context(session)
                self.llm_client.system_prompt = self.llm_client._base_system_prompt + context
            return (
                f"Your first session is ready! "
                f"We will start with: {session.get('focus', 'your topic')}. "
                f"Ready to begin?"
            )

        # ── generating_next_session: deeper follow-up session generated on demand ──
        if state == "generating_next_session":
            session = self.study_session_manager.generate_next_session()
            if not session:
                with self._study_state_lock:
                    self._study_state = "normal"
                return "Sorry, I had trouble generating the next session. Please try again."
            self.study_session_manager.mark_session_in_progress(session["session_id"])
            with self._study_state_lock:
                self._active_session = session
                self._study_state = "in_session"
                context = self.study_session_manager.build_session_context(session)
                self.llm_client.system_prompt = self.llm_client._base_system_prompt + context
            return (
                f"Your next session is ready! "
                f"We will go deeper into: {session.get('focus', 'your topic')}. "
                f"Ready to begin?"
            )

        # ── awaiting_resume_choice: LLM classifies continue_topic / different_topic / new_syllabus ──
        if state == "awaiting_resume_choice":
            data = self.study_session_manager.backend.load()
            topic = (data.get("study_plan") or {}).get("topic", "your previous topic")
            intent = self.study_session_manager.classify_resume_intent(user_text, topic)

            if intent == "continue_topic":
                with self._study_state_lock:
                    self._study_state = "generating_next_session"
                self._request_queue.put("__GENERATE_NEXT_SESSION__")
                return "Great! Let me put together the next session for you. One moment."

            if intent == "different_topic":
                source_text = self.study_session_manager.get_active_source_text()
                if source_text:
                    with self._study_state_lock:
                        self._pending_syllabus_text = source_text
                        self._study_state = "identifying_topics"
                    self._request_queue.put("__IDENTIFY_TOPICS__")
                    return "Sure! Let me find the topics from your syllabus so you can pick one."
                # Source text not saved — need a new file
                with self._study_state_lock:
                    self._study_state = "awaiting_syllabus"
                return (
                    "I do not have your previous syllabus saved. "
                    "Please upload it again and let me know when it is ready."
                )

            # intent == "new_syllabus"
            with self._study_state_lock:
                self._study_state = "awaiting_syllabus"
            return (
                "Sure! Please upload a new syllabus file and let me know when it is ready. "
                "I can read text files, images, and PDFs."
            )

        # ── awaiting_syllabus: re-check for file on every utterance ──
        if state == "awaiting_syllabus":
            lowered = user_text.lower()
            if any(phrase in lowered for phrase in self._NO_SYLLABUS_PHRASES):
                with self._study_state_lock:
                    self._study_state = "govt_syllabus"
                self._request_queue.put("__GOVT_SYLLABUS__")
                return "No problem! What subject or topic would you like to study today?"
            syllabus_path = self.study_session_manager.get_new_syllabus_file()
            if not syllabus_path:
                return (
                    "I still do not see a syllabus file. "
                    "Please upload it and let me know when it is ready. "
                    "Or if you do not have one, just say so and I can use a standard curriculum."
                )
            return self._process_syllabus_file(syllabus_path)

        # ── normal: study session start handled via begin_onboarding tool call ──
        return None

    def _process_syllabus_file(self, file_path: str) -> str:
        """Queue syllabus extraction as a synthetic turn and return an interim response."""
        with self._study_state_lock:
            self._pending_syllabus_path = file_path
            self._study_state = "extracting_syllabus"
        self._request_queue.put("__EXTRACT_SYLLABUS__")
        return "I found a syllabus file! Let me read through it, one moment."

    def _execute_begin_onboarding(self) -> str:
        """
        Execute the begin_onboarding tool call.

        Resumes the active session, generates the next deeper session, or starts
        fresh onboarding — same routing logic as the old _is_study_trigger path.
        Returns the spoken response directly; no second LLM call needed.
        """
        session = self.study_session_manager.get_next_session()
        if session:
            self.study_session_manager.mark_session_in_progress(session["session_id"])
            with self._study_state_lock:
                self._active_session = session
                self._study_state = "in_session"
                context = self.study_session_manager.build_session_context(session)
                self.llm_client.system_prompt = self.llm_client._base_system_prompt + context
            return (
                f"Welcome back! Continuing with: "
                f"{session.get('focus', 'your topic')}. Ready to begin?"
            )

        if self.study_session_manager.backend.has_any_plan():
            data = self.study_session_manager.backend.load()
            topic = (data.get("study_plan") or {}).get("topic", "your previous topic")
            with self._study_state_lock:
                self._study_state = "awaiting_resume_choice"
            return (
                f"Welcome back! Last time we were working on {topic}. "
                f"Would you like to continue with that, switch to a different topic, "
                f"or upload a new syllabus for something completely different?"
            )

        syllabus_path = self.study_session_manager.get_new_syllabus_file()
        if not syllabus_path:
            with self._study_state_lock:
                self._study_state = "awaiting_syllabus"
            return (
                "I would love to help you study! I do not have a syllabus yet. "
                "Please upload a syllabus file "
                "and let me know when it is ready. "
                "I can read text files, images, and PDFs."
            )

        return self._process_syllabus_file(syllabus_path)

    def _listener_loop(self, continuous_vad, request_queue):
        """Producer: Always listening, detecting interruptions in real-time.
        
        Three-layer echo defense:
          L1 – Discard ALL recognitions while bot is speaking (unless barge-in detected)
          L2 – Text similarity filter rejects echo transcriptions that match bot's last response  
          L3 – Echo cooldown after playback ends (500ms grace period)
        """
        logger.info("📡 Listener worker started")
        ECHO_COOLDOWN_S = 0.5          # Ignore speech for 500ms after playback ends
        SIMILARITY_THRESHOLD = 0.55    # Reject if >55% word overlap with bot's last response
        SIMILARITY_WINDOW_S = 3.0      # Only apply similarity filter within 3s of playback ending
        
        try:
            # Create a never-ending generator for STT
            audio_gen = continuous_vad.always_streaming()
            
            # Start continuous recognition
            for text, is_final, timestamp in self.stt_client.recognize_streaming(audio_gen):
                if not text.strip():
                    continue
                
                now = time.time()
                
                # ── LAYER 1: Playback gate ──
                # While the bot is speaking, discard EVERYTHING unless the VAD
                # layer has confirmed a genuine barge-in (energy above threshold).
                if self._bot_is_speaking:
                    if not self._barge_in_detected.is_set():
                        # Echo — silently discard
                        continue
                    else:
                        # Barge-in confirmed by VAD energy gate!
                        # Stop playback immediately and let this through
                        if self._speaker_busy.is_set() or self.audio_player._is_playing:
                            logger.info(f"🎤 BARGE-IN: '{text}' (Stopping playback)")
                            self._interruption_event.set()
                            self.audio_player.stop_streaming(immediate=True)
                
                # ── LAYER 3: Echo cooldown ──
                # Short grace period after playback ends to catch echo tail
                if self._playback_ended_time > 0 and (now - self._playback_ended_time) < ECHO_COOLDOWN_S:
                    logger.debug(f"🔇 Echo cooldown — discarding: '{text[:40]}...'")
                    continue
                
                # ── LAYER 2: Text similarity filter ──
                # Reject finalized text that matches the bot's last response,
                # but only within a short window after playback ends (real echo
                # arrives immediately; user follow-up speech can share words too).
                within_window = (self._playback_ended_time > 0 and
                                 (now - self._playback_ended_time) < SIMILARITY_WINDOW_S)
                if is_final and self._last_bot_response and within_window:
                    sim = self._text_similarity(text, self._last_bot_response)
                    if sim > SIMILARITY_THRESHOLD:
                        logger.info(f"🔇 Echo rejected (similarity={sim:.0%}): '{text[:60]}...'")
                        continue
                
                # ── FINISHED UTTERANCE - Queue for processing ──
                if is_final:
                    logger.info(f"📌 User finalized: '{text}'")
                    was_barge_in = self._barge_in_detected.is_set()
                    self._interruption_event.clear()
                    self._barge_in_detected.clear()
                    request_queue.put((text, was_barge_in))
                    
            logger.info("📡 Listener worker stopped cleanly")
        except Exception as e:
            logger.error(f"Listener error: {e}")
            self._conversation_active.clear()

    def _speaker_loop(self, continuous_vad, output_device_index):
        """Consumer: Processes requests and speaks."""
        logger.info("🔊 Speaker worker started")
        try:
            while self._conversation_active.is_set():
                # Wait for next user request
                try:
                    item = self._request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is None:  # Termination sentinel
                    break

                user_text, was_barge_in = item if isinstance(item, tuple) else (item, False)

                # Reset idle timer immediately so long camera/LLM processing doesn't trigger timeout
                continuous_vad.reset_idle_timer()

                # Signal we are busy
                self._speaker_busy.set()
                self._interruption_event.clear()
                self._barge_in_detected.clear()
                
                try:
                    logger.info(f"📝 Processing Turn: {user_text}")

                    # ── Study Session State Machine ────────────────────────────
                    # Intercepts before camera/guardrails/LLM. Returns a direct
                    # response string, or None to let the normal pipeline run.
                    study_response = self._handle_study_input(user_text)
                    if study_response is not None:
                        final_text = study_response
                        self.conversation_manager.add_assistant_message(final_text)
                        self._last_bot_response = final_text
                        continuous_vad.reset_idle_timer()
                    else:
                        anonymized_text = self.privacy_manager.anonymize(user_text)
                        allowed, refusal_msg = self.guardrails.check_input(anonymized_text)
                        continuous_vad.reset_idle_timer()

                        if allowed:
                            self.conversation_manager.add_user_message(anonymized_text)
                            messages = self.conversation_manager.get_messages()

                            if self.face:
                                self.face.start_thinking()
                            response_chunks = []
                            tool_fired: Optional[ToolCall] = None

                            # ── Tool-aware stream — LLM decides which tool to call ──
                            _tools = [self._CAMERA_TOOL, self._BEGIN_ONBOARDING_TOOL]
                            for item in self.llm_client.generate_response_stream_with_tools(
                                messages, _tools
                            ):
                                if isinstance(item, ToolCall):
                                    tool_fired = item
                                    logger.info("🔧 Tool call: %s (id=%s)", tool_fired.name, tool_fired.call_id)
                                    break
                                if self._interruption_event.is_set():
                                    break
                                response_chunks.append(item)

                            if not tool_fired:
                                logger.info("🔧 No tool called — LLM responded with text")

                            if tool_fired and tool_fired.name == "begin_onboarding":
                                logger.info("📚 Executing begin_onboarding (study_state=%s)", self._study_state)
                                response_chunks = [self._execute_begin_onboarding()]
                                logger.info("📚 begin_onboarding complete → study_state=%s", self._study_state)

                            # ── Execute camera tool if requested ──────────────────
                            elif tool_fired and tool_fired.name == "capture_photo":
                                logger.info("📷 Executing capture_photo")
                                if self.face:
                                    self.face.start_scanning()
                                try:
                                    saved_path = self.camera.capture_and_save()
                                    logger.info("📷 Image saved to %s", saved_path)
                                    import base64 as _b64
                                    with open(saved_path, "rb") as _f:
                                        data_url = f"data:image/jpeg;base64,{_b64.b64encode(_f.read()).decode()}"
                                    extracted = self.llm_client.extract_image_content(data_url)
                                    continuous_vad.reset_idle_timer()
                                    logger.info("📷 Image extracted (%d chars) — sending to LLM", len(extracted))
                                    tool_messages = messages + [
                                        {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [{
                                                "id": tool_fired.call_id,
                                                "type": "function",
                                                "function": {"name": "capture_photo", "arguments": "{}"},
                                            }],
                                        },
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_fired.call_id,
                                            "content": extracted,
                                        },
                                    ]
                                    if self.face:
                                        self.face.start_thinking()
                                    for chunk in self.llm_client.generate_response_stream(tool_messages):
                                        if self._interruption_event.is_set():
                                            break
                                        response_chunks.append(chunk)
                                    logger.info("📷 capture_photo complete — LLM follow-up streamed")
                                except Exception as cam_err:
                                    logger.error("📷 Camera tool failed: %s", cam_err)
                                    response_chunks = ["Sorry, I had trouble with the camera. Please try again."]

                            if self._interruption_event.is_set():
                                partial = "".join(response_chunks)
                                if partial:
                                    self._last_bot_response = partial
                                continuous_vad.reset_idle_timer()
                                continue

                            full_response = "".join(response_chunks)

                            # ── Session complete detection ─────────────────────
                            # Strip sentinels before guardrails so they never reach TTS.
                            if SESSION_COMPLETE_RE.search(full_response):
                                full_response = SESSION_COMPLETE_RE.sub("", full_response).strip()
                                self._pending_session_finalize = True
                                logger.info("📚 Session complete tag detected")

                            if OFF_TOPIC_RE.search(full_response):
                                full_response = OFF_TOPIC_RE.sub("", full_response).strip()
                                focus = (self._active_session or {}).get("focus", "our current topic")
                                with self._study_state_lock:
                                    self._study_state = "awaiting_session_redirect"
                                full_response = (
                                    f"That seems to be outside our session on {focus}. "
                                    f"You have three choices: keep going and stay in this session, "
                                    f"take a break and save your spot so we can pick up {focus} later, "
                                    f"or stop this session and start something on a completely different topic."
                                )
                                logger.info("📚 Off-topic tag detected — asking redirect question")

                            if not full_response:
                                continuous_vad.reset_idle_timer()
                                continue

                            # ── Guardrails: output check ───────────────────────
                            safe, final_text = self.guardrails.check_output(full_response)
                            if safe:
                                self.conversation_manager.add_assistant_message(final_text)
                                self._last_bot_response = final_text
                                logger.info(f"Bot: {final_text}")
                            else:
                                logger.warning("🛡️ Guardrails blocked LLM output — substituting refusal")
                        else:
                            logger.info("🛡️ Guardrails blocked input — speaking refusal")
                            final_text = refusal_msg

                    # ── Shared playback (study-machine and normal paths both land here) ──
                    self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
                    if self.face:
                        self.face.start_talking()
                    self.audio_player.start_streaming(output_device_index=output_device_index)

                    self._bot_is_speaking = True
                    continuous_vad.set_playback_state(True)

                    tts_stream = self.tts_client.synthesize_stream(iter([final_text]))

                    for audio_chunk in tts_stream:
                        if self._interruption_event.is_set():
                            logger.warning("🛑 Speaker aborted due to interruption event")
                            break
                        if self.audio_player._is_playing:
                            try:
                                self.audio_player.queue_audio(audio_chunk)
                            except RuntimeError as e:
                                logger.warning(f"⚠️ Playback queueing failed (likely stopped): {e}")
                                break
                        else:
                            break

                    if not self._interruption_event.is_set():
                        self.audio_player.stop_streaming(immediate=False)

                    continuous_vad.reset_idle_timer()

                except Exception as turn_err:
                    logger.error(f"Error in speaker turn: {turn_err}")
                finally:
                    if self.face:
                        self.face.start_idle()
                    # ── Echo guard: signal playback ended ──
                    self._bot_is_speaking = False
                    self._playback_ended_time = time.time()
                    continuous_vad.set_playback_state(False)
                    self._barge_in_detected.clear()
                    self._speaker_busy.clear()
                    self.audio_player.stop_streaming(immediate=True)

                    # ── Session finalization (background thread, no latency hit) ──
                    if self._pending_session_finalize:
                        self._pending_session_finalize = False
                        session_snapshot = self._active_session
                        messages_snapshot = self.conversation_manager.get_messages()
                        with self._study_state_lock:
                            self._active_session = None
                            self._study_state = "normal"
                            self.llm_client.system_prompt = self.llm_client._base_system_prompt
                        threading.Thread(
                            target=self.study_session_manager.complete_session,
                            args=(session_snapshot, messages_snapshot),
                            daemon=True,
                            name="SessionFinalizeThread"
                        ).start()
                        logger.info("📚 Session finalization dispatched to background thread")

                    # ── Partial save on break (session stays in_progress for later resume) ──
                    if self._pending_partial_save:
                        self._pending_partial_save = False
                        start_new = self._pending_new_session
                        self._pending_new_session = False
                        session_snapshot = self._active_session
                        messages_snapshot = self.conversation_manager.get_messages()
                        if start_new:
                            # Priority: new unprocessed file > saved source_text > ask for upload
                            syllabus_path = self.study_session_manager.get_new_syllabus_file()
                            source_text = (
                                None if syllabus_path
                                else self.study_session_manager.get_active_source_text()
                            )
                            with self._study_state_lock:
                                self._active_session = None
                                self.llm_client.system_prompt = self.llm_client._base_system_prompt
                                if syllabus_path:
                                    self._pending_syllabus_path = syllabus_path
                                    self._study_state = "extracting_syllabus"
                                elif source_text:
                                    self._pending_syllabus_text = source_text
                                    self._study_state = "identifying_topics"
                                else:
                                    self._study_state = "awaiting_syllabus"
                            if syllabus_path:
                                self._request_queue.put("__EXTRACT_SYLLABUS__")
                            elif source_text:
                                self._request_queue.put("__IDENTIFY_TOPICS__")
                            logger.info("📚 New session onboarding triggered after partial save")
                        else:
                            with self._study_state_lock:
                                self._active_session = None
                                self._study_state = "normal"
                                self.llm_client.system_prompt = self.llm_client._base_system_prompt
                        threading.Thread(
                            target=self.study_session_manager.save_session_progress,
                            args=(session_snapshot, messages_snapshot),
                            daemon=True,
                            name="SessionPartialSaveThread"
                        ).start()
                        logger.info("📚 Partial session save dispatched to background thread")
                    
            logger.info("🔊 Speaker worker stopped cleanly")
        except Exception as e:
            logger.error(f"Speaker loop error: {e}")

    def _get_pulse_device_index(self, pa) -> int:
        """Find the PipeWire/PulseAudio ALSA device index (pulse) for audio I/O."""
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = info.get('name', '').lower() if info else ''
                if 'pulse' in name or 'pipewire' in name:
                    logger.info(f"✅ Found PipeWire/PulseAudio device at index {i}: {info['name']}")
                    return i
        except Exception as e:
            logger.warning(f"Error searching for PipeWire/PulseAudio device: {e}")

        logger.warning(
            "⚠️  PipeWire/PulseAudio ALSA device not found — falling back to hw index 1. "
            "Install libasound2-plugins and restart PipeWire to enable AEC."
        )
        return 1

    def _get_input_device_index(self, pa, pulse_index: int) -> int:
        """Return the PyAudio input device index for the VAD mic stream.

        When PipeWire AEC is active (pulse device found), input goes through
        pulse so PipeWire routes it to echo-cancel-source. Falls back to
        AUDIO_INPUT_DEVICE_INDEX env var if pulse is unavailable.
        """
        if pulse_index is not None:
            logger.info(f"🎤 Input routed through PipeWire (device {pulse_index}) — AEC active")
            return pulse_index
        env_idx = os.getenv('AUDIO_INPUT_DEVICE_INDEX')
        if env_idx not in (None, ""):
            idx = int(env_idx)
            logger.info(f"🎤 Using AUDIO_INPUT_DEVICE_INDEX={idx} (no PipeWire, direct hw)")
            return idx
        logger.warning("🎤 No input device configured — using system default")
        return None

    def _handle_wake_word(self):
        """Handle wake word detection - enter continuous conversation mode."""
        # Cancel display sleep timer and restore display before anything else
        self._cancel_display_sleep()
        if not self._display_on:
            self._set_display_power(True)

        # Prevent concurrent interactions
        if not self._interaction_lock.acquire(blocking=False):
            logger.warning("Already in conversation, ignoring wake word")
            return
        
        try:
            logger.info("\n" + "="*60)
            logger.info("🎤 WAKE WORD DETECTED - CONVERSATION MODE ACTIVATED")
            logger.info("="*60)
            logger.info(f"⏱️  Will end after 10 seconds of silence")
            
            # Stop wake word detection to free microphone
            self.wake_word_detector.stop()

            # Flush any stale items (e.g. a leftover None sentinel from the previous
            # conversation's idle-timeout cleanup) so the new speaker thread starts clean.
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except queue.Empty:
                    break

            # Enter continuous conversation mode
            idle_timeout = int(os.getenv('CONVERSATION_IDLE_TIMEOUT_SECONDS', 10))
            
            # Output goes to pulse (PipeWire routes to echo-cancel-sink → USB speaker).
            # Input goes to pulse too (PipeWire routes from echo-cancel-source → AEC-processed ReSpeaker).
            pulse_index = self._get_pulse_device_index(self.pa)
            input_index = self._get_input_device_index(self.pa, pulse_index)

            # Initialize VAD with dedicated INPUT stream
            continuous_vad = ContinuousVADCapture(
                idle_timeout_seconds=idle_timeout,
                pa=self.pa,
                input_device_index=input_index,
                player=self.audio_player
            )
            
            # Wire callback so VAD knows when bot is speaking (for AEC/Duck)
            self.audio_player.on_audio_played = continuous_vad.on_audio_played
            
            # Wire barge-in event so VAD can signal listener when user interrupts
            continuous_vad.set_barge_in_event(self._barge_in_detected)
            
            # Note: We NO LONGER share streams. 
            # VAD has its own Input stream. Player has its own Output stream.
            # PulseAudio handles the mixing.
                
            # ENTER FULL DUPLEX MODE
            self._conversation_active.set()
            
            # Start Worker Threads
            listener_thread = threading.Thread(
                target=self._listener_loop, 
                args=(continuous_vad, self._request_queue),
                name="ListenerThread"
            )
            speaker_thread = threading.Thread(
                target=self._speaker_loop, 
                args=(continuous_vad, pulse_index),
                name="SpeakerThread"
            )
            
            listener_thread.start()
            speaker_thread.start()
            
            logger.info("🚀 Full-Duplex engines started")
            
            # Wait for conversation to end (timeout or manual stop)
            while self._conversation_active.is_set():
                # Check for fatal errors in audio components and recover
                if self.audio_player.has_fatal_error:
                    logger.warning("♻️  FATAL AUDIO ERROR - Recreating system...")
                    self._recreate_audio_system()
                
                # Check if idle timeout exceeded — but ONLY when the speaker is
                # idle. Long turns (camera + image extraction + LLM) can exceed
                # idle_timeout during processing, which would tear down the
                # conversation mid-response and leave threads in a bad state.
                if not self._speaker_busy.is_set():
                    idle_duration = time.time() - continuous_vad.last_speech_time
                    if idle_duration >= idle_timeout:
                        logger.info(f"⏱️  {idle_timeout}s idle timeout - ending")
                        self._conversation_active.clear()
                        break
                    
                time.sleep(0.5)
            
            logger.info(f"\n👋 Conversation ended")

        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        
        finally:
            # 1. SIGNAL WORKERS TO STOP
            self._conversation_active.clear()
            self._request_queue.put(None)  # Sentinel for speaker

            # Reset study state if conversation ended mid-flow (e.g. idle timeout
            # while waiting for the student to answer syllabus questions).
            # "in_session" is intentionally NOT reset — the session persists across
            # wake-word cycles until [SESSION_COMPLETE] is detected.
            with self._study_state_lock:
                if self._study_state in (
                    "awaiting_syllabus",
                    "extracting_syllabus",
                    "identifying_topics",
                    "govt_syllabus",
                    "awaiting_govt_topic",
                    "awaiting_topic_choice",
                    "generating_diagnostic",
                    "asking_diagnostic",
                    "generating_plan",
                    "awaiting_resume_choice",
                    "generating_next_session",
                ):
                    logger.info("Conversation ended mid-study-flow — resetting study state to normal")
                    self._pending_syllabus_path = None
                    self._pending_syllabus_text = None
                    self._pending_topic = None
                    self._diagnostic_questions = []
                    self._diagnostic_answers = []
                    self._diagnostic_index = 0
                    self._pending_new_session = False
                    self._study_state = "normal"
            
            # 2. Stop VAD first — this closes the audio stream and unblocks the listener thread
            try:
                if 'continuous_vad' in locals():
                    continuous_vad.stop_background_monitoring()
                    continuous_vad.cleanup()
            except:
                pass

            # 3. Now join workers — listener can exit cleanly within the timeout
            try:
                if 'listener_thread' in locals() and listener_thread.is_alive():
                    listener_thread.join(timeout=5.0)
                if 'speaker_thread' in locals() and speaker_thread.is_alive():
                    speaker_thread.join(timeout=1.0)
            except:
                pass
            
            # 4. Always restart wake word detection
            logger.info("▶️  Resuming wake word detection...")
            self._restart_wake_word()
            self._interaction_lock.release()
    
    def _process_turn_streaming(self, continuous_vad: ContinuousVADCapture, output_device_index: int = None) -> bool:
        """ Process one turn of the conversation using streaming STT.
        Args:
            continuous_vad: Continuous VAD instance
            output_device_index: PulseAudio output index for playback
            
        Returns:
            True if speech was processed, False otherwise
        """
        try:
            # Step 1: Stream audio chunks to STT
            if self.face:
                self.face.start_thinking()
            logger.info("☁️  Starting streaming speech recognition...")
            stt_start = time.perf_counter()
            
            # Get streaming audio chunks from VAD
            audio_stream = continuous_vad.stream_audio_chunks()
            
            # Stream to Azure STT
            user_text = ""
            first_text_time = None
            for result_tuple in self.stt_client.recognize_streaming(audio_stream):
                # Unpack tuple: (text, first_recognition_time)
                text_result, first_text_time = result_tuple
                user_text = text_result  # Get the complete text
            
            # Check if we got any speech after all recognition events
            if not user_text.strip():
                    logger.warning("No speech recognized")
                    return False
            
            # LATENCY METRICS
            # STT Latency: Time from when user stopped speaking to when the FINAL
            # text was recognized by Azure.
            if continuous_vad.silence_detected_time:
                # Use current time as 'final' text arrival time
                final_text_time = time.perf_counter()
                stt_latency = final_text_time - continuous_vad.silence_detected_time
                logger.info(f"⏱️  STT Latency (Full Turn): {stt_latency:.3f}s")
                
                # Also log how fast the FIRST words were detected
                if first_text_time:
                    # Note: this might be negative if first words arriving before 
                    # VAD's 800ms silence period ends!
                    raw_stt_speed = first_text_time - continuous_vad.silence_detected_time
                    logger.info(f"⏱️  First-Text Offset: {raw_stt_speed:.3f}s")
            
            # 2. End-to-TTFT: Will be calculated when first LLM token arrives

            logger.info(f"📝 Student: {user_text}")
            
            # Step 2: Anonymize PII
            anonymized_text = self.privacy_manager.anonymize(user_text)
            # Step 3: Add to conversation history
            self.conversation_manager.add_user_message(anonymized_text)
            # Step 4: Generate LLM response with streaming
            logger.info("🧠 Generating response...")
            llm_start = time.perf_counter()
            messages = self.conversation_manager.get_messages()
            
            # Collect response text chunks as they stream
            response_chunks = []
            first_token = True
            try:
                # Start audio player BEFORE first audio arrives for lower latency
                # CRITICAL: Use dedicated PulseAudio output stream
                self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
                if self.face:
                    self.face.start_talking()
                self.audio_player.start_streaming(output_device_index=output_device_index)
                
                # Full Duplex: Start monitoring for interruptions while bot speaks
                self._interruption_event.clear()
                def on_interruption():
                    # Stop monitoring IMMEDIATELY 
                    continuous_vad.stop_background_monitoring()
                    self._interruption_event.set()
                    self.audio_player.stop_streaming(immediate=True)
                
                # Grace period: Wait 500ms before starting monitoring to avoid 
                # catching the user's trailing breath or room echo.
                time.sleep(0.5)
                continuous_vad.start_background_monitoring(on_interruption)
                
                # Create a wrapper that collects chunks while streaming
                def text_chunk_collector(llm_stream):
                    """Collect text chunks while passing them through."""
                    nonlocal first_token
                    first_token_time = None
                    for chunk in llm_stream:
                        response_chunks.append(chunk)
                        if first_token:
                            first_token_time = time.perf_counter()
                            llm_latency = first_token_time - llm_start
                            
                            # LATENCY: End-to-TTFT (Silence detected → First LLM token)
                            if continuous_vad.silence_detected_time:
                                end_to_ttft = first_token_time - continuous_vad.silence_detected_time
                                logger.info(f"⏱️  End-to-TTFT: {end_to_ttft:.3f}s")
                            
                            logger.info(f"⏱️  LLM TTFT: {llm_latency:.3f}s")
                            
                            # Track for TTFAS calculation
                            self.audio_player.first_token_time = first_token_time
                            first_token = False
                        yield chunk
                
                # Stream LLM output through collector to TTS
                llm_stream = self.llm_client.generate_response_stream(messages)
                collected_stream = text_chunk_collector(llm_stream)
                tts_stream = self.tts_client.synthesize_stream(collected_stream)
                
                # Stream audio to player (player already started, audio plays immediately)
                for audio_chunk in tts_stream:
                    # Check for interruption or if player died
                    if self._interruption_event.is_set():
                        logger.warning("🛑 INTERRUPTION DETECTED during TTS stream - stopping...")
                        self.audio_player.stop_streaming(immediate=True)
                        break
                    
                    if not self.audio_player._is_playing:
                        logger.warning("⚠️  Playback stopped unexpectedly - ending turn")
                        break

                    try:
                        self.audio_player.queue_audio(audio_chunk)
                    except Exception as queue_err:
                        logger.error(f"Failed to queue audio: {queue_err}")
                        break
                
                # Stop monitoring immediately when we exit the stream loop
                continuous_vad.stop_background_monitoring()

                # Wait for playback to complete (unless interrupted)
                if not self._interruption_event.is_set() and self.audio_player._is_playing:
                    self.audio_player.stop_streaming()
                
                # If we were interrupted, we return True so Turn Count advances 
                # and next turn starts immediately with the handoff audio.
                if self._interruption_event.is_set():
                    logger.info("🛑 Response interrupted - advancing to next turn")
                    # Add partial response to history? (Optional, skipping for brevity)
                    return True 
                
                # Combine collected chunks into full response
                response_text = ''.join(response_chunks)
                logger.info(f"Chippy: {response_text}")
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)
                
                # Reset idle timer AFTER bot finishes speaking
                # This ensures we don't timeout while bot is generating/speaking
                continuous_vad.reset_idle_timer()
                
                if self.face:
                    self.face.start_idle()
                return True
                
            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
                self.audio_player.stop_streaming()
                raise
            
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return False
        
    def _restart_wake_word(self):
        """Restart wake word detection in a non-blocking way."""
        if not self._is_running:
            return
        
        # Make sure previous detector is fully stopped
        if self.wake_word_detector.is_running():
            logger.info("Waiting for previous wake word detector to stop...")
            self.wake_word_detector.stop()
            time.sleep(0.5)  # Give it time to clean up
        
        # Start wake word detector in a new thread
        logger.info("Starting new wake word detection thread...")
        wake_thread = threading.Thread(
            target=self.wake_word_detector.start,
            args=(self._handle_wake_word,),
            daemon=True
        )
        wake_thread.start()
        self._schedule_display_sleep()
    
    def run(self):
        """Start Jarvis and run the main loop."""
        if self.face:
            self.face.start_idle()
        logger.info("\n" + "="*20)
        logger.info("Jarvis TUTORING ROBOT STARTED")
        logger.info("="*20 + "\n")
        logger.info("Listening for wake word: 'Hey Jarvis'")
        logger.info("Press Ctrl+C to stop\n")
        
        self._is_running = True
        self._schedule_display_sleep()

        try:
            # First blocking wake word listen. When a wake word fires, _handle_wake_word()
            # is called inline, stops the detector, runs the conversation, then calls
            # _restart_wake_word() which takes over in a daemon thread. After that,
            # start() returns here — but we must NOT call stop() at that point because
            # the daemon thread and all shared resources (self.pa, TTS) are still needed.
            self.wake_word_detector.start(self._handle_wake_word)

            # Keep run() alive while wake-word/conversation cycles continue in daemon threads.
            # Only exit when _is_running is cleared by stop() or a signal arrives.
            while self._is_running:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")

        except Exception as e:
            logger.error(f"Error in main loop: {e}")

        finally:
            self.stop()
    
    def stop(self):
        """Stop Jarvis and cleanup resources."""
        logger.info("Shutting down Jarvis...")

        self._is_running = False
        self._cancel_display_sleep()
        self._set_display_power(True)  # Restore display on shutdown

        # Stop wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        
        # Cleanup audio player
        if self.audio_player:
            self.audio_player.cleanup()
        
        # Cleanup TTS client
        if self.tts_client:
            self.tts_client.cleanup()
        
        # Show conversation summary
        logger.info(f"\nFinal conversation state: {self.conversation_manager}")
        
        # Cleanup and terminate shared PyAudio
        if self.pa:
            try:
                self.pa.terminate()
                logger.info("Shared PyAudio terminated")
            except:
                pass
            self.pa = None
            
        logger.info("Jarvis shutdown complete. Goodbye! 👋\n")

def main():
    import os

    # ── Display configuration ──────────────────────────────────────────────────
    # FACE_ENABLED=true          → show face animation
    # FACE_ENABLED=false         → headless, no GUI (default when DISPLAY not set)
    # ──────────────────────────────────────────────────────────────────────────
    face_env = os.getenv("FACE_ENABLED", "").strip().lower()
    if face_env in ("true", "1", "yes"):
        face_enabled = True
    elif face_env in ("false", "0", "no"):
        face_enabled = False
    else:
        # Auto: enable face only when DISPLAY is already set in the environment
        face_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if face_enabled:
        # Use whatever DISPLAY is set in the environment (e.g. by XWayland).
        # Fall back to :0 if nothing is set.
        if not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = ":0"

        # Ensure X authentication is available.
        if not os.environ.get("XAUTHORITY"):
            xauth_path = os.path.expanduser("~/.Xauthority")
            if os.path.exists(xauth_path):
                os.environ["XAUTHORITY"] = xauth_path

        # Pre-check: verify the X11 socket actually exists before letting Qt try.
        # Qt calls abort() on a missing display — that can't be caught by Python.
        display = os.environ.get("DISPLAY", "")
        display_num = display.lstrip(":").split(".")[0]
        socket_path = f"/tmp/.X11-unix/X{display_num}"
        if not os.path.exists(socket_path):
            logger.warning(
                f"X11 socket {socket_path} not found — is the display server running? "
                f"Falling back to headless."
            )
            face_enabled = False
        else:
            logger.info(f"Face animation enabled on display {display}")

    if not face_enabled:
        logger.info("Face animation disabled — running headless (set FACE_ENABLED=true to enable)")

    face = None
    if face_enabled:
        try:
            face = FaceAnimator("src/visuals/faces")
        except Exception as e:
            logger.warning(f"Failed to init face animation: {e}. Falling back to headless.")

    jarvis = JarvisBot(face)

    if face:
        worker = threading.Thread(target=jarvis.run, daemon=True)
        worker.start()
        # cv2 GUI event loop must run on the main thread
        face.render_forever()
    else:
        jarvis.run()

if __name__ == "__main__":
    main()
