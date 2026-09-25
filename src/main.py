import os
# Allow the OS to use its default display and QT backend, rather than hardcoding.

import re
import subprocess
import threading
import time
import queue
from queue import Queue
from typing import Dict, Optional
from difflib import SequenceMatcher
from dotenv import load_dotenv
import numpy as np

# audio.wake_word (openwakeword -> onnxruntime) MUST be imported before any
# PyQt5 import (visuals.ui.*, below) - onnxruntime and PyQt5 each bundle
# their own native runtime DLLs, and loading PyQt5 first causes a hard
# segfault on Windows when onnxruntime is imported afterward. Importing
# onnxruntime's DLLs into the process first avoids the conflict. Verified:
# swapping this order reliably segfaults; this order doesn't.
from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from azure_services.interfaces import SpeechToTextProvider, LLMProvider, TextToSpeechProvider
from privacy.privacy_manager import PrivacyManager
from profiles.profile_manager import ProfileManager
from profiles.camera_capture import capture_avatar_photo
from session import Session
from knowledge.textbook_search import TextbookSearch, normalize_grade
from guardrails.guardrails_manager import GuardrailsManager
from guardrails.fast_content_filter import FastContentFilter
from identity.identity_provider import IdentityObservation
from identity.identity_resolver import IdentityResolver
from identity.name_identity_provider import NameIdentityProvider
from tutor.decision_engine import TutorAction, TutorDecisionEngine
from tutor.review_scheduler import ReviewScheduler
from tutor.question_engine import ProactiveQuestionEngine
from curriculum.graph import CurriculumGraph
from affect.interaction_signals import InteractionSignalEngine
from expression.controller import ExpressionController

from visuals.ui.ui_signals import UISignals
from visuals.ui.main_window import MainWindow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI function-calling schema for the one real tool Jarvis currently
# exposes to the LLM - see JarvisBot._tool_search_curriculum(). The model
# decides for itself whether a question needs this, rather than a
# keyword-routed heuristic like is_math_query().
CURRICULUM_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_curriculum",
        "description": (
            "Search the official California Common Core math curriculum "
            "standards for this student's grade level. Use this to ground "
            "your explanation in what's actually expected at their grade, "
            "or to check whether a topic is grade-appropriate before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'fractions with unlike denominators' or 'area of shapes'"
                }
            },
            "required": ["query"]
        }
    }
}

# System-level tool: lets the LLM actually act on requests like "please talk
# more loudly" or "can you be quieter" instead of just replying in words.
# `level` is relative (louder/quieter/normal) rather than a raw number since
# that's how kids actually phrase the request - see JarvisBot._tool_set_volume().
VOLUME_CONTROL_TOOL = {
    "type": "function",
    "function": {
        "name": "set_volume",
        "description": (
            "Actually changes how loud Jarvis's speaking voice is - this is "
            "a real hardware volume change, not a suggestion. Call this "
            "whenever the student asks you to speak louder, quieter, or "
            "back to normal volume. The call always succeeds immediately, "
            "so afterward tell the student in a natural, upbeat way that "
            "you've done it (e.g. 'Sure, I'll talk louder!'). Never claim "
            "you can't change your volume - you just did. Only call this "
            "once per request; don't call it again on a later, unrelated "
            "turn unless the student asks again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["louder", "quieter", "normal"],
                    "description": "Direction to adjust the speaking volume."
                }
            },
            "required": ["level"]
        }
    }
}

class JarvisBot:
    """Main orchestrator for Jarvis tutoring robot."""

    # Asked in order right after a brand new profile is created - this is
    # Jarvis's only chance to learn who a student actually is before
    # teaching them anything (see _handle_onboarding_answer for where each
    # answer gets parsed/stored). Deliberately more than just grade level:
    # interests and what feels hard to a student are the kind of thing a
    # real tutor picks up on over weeks of working with a 3rd-5th grader -
    # asking directly up front is the only way Jarvis gets it at all.
    ONBOARDING_QUESTIONS = [
        "What grade are you in?",
        "What's your favorite subject to learn about?",
        "What do you like to do for fun, outside of school?",
        "Is there anything about learning that feels hard or frustrating for you?",
    ]

    def __init__(self, ui_signals: Optional['UISignals'] = None):
        """Initialize Jarvis with all components."""
        logger.info("Initializing Jarvis...")
        self.ui_signals = ui_signals

        # Initialize shared PyAudio instance
        import pyaudio
        self.pa = pyaudio.PyAudio()

        # Initialize components with shared PyAudio
        self.wake_word_detector = WakeWordDetector(pa=self.pa)
        self.audio_player = AudioPlayer(
            pa=self.pa,
            # Qt signal emission is thread-safe (queued to the GUI thread)
            # unlike calling a widget method directly from this audio thread.
            on_level=self.ui_signals.mouth_level.emit if self.ui_signals else None
        )
        self.stt_client = SpeechToTextClient()
        self.llm_client = LLMClient()
        self.tts_client = TextToSpeechClient()
        # Verifies each client actually satisfies the provider-agnostic
        # interface (see interfaces.py) - not just documentation, a real
        # check that a future provider swap won't silently break main.py.
        assert isinstance(self.stt_client, SpeechToTextProvider)
        assert isinstance(self.llm_client, LLMProvider)
        assert isinstance(self.tts_client, TextToSpeechProvider)
        self.privacy_manager = PrivacyManager()

        # Grade-scoped curriculum retrieval (RAG) - the data was already
        # extracted (preprocess_textbooks.py) but never actually connected
        # to the LLM until now. Exposed to the LLM as a callable tool, see
        # llm_client.py's generate_response_with_tools().
        textbooks_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "processed_json_textbooks"
        )
        self.textbook_search = TextbookSearch(textbooks_dir)

        # Per-profile persistent history (one physical robot, multiple family
        # members) - resolved relative to this file's own location, not CWD,
        # same reasoning as the SYSTEM_PROMPT_PATH/face-path fixes.
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "jarvis_profiles.db"
        )
        self.profile_manager = ProfileManager(
            db_path=db_path,
            max_history=int(os.getenv('MAX_CONVERSATION_HISTORY', 20))
        )
        self.conversation_manager = self.profile_manager.get_conversation_manager()

        # Name-only identity resolution for now (see identity/identity_provider.py
        # for why voice biometrics aren't in scope yet) - a list of one
        # provider so a future SpeakerIdentityProvider can be appended
        # here without JarvisBot needing to change anywhere else.
        self.identity_resolver = IdentityResolver([NameIdentityProvider(self.profile_manager)])

        # Centralizes the hint/reteach/continue/challenge policy applied
        # after each comprehension-check answer - see tutor/decision_engine.py.
        self.tutor_decision_engine = TutorDecisionEngine()

        # Picks and freshens a retrieval-practice question - see tutor/review_scheduler.py.
        self.review_scheduler = ReviewScheduler(self.profile_manager, self.llm_client)

        # Concept prerequisite graph (starter scaffold, see curriculum/graph.py)
        # and the proactive suggestion engine built on top of it - lets
        # Jarvis offer something to work on instead of only ever reacting
        # to a question the student thought to ask.
        self.curriculum_graph = CurriculumGraph()
        self.proactive_question_engine = ProactiveQuestionEngine(self.profile_manager, self.curriculum_graph)

        # Text-heuristic signals (hesitation, repeated struggle) and the
        # voice-delivery mapping built on top of them - see
        # affect/interaction_signals.py and expression/controller.py.
        self.interaction_signal_engine = InteractionSignalEngine()
        self.expression_controller = ExpressionController()

        self._avatars_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "avatars"
        )
        
        self._is_running = False
        self._interaction_lock = threading.Lock()
        self._interruption_event = threading.Event()
        self._speaker_busy = threading.Event() # Track if speaker is active
        self._conversation_active = threading.Event() 
        self._request_queue = Queue()
        
        # ── Echo Guard State ──
        self._bot_is_speaking = False          # True while TTS audio is playing
        self._last_bot_response = ""           # Full text of last bot response
        self._playback_ended_time = 0.0        # time.time() when playback stopped
        self._barge_in_detected = threading.Event()  # Set by VAD when real user speech detected

        # ── New-profile onboarding state ──
        # A short Q&A after a brand new profile is created - doubles as
        # useful context-gathering and (later) enrollment audio for voice
        # verification, since one "my name is X" utterance isn't enough
        # speech for a reliable voiceprint. None when no onboarding is active.
        self._pending_onboarding: Optional[dict] = None

        # True right after asking "I heard Voice Recognition but didn't
        # catch a name" - the next utterance is the name, whatever it is,
        # not a fresh unrelated turn.
        self._pending_name_capture: bool = False

        # Set right after asking "what's your name?" at the start of a
        # conversation with no obvious active profile (see
        # _start_identity_checkin) - {"step"} where step is "ask_name" or
        # "confirm" (plus "candidate_id"/"candidate_name" once a fuzzy
        # name match needs confirming). None when no check-in is pending.
        # See _handle_identity_checkin_step.
        self._pending_identity_checkin: Optional[dict] = None

        # Set right after a teaching turn asks a comprehension-check
        # question (see _run_teaching_turn) - {"concept", "question",
        # "attempts"}. The next utterance is the student's answer to that
        # question, not a fresh unrelated turn. None when no check is
        # pending. See _handle_teaching_answer.
        self._pending_teaching_check: Optional[dict] = None

        # Set right at the start of a conversation, replacing the old
        # canned greeting (see _start_warmup_checkin) - {"step"} where step
        # is "warmup" or "discovery". The next one or two utterances are
        # answers to this check-in, not a fresh turn. None when no
        # check-in is pending. See _handle_warmup_step.
        self._pending_warmup_checkin: Optional[dict] = None

        # Set right after Jarvis proactively offers a concept to work on
        # (see _handle_warmup_step's discovery branch / tutor/question_engine.py)
        # - {"concept"}. The next utterance is a yes/no answer to that
        # offer, not a fresh turn. None when no offer is pending. See
        # _handle_proactive_suggestion_reply.
        self._pending_proactive_suggestion: Optional[dict] = None

        # ── Current conversation session (see session.py) ──
        # None outside of an active conversation; a fresh Session is created
        # each time the wake word fires.
        self.current_session: Optional[Session] = None

        # Output-only safety check (NeMo Guardrails) - checks what the LLM
        # actually said, run in the background in parallel with TTS already
        # speaking it (see _run_output_safety_check). Input-side checking
        # was evaluated and deliberately left disabled: it roughly doubled
        # per-turn latency and showed non-deterministic false-positive
        # blocks on trivial input like "hello" in live testing. Fails open
        # (GuardrailsManager.is_enabled is False) if NeMo/the config/the
        # env vars aren't available, so a guardrails outage never silences
        # Jarvis - it just runs without this safety net.
        self.guardrails_manager = GuardrailsManager(
            config_path=os.getenv('GUARDRAILS_CONFIG_PATH', 'config/guardrails')
        )
        # Synchronous keyword/pattern check layered in front of the async
        # NeMo check above - see fast_content_filter.py's module docstring
        # for why the async check alone leaves a real (if narrow)
        # exposure window before it can act.
        self.fast_content_filter = FastContentFilter()
        # Identifies which turn's audio a background safety check belongs
        # to (by object identity) - lets a check that resolves after the
        # conversation has already moved to a newer turn recognize it's
        # stale and skip interrupting unrelated, already-playing audio.
        self._active_turn_token = None

        # ── Display sleep tracking ──
        self._standby_since = time.time()      # When we last returned to wake-word listening
        self._display_sleeping = False         # Replaces the old FaceAnimator's self.face.sleeping

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
            # Release the old PyAudio/PortAudio host handle before replacing
            # it - every other shutdown path in this codebase (AudioPlayer.
            # shutdown(), WakeWordDetector's cleanup, JarvisBot.stop()) does
            # this; skipping it here leaked the old handle on every ALSA
            # crash-recovery cycle, which matters on a device meant to run
            # continuously.
            if hasattr(self, 'pa') and self.pa:
                try:
                    self.pa.terminate()
                except Exception:
                    pass
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
        return len(overlap) / min(len(words_a), len(words_b))

    def _speak_fixed_phrase(self, phrase: str, continuous_vad, output_device_index: int, delivery=None) -> None:
        """
        Speak a fixed, non-LLM-generated phrase (profile switch/creation
        confirmations). Same TTS/playback/echo-guard pattern as a normal
        turn in _speaker_loop, just with fixed text.

        delivery: optional expression.controller.DeliveryStyle - a
        per-utterance rate/pitch nudge (see ExpressionController), passed
        straight through to TextToSpeechClient.synthesize_stream(). None
        (the default) plays at the normal, unadjusted delivery.
        """
        logger.info(f"🗣️  {phrase}")
        try:
            self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
            if self.ui_signals:
                self.ui_signals.start_talking.emit()
            self.audio_player.start_streaming(output_device_index=output_device_index)
            self._bot_is_speaking = True
            continuous_vad.set_playback_state(True)

            for audio_chunk in self.tts_client.synthesize_stream(iter([phrase]), delivery=delivery):
                if self.audio_player._is_playing:
                    self.audio_player.queue_audio(audio_chunk)
                else:
                    break

            self.audio_player.stop_streaming(immediate=False)
            self._last_bot_response = phrase
        except Exception as e:
            logger.error(f"Error speaking fixed phrase: {e}")
        finally:
            if self.ui_signals:
                self.ui_signals.stop_talking.emit()
            self._bot_is_speaking = False
            self._playback_ended_time = time.time()
            continuous_vad.set_playback_state(False)
            self.audio_player.stop_streaming(immediate=True)

    def _run_output_safety_check(self, full_response: str, turn_token: object, continuous_vad, output_device_index: int) -> None:
        """
        Runs in a background thread, kicked off right after a turn's full
        response text is known - which by then is already synthesized and
        queued/playing through the speaker (see _speaker_loop). A real LLM
        round-trip (~8s live), too slow to gate speech-start on without
        badly hurting time-to-first-audio, so it acts as a safety net
        instead: if the response gets flagged, cut off playback immediately
        (same mechanism as barge-in interruption) and speak a kid-friendly
        refusal in its place.

        turn_token is compared by identity against self._active_turn_token
        so a check that resolves after the conversation has already moved
        on to a newer turn recognizes it's stale and doesn't interrupt
        unrelated, already-playing audio.
        """
        try:
            safe, refusal_text = self.guardrails_manager.check_output(full_response)
        except Exception as exc:
            logger.error(f"Output safety check crashed (failing open): {exc}")
            return

        if safe:
            return
        if self._active_turn_token is not turn_token:
            logger.warning(
                "Guardrails flagged a response, but the conversation already "
                "moved on to a newer turn - not interrupting."
            )
            return

        logger.warning(f"🚨 Guardrails BLOCKED output after it started playing: '{full_response[:80]}'")
        # request_immediate_stop() (not stop_streaming(), which this thread
        # doesn't own) only touches primitives that are already safe to use
        # cross-thread - see its docstring in playback.py for why calling
        # the full stop_streaming(immediate=True) here raced with the
        # speaker thread's own in-flight stop_streaming(immediate=False)
        # call and could reach PortAudio's stream.stop_stream() from two
        # threads at once.
        self.audio_player.request_immediate_stop()
        # Brief margin before opening a new stream for the refusal:
        # start_streaming() closes any stale stream under its own
        # "no worker thread is running yet" assumption - this gives the
        # speaker thread's own stop_streaming() call time to notice the
        # abort (one ~20ms callback cycle) and finish that cleanup itself
        # first, so this thread isn't racing it to touch the same stream.
        time.sleep(0.2)
        self._speak_fixed_phrase(refusal_text, continuous_vad, output_device_index)

    def _try_handle_profile_command(self, user_text: str, continuous_vad, output_device_index: int) -> bool:
        """
        Intercepts profile-management voice commands before they'd otherwise
        be sent to the LLM as a normal question. Returns True if user_text
        was such a command (already handled here - caller should skip the
        normal LLM turn for it), False if it's a normal conversational turn.
        """
        normalized = user_text.strip().lower()

        # "go back to your last session" (fuzzy - STT won't transcribe this
        # identically every time, so match on the key phrase, not verbatim)
        if "go back" in normalized and "session" in normalized:
            manager = self.profile_manager.go_back()
            if manager:
                self.conversation_manager = manager
                self._pending_onboarding = None  # switching profiles cancels any in-progress onboarding
                self._pending_name_capture = False
                self._pending_warmup_checkin = None
                self._pending_identity_checkin = None
                self._pending_proactive_suggestion = None
                name = self.profile_manager.get_profile_name(self.profile_manager.get_active_profile_id())
                self._speak_fixed_phrase(f"Okay, switching back to {name}'s session.", continuous_vad, output_device_index)
                self._start_warmup_checkin(continuous_vad, output_device_index)
            else:
                self._speak_fixed_phrase("There's no previous session to go back to.", continuous_vad, output_device_index)
            return True

        # "Jarvis Voice Recognition <name>" - switch to or create a profile
        trigger = "voice recognition"
        idx = normalized.find(trigger)
        if idx != -1:
            spoken_name = user_text[idx + len(trigger):].strip(" ,.!?")
            if not spoken_name:
                self._speak_fixed_phrase(
                    "I heard Voice Recognition, but I didn't catch a name — can you say your name too?",
                    continuous_vad, output_device_index
                )
                # Track that the NEXT utterance is the answer to this
                # question, regardless of its content - otherwise it falls
                # through to the normal LLM path and gets treated as an
                # unrelated question (confirmed via live testing: saying
                # "Ryan Lewis" as a follow-up got answered as if asking
                # about a music producer named Ryan Lewis).
                self._pending_name_capture = True
                return True

            self._switch_or_create_profile(spoken_name, continuous_vad, output_device_index)
            return True

        return False

    # Keyword classification for the "is that you?" confirmation in
    # _handle_identity_checkin_step - a plain regex, not an LLM call, since
    # this only needs to gate a binary decision and the LLM round-trip
    # would add real latency to identity resolution specifically.
    _YES_PATTERN = re.compile(r"\b(yes|yeah|yep|yup|correct|that'?s me|right|sure|uh[\s-]?huh)\b", re.IGNORECASE)
    _NO_PATTERN = re.compile(r"\b(no|nope|nah|not me|wrong|incorrect)\b", re.IGNORECASE)

    # Matches a discovery-step reply that already states a specific
    # request, as opposed to open-ended small talk - see
    # _handle_warmup_step's discovery branch.
    _EXPLICIT_INTENT_PATTERN = re.compile(
        r"\b(help|learn|teach|practice|work on|curious about|homework|question)\b", re.IGNORECASE
    )

    def _classify_yes_no(self, text: str) -> Optional[bool]:
        """
        Returns True/False for a clear yes/no, None when neither pattern
        matches. None means "ask again" (see _handle_identity_checkin_step) -
        treating an unclear answer as confirmation would risk attaching
        this conversation's learning history to the wrong student, which
        is worse than asking once more.
        """
        if self._NO_PATTERN.search(text):
            return False
        if self._YES_PATTERN.search(text):
            return True
        return None

    def _start_identity_checkin(self, continuous_vad, output_device_index: int) -> None:
        """
        Replaces the old "please say Voice Recognition, then your name"
        prompt for a conversation with no obvious active profile (the
        device's last-active profile is still the default placeholder) -
        see _handle_wake_word. Runs the spoken name through
        self.identity_resolver rather than creating a profile outright -
        see _handle_identity_checkin_step for why an existing-name match
        is always confirmed out loud instead of silently assumed (two
        people, e.g. siblings, can share a name).
        """
        phrase = os.getenv('IDENTITY_CHECKIN_MESSAGE', "Hey! I don't think we've met yet. What's your name?")
        self._speak_fixed_phrase(phrase, continuous_vad, output_device_index)
        self.conversation_manager.add_assistant_message(phrase)
        self._pending_identity_checkin = {"step": "ask_name"}
        continuous_vad.reset_idle_timer()

    def _handle_identity_checkin_step(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Handles either step of the identity check-in started by
        _start_identity_checkin: "ask_name" resolves a freshly spoken name
        against known profiles (unknown -> onboard as new; uncertain ->
        confirm out loud before switching); "confirm" handles the
        yes/no reply to that confirmation.
        """
        pending = self._pending_identity_checkin

        if pending["step"] == "ask_name":
            spoken_name = user_text.strip(" ,.!?")
            result = self.identity_resolver.resolve(IdentityObservation(spoken_name=spoken_name))

            if result.status == "uncertain":
                self._pending_identity_checkin = {
                    "step": "confirm", "candidate_id": result.profile_id, "candidate_name": result.name,
                }
                self._speak_fixed_phrase(
                    f"Oh, I know a {result.name} already - is that you?", continuous_vad, output_device_index
                )
                continuous_vad.reset_idle_timer()
                return

            # Unknown - no existing profile has this name. Hand off to the
            # existing create+onboarding flow rather than duplicating it.
            self._pending_identity_checkin = None
            self._switch_or_create_profile(spoken_name, continuous_vad, output_device_index)
            return

        # step == "confirm"
        confirmed = self._classify_yes_no(user_text)
        if confirmed is None:
            self._speak_fixed_phrase("Sorry, is that you - yes or no?", continuous_vad, output_device_index)
            continuous_vad.reset_idle_timer()
            return

        if confirmed:
            candidate_id = pending["candidate_id"]
            candidate_name = pending["candidate_name"]
            self._pending_identity_checkin = None
            self.conversation_manager = self.profile_manager.switch_to(candidate_id)
            self._speak_fixed_phrase(f"Welcome back, {candidate_name}!", continuous_vad, output_device_index)
            self._start_warmup_checkin(continuous_vad, output_device_index)
        else:
            # Not them - a name collision (two different people, same
            # first name). Ask again rather than silently creating a
            # profile with the exact same name, which would hit the
            # UNIQUE constraint on profiles.name.
            self._pending_identity_checkin = {"step": "ask_name"}
            self._speak_fixed_phrase(
                "Got it - what should I call you, so I don't mix you two up?", continuous_vad, output_device_index
            )
            continuous_vad.reset_idle_timer()

    def _switch_or_create_profile(self, spoken_name: str, continuous_vad, output_device_index: int) -> None:
        """
        Switch to (or create) a profile by name, speaking the appropriate
        confirmation and kicking off onboarding for new profiles. Shared by
        both the direct "Voice Recognition <name>" trigger and the
        name-recovery follow-up (see _pending_name_capture).
        """
        profile_id, created = self.profile_manager.find_or_create_profile(spoken_name)
        self.conversation_manager = self.profile_manager.switch_to(profile_id)
        self._pending_onboarding = None  # switching profiles cancels any in-progress onboarding
        self._pending_name_capture = False
        self._pending_warmup_checkin = None
        self._pending_identity_checkin = None
        self._pending_proactive_suggestion = None

        if created:
            self._speak_fixed_phrase(
                f"Nice to meet you, {spoken_name}! I'll remember our conversations from now on. "
                "Let me take your picture - look at the camera!",
                continuous_vad, output_device_index
            )

            avatar_path = os.path.join(self._avatars_dir, f"{profile_id}.jpg")
            if capture_avatar_photo(avatar_path):
                self.profile_manager.set_avatar(profile_id, avatar_path)
            # If capture fails (no camera, wrong platform, etc.) we just
            # skip the avatar - not fatal to profile creation.

            self._speak_fixed_phrase(
                f"Let's get to know each other a bit - {self.ONBOARDING_QUESTIONS[0]}",
                continuous_vad, output_device_index
            )
            self._pending_onboarding = {"profile_id": profile_id, "question_index": 1}
        else:
            self._speak_fixed_phrase(f"Welcome back, {spoken_name}!", continuous_vad, output_device_index)
            self._start_warmup_checkin(continuous_vad, output_device_index)

    def _handle_onboarding_answer(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Records the answer to the current onboarding question, then asks the
        next one or wraps up. Stored as normal conversation turns (via
        add_user_message/add_assistant_message) so it's genuinely remembered,
        not just thrown away after asking.
        """
        anonymized = self.privacy_manager.anonymize(user_text)
        self.conversation_manager.add_user_message(anonymized)

        next_index = self._pending_onboarding["question_index"]
        profile_id = self._pending_onboarding["profile_id"]

        # question_index == 1 means this answer is responding to
        # ONBOARDING_QUESTIONS[0] ("What grade are you in?") - parse and
        # store it so RAG retrieval (search_curriculum tool) can scope
        # results to this student's actual grade level.
        if next_index == 1:
            grade = normalize_grade(user_text)
            if grade:
                self.profile_manager.set_grade(profile_id, grade)
                logger.info(f"📓 Stored grade '{grade}' for profile {profile_id}")
            else:
                logger.info(f"📓 Could not parse a grade from: '{user_text}'")

        # question_index == 2 answers "favorite subject" - the start of
        # this student's interests, used later to frame examples around
        # things they actually care about (see LLMClient's student_context).
        elif next_index == 2:
            self.profile_manager.set_interests(profile_id, f"Enjoys learning about: {user_text.strip()}.")

        # question_index == 3 answers "what do you like to do for fun" -
        # appended onto interests rather than overwriting the subject answer.
        elif next_index == 3:
            existing = self.profile_manager.get_interests(profile_id) or ""
            self.profile_manager.set_interests(profile_id, f"{existing} Outside school, likes: {user_text.strip()}.".strip())

        # question_index == 4 answers "what feels hard or frustrating" -
        # stored separately from interests since it's used differently
        # (pacing/tone/encouragement, not example framing).
        elif next_index == 4:
            self.profile_manager.set_learning_challenges(profile_id, user_text.strip())

        if next_index < len(self.ONBOARDING_QUESTIONS):
            question = self.ONBOARDING_QUESTIONS[next_index]
            self._speak_fixed_phrase(question, continuous_vad, output_device_index)
            self.conversation_manager.add_assistant_message(question)
            self._pending_onboarding["question_index"] = next_index + 1
        else:
            wrap_up = "Great, thanks for telling me about yourself! I'm ready whenever you want to start learning."
            self._speak_fixed_phrase(wrap_up, continuous_vad, output_device_index)
            self.conversation_manager.add_assistant_message(wrap_up)
            self._pending_onboarding = None

    def _tool_search_curriculum(self, query: str) -> str:
        """Implementation behind the search_curriculum tool the LLM can call."""
        profile_id = self.profile_manager.get_active_profile_id()
        grade = self.profile_manager.get_grade(profile_id) if profile_id else None
        if not grade:
            return "No grade level is known for this student yet, so results may not be grade-appropriate."

        results = self.textbook_search.search(query, grade)
        if not results:
            return f"No matching curriculum content found for grade {grade} on '{query}'."
        return "\n\n".join(results)

    def _build_student_context(self) -> Optional[str]:
        """
        Assembles the active profile's interests/learning-challenges (see
        ONBOARDING_QUESTIONS/_handle_onboarding_answer) into a short blurb
        passed to the LLM as student_context - see LLMClient._with_student_
        context. Returns None for the default/no-profile case, or a
        profile that hasn't been through onboarding yet, so callers don't
        need to special-case "nothing on file".
        """
        profile_id = self.profile_manager.get_active_profile_id()
        if not profile_id:
            return None

        interests = self.profile_manager.get_interests(profile_id)
        challenges = self.profile_manager.get_learning_challenges(profile_id)
        # Recent misconceptions were being captured (see record_learning_event)
        # but never read back anywhere until now - surfacing them here so
        # the LLM can actually watch for a misunderstanding recurring
        # instead of re-discovering it from scratch each time.
        misconceptions = self.profile_manager.get_recent_misconceptions(profile_id, limit=3)

        parts = []
        if interests:
            parts.append(interests.strip())
        if challenges:
            parts.append(f"Finds this hard/frustrating: {challenges}")
        if misconceptions:
            summary = "; ".join(f"{m['concept']}: {m['misconception']}" for m in misconceptions)
            parts.append(f"Past misconceptions to watch for: {summary}")
        return " ".join(parts) if parts else None

    # Step size per "louder"/"quieter" call, and the hard ceiling/floor -
    # matches the clamp already enforced in AudioPlayer.set_volume().
    _VOLUME_STEP = 0.25
    _VOLUME_MIN = 0.25
    _VOLUME_MAX = 2.0

    def _tool_set_volume(self, level: str) -> str:
        """Implementation behind the set_volume tool the LLM can call."""
        current = self.audio_player.volume
        if level == "louder":
            new_volume = min(current + self._VOLUME_STEP, self._VOLUME_MAX)
        elif level == "quieter":
            new_volume = max(current - self._VOLUME_STEP, self._VOLUME_MIN)
        elif level == "normal":
            new_volume = 1.0
        else:
            return f"Unknown volume level: {level}"

        self.audio_player.set_volume(new_volume)
        if self.ui_signals:
            self.ui_signals.volume_changed.emit(new_volume)

        if new_volume >= self._VOLUME_MAX:
            detail = "Volume raised to maximum."
        elif new_volume <= self._VOLUME_MIN:
            detail = "Volume lowered to minimum."
        else:
            detail = f"Volume set to {int(new_volume * 100)}% of normal."
        # The tool already succeeded by the time the model sees this -
        # phrased as a direct instruction because gpt-4o-mini otherwise
        # tends to hedge and tell the student it "can't" change volume
        # even in the same turn it just changed it (seen live).
        return f"{detail} This already happened. Confirm it to the student now - do not say you can't change volume."

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Dispatches a tool call requested by the LLM and records it in the
        current session's audit trail (Session.tool_calls)."""
        if tool_name == "search_curriculum":
            result = self._tool_search_curriculum(args.get("query", ""))
        elif tool_name == "set_volume":
            result = self._tool_set_volume(args.get("level", ""))
        else:
            result = f"Unknown tool: {tool_name}"

        if self.current_session:
            self.current_session.record_tool_call(tool_name, args, result)
        return result

    def is_math_query(self, text: str) -> bool:
        """Detect math/geometry questions that should get a teaching plan
        (whiteboard diagram + synced speech) instead of a plain answer."""
        t = text.lower().strip()

        math_keywords = [
            # arithmetic / algebra
            "solve", "equation", "equations", "add", "subtract", "multiply", "divide",
            "fraction", "algebra", "calculate", "compute", "simplify", "evaluate",
            "factor", "expand", "expression", "variable", "coefficient",
            # geometry - shapes
            "area", "perimeter", "surface area",
            "radius", "diameter", "circumference",
            "rectangle", "square", "circle", "triangle", "polygon",
            "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
            "trapezoid", "trapezium", "parallelogram", "rhombus", "kite",
            "ellipse", "oval", "sector", "segment",
            "sided", "sides", "shape", "diagonal", "hypotenuse",
            # geometry - measurements
            "angle", "degree", "height", "width", "length", "base", "depth",
            "pythagorean", "theorem", "congruent", "similar",
            # misc math
            "graph", "geometry", "probability", "percent", "ratio", "proportion",
            "mean", "median", "mode", "average", "prime", "exponent", "power",
            "square root", "cube root", "logarithm",
        ]

        if any(word in t for word in math_keywords):
            return True

        # "volume" alone is ambiguous - a real geometry term (volume of a
        # cube) but also loudness (see the set_volume tool). A bare
        # substring match here sent "please tell me in maximum volume"
        # to the teaching-plan pipeline instead of set_volume, confirmed
        # live - the JSON-only prompt got a conversational reply back and
        # errored, ultimately just apologizing instead of changing the
        # volume. Only treat it as math when paired with "of" or an actual
        # 3D-shape word, not just present anywhere in the sentence.
        if re.search(r"\bvolume\s+of\b", t) or (
            "volume" in t
            and re.search(r"\b(cube|sphere|cylinder|cone|prism|pyramid|box|container|tank)\b", t)
        ):
            return True

        return bool(re.search(r"\d", t) and re.search(r"[\+\-\*/=]", t))

    def _start_warmup_checkin(self, continuous_vad, output_device_index: int) -> None:
        """
        Replaces the old canned self-introduction ("Hi! I'm Jarvis...") at
        the very start of a conversation, for any profile that isn't the
        default placeholder - see _handle_wake_word. A real tutor doesn't
        open with a scripted pitch; a short, genuine check-in ("how's your
        day going") reads as someone who's actually paying attention.
        Two short back-and-forth exchanges, not the original spec's full
        1-2 minutes of small talk - Jarvis is used ambiently, so the
        check-in itself needs to stay quick.
        """
        profile_id = self.profile_manager.get_active_profile_id()
        name = self.profile_manager.get_profile_name(profile_id) if profile_id else None

        default_checkin = f"Hey {name}! How's your day going so far?" if name else "Hey! How's your day going so far?"
        phrase = os.getenv('WARMUP_CHECKIN_MESSAGE', default_checkin)

        self._speak_fixed_phrase(phrase, continuous_vad, output_device_index)
        self.conversation_manager.add_assistant_message(phrase)
        self._pending_warmup_checkin = {"step": "warmup"}
        continuous_vad.reset_idle_timer()

    def _handle_warmup_step(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Handles the student's reply to either half of the warm-up
        check-in started by _start_warmup_checkin. The first reply is
        just recorded (genuinely remembered, not analyzed yet - see the
        review's InteractionSignals/StudentModel proposals for where that
        would eventually plug in) and followed by a second, lighter
        question. The second reply is deliberately NOT consumed here -
        it's put back on the request queue so it flows through the
        normal dispatch chain instead (see _speaker_loop): the student may
        already be answering with their actual request ("help me with
        fractions"), and treating that as throwaway small talk would just
        make them repeat themselves.
        """
        pending = self._pending_warmup_checkin

        if pending["step"] == "warmup":
            anonymized = self.privacy_manager.anonymize(user_text)
            self.conversation_manager.add_user_message(anonymized)

            phrase = os.getenv(
                'DISCOVERY_MESSAGE',
                "Nice. Is there something you'd like help with today, or something new you're curious about?"
            )
            self._speak_fixed_phrase(phrase, continuous_vad, output_device_index)
            self.conversation_manager.add_assistant_message(phrase)
            pending["step"] = "discovery"
            continuous_vad.reset_idle_timer()
            return

        # step == "discovery" - interlude is done. Mark it so nothing
        # later this conversation re-triggers it.
        self._pending_warmup_checkin = None
        if self.current_session:
            self.current_session.warmup_done = True

        # A reply that already looks like a specific request ("help me
        # with fractions", or a math question outright) goes to the
        # normal dispatch chain unconsumed (see docstring above) rather
        # than being treated as throwaway small talk. Only a genuinely
        # open-ended reply ("not really", "I don't know") triggers a
        # proactive suggestion - see tutor/question_engine.py - instead of
        # falling through to a generic chat response with nothing to go on.
        if self.is_math_query(user_text) or self._EXPLICIT_INTENT_PATTERN.search(user_text):
            self._request_queue.put(user_text)
            return

        profile_id = self.profile_manager.get_active_profile_id()
        suggestion = self.proactive_question_engine.suggest(profile_id) if profile_id else None
        if suggestion:
            self._speak_fixed_phrase(suggestion["prompt"], continuous_vad, output_device_index)
            self.conversation_manager.add_assistant_message(suggestion["prompt"])
            self._pending_proactive_suggestion = {"concept": suggestion["concept"]}
            continuous_vad.reset_idle_timer()
        else:
            self._request_queue.put(user_text)

    def _handle_proactive_suggestion_reply(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Handles the yes/no reply to a proactive concept suggestion (see
        _handle_warmup_step's discovery branch). A clear yes turns the
        offer into an actual teaching turn on that concept - without this,
        the offer was a dead end: Jarvis would ask "want to work on
        fractions?" and then do nothing with a "yes" beyond letting it hit
        generic chat. A no (or anything unclear) just lets the reply flow
        through normal dispatch instead of forcing the suggestion on them.
        """
        pending = self._pending_proactive_suggestion
        self._pending_proactive_suggestion = None

        if self._YES_PATTERN.search(user_text):
            concept_readable = pending["concept"].replace("_", " ")
            self._run_teaching_turn(f"Can you help me with {concept_readable}?", continuous_vad, output_device_index)
        else:
            self._request_queue.put(user_text)

    def _run_teaching_turn(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Handle a math/geometry question: generate a teaching plan (speech
        steps + synchronized whiteboard draw actions) via
        LLMClient.generate_teaching_plan(), instead of a plain LLM answer.
        Each step's draw actions are emitted right before that step's speech
        is synthesized/played, using the same streaming TTS pipeline as
        normal turns (one text chunk per step, since each step's text is
        already complete - not token-by-token like a live LLM stream).
        """
        self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
        if self.ui_signals:
            self.ui_signals.show_teaching_layout.emit()
            self.ui_signals.clear_canvas.emit()
            self.ui_signals.start_talking.emit()
        self.audio_player.start_streaming(output_device_index=output_device_index)
        self._bot_is_speaking = True
        continuous_vad.set_playback_state(True)

        response_parts = []
        try:
            logger.info(f"📐 Processing Teaching Turn: {user_text}")
            anonymized_text = self.privacy_manager.anonymize(user_text)
            self.conversation_manager.add_user_message(anonymized_text)
            messages = self.conversation_manager.get_messages()

            plan = self.llm_client.generate_teaching_plan(messages, student_context=self._build_student_context())
            visuals = plan.get("visuals", [])
            concept = plan.get("concept") or "general_math"
            check_question = plan.get("check_question")

            fast_blocked = False
            for step in plan.get("speech", []):
                if self._interruption_event.is_set():
                    break

                step_text = step.get("text", "").strip()
                if not step_text:
                    continue

                # Fast synchronous pre-check (see guardrails/fast_content_filter.py) -
                # cheap enough to run inline without delaying time-to-first-audio,
                # unlike the ~8s async NeMo check below. Aborts the whole plan
                # rather than skipping just this one step, so a refusal never
                # gets sandwiched in the middle of an otherwise-normal explanation.
                fast_safe, fast_refusal = self.fast_content_filter.check(step_text)
                if not fast_safe:
                    logger.warning(f"🚨 Fast content filter blocked a teaching step: '{step_text[:80]}'")
                    fast_blocked = True
                    break

                actions = [v for v in visuals if v.get("speech_id") == step.get("id")]
                if actions and self.ui_signals:
                    self.ui_signals.draw_actions.emit(actions)

                response_parts.append(step_text)
                for audio_chunk in self.tts_client.synthesize_stream(iter([step_text])):
                    if self._interruption_event.is_set():
                        break
                    if self.audio_player._is_playing:
                        try:
                            self.audio_player.queue_audio(audio_chunk)
                        except RuntimeError as e:
                            logger.warning(f"⚠️ Playback queueing failed (likely stopped): {e}")
                            break
                    else:
                        break

            if fast_blocked:
                # Same abort pattern as _run_output_safety_check - only
                # touch primitives that are safe regardless of what
                # in-flight audio state this stream is currently in, then
                # a brief settle margin before opening a fresh stream for
                # the refusal (see that method's docstring for why).
                self.audio_player.request_immediate_stop()
                time.sleep(0.2)
                self._speak_fixed_phrase(fast_refusal, continuous_vad, output_device_index)
                self._last_bot_response = fast_refusal
                continuous_vad.reset_idle_timer()
                return

            # Ask the comprehension-check question, if the plan included
            # one, right after the explanation - same streaming pattern as
            # the speech steps above. Part of the same response for safety
            # checking/history purposes (appended to response_parts).
            if not self._interruption_event.is_set() and check_question:
                response_parts.append(check_question)
                for audio_chunk in self.tts_client.synthesize_stream(iter([check_question])):
                    if self._interruption_event.is_set():
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
                # Same output-safety net as the normal conversational path
                # (see _speaker_loop) - teaching-turn speech is still
                # LLM-generated text and was previously not checked at all,
                # a real gap since any math question routes here via
                # is_math_query(). Launched before the blocking wait below
                # for the same reason: doesn't delay time-to-first-audio.
                if self.guardrails_manager.is_enabled:
                    full_response_so_far = " ".join(response_parts)
                    if full_response_so_far:
                        turn_token = object()
                        self._active_turn_token = turn_token
                        threading.Thread(
                            target=self._run_output_safety_check,
                            args=(full_response_so_far, turn_token, continuous_vad, output_device_index),
                            daemon=True,
                        ).start()

                self.audio_player.stop_streaming(immediate=False)
                full_response = " ".join(response_parts)
                if full_response:
                    self._last_bot_response = full_response
                    self.conversation_manager.add_assistant_message(full_response)
                    logger.info(f"🤖 Bot (teaching): {full_response}")
                    if self.current_session:
                        self.current_session.record_concept_covered(concept)

                # Set AFTER speaking, so the next turn's answer routes to
                # _handle_teaching_answer instead of being treated as a
                # fresh, unrelated question.
                if check_question:
                    self._pending_teaching_check = {"concept": concept, "question": check_question, "attempts": 0}
            else:
                partial = " ".join(response_parts)
                if partial:
                    self._last_bot_response = partial

            continuous_vad.reset_idle_timer()

        except Exception as e:
            logger.error(f"Error in teaching turn: {e}")
            # Fall back to a spoken apology rather than leaving the student
            # in silence if the LLM didn't return a valid teaching plan.
            try:
                apology = "Sorry, I had trouble working that one out. Could you try asking again?"
                for audio_chunk in self.tts_client.synthesize_stream(iter([apology])):
                    if self.audio_player._is_playing:
                        self.audio_player.queue_audio(audio_chunk)
                self._last_bot_response = apology
            except Exception:
                pass
        finally:
            if self.ui_signals:
                self.ui_signals.stop_talking.emit()
            self._bot_is_speaking = False
            self._playback_ended_time = time.time()
            continuous_vad.set_playback_state(False)
            self.audio_player.stop_streaming(immediate=True)

    # Matches asking Jarvis to repeat/clarify the question - NOT an answer
    # attempt. Found live: "can you say the question again?" was fed
    # straight into evaluate_answer(), which judged it as a wrong answer
    # ("the student did not attempt to answer the question"), burned an
    # attempt, and gave a hint instead of just repeating the question -
    # which then made the student's actual next attempt get unfairly
    # escalated straight to "reteach" since the attempt counter was
    # already at 1.
    _REPEAT_REQUEST_PATTERN = re.compile(
        r"\b(repeat (the|that|it)?|say (that|it|the question)( again)?|"
        r"what was the question|didn't (catch|hear)|come again|one more time)\b",
        re.IGNORECASE,
    )

    def _handle_teaching_answer(self, user_text: str, continuous_vad, output_device_index: int) -> None:
        """
        Handles the student's answer to a teaching turn's comprehension-
        check question (see _run_teaching_turn, which sets
        self._pending_teaching_check before returning).

        A deliberately small policy - just the rules that matter for a
        single comprehension check: a correct answer means praise and move
        on; a first wrong attempt gets one hint and a second try at the
        same question; a repeated wrong attempt stops re-testing and
        explains the concept a different way instead, rather than looping
        forever on a question the student clearly isn't getting. The
        actual judgment AND the response text itself both come from
        LLMClient.evaluate_answer(); self.tutor_decision_engine turns that
        into a concrete TutorDecision (see tutor/decision_engine.py) -
        this method applies it and updates persistent mastery tracking.
        """
        pending = self._pending_teaching_check

        if self._REPEAT_REQUEST_PATTERN.search(user_text):
            # Not an answer attempt - just re-ask the same question,
            # without touching attempts/mastery or the pending state.
            self._speak_fixed_phrase(pending["question"], continuous_vad, output_device_index)
            continuous_vad.reset_idle_timer()
            return

        self._pending_teaching_check = None  # consumed either way below

        concept = pending["concept"]
        question = pending["question"]
        attempts_so_far = pending["attempts"]

        try:
            anonymized_text = self.privacy_manager.anonymize(user_text)
            self.conversation_manager.add_user_message(anonymized_text)

            evaluation = self.llm_client.evaluate_answer(question, concept, user_text, attempts_so_far)
            decision = self.tutor_decision_engine.decide(evaluation)

            profile_id = self.profile_manager.get_active_profile_id()
            if profile_id:
                self.profile_manager.record_attempt(
                    profile_id, concept, correct=decision.is_correct, used_hint=decision.used_hint, question=question
                )
                self.profile_manager.record_learning_event(
                    profile_id,
                    session_id=self.current_session.session_id if self.current_session else None,
                    concept=concept,
                    result=evaluation.get("correctness"),
                    action=decision.action.value,
                    attempt_number=attempts_so_far + 1,
                    misconception=evaluation.get("misconception"),
                )

            logger.info(
                f"📊 Answer evaluation for '{concept}': {evaluation.get('correctness')} -> {decision.action.value}"
            )

            # Fast synchronous pre-check (see guardrails/fast_content_filter.py) -
            # same reasoning as _run_teaching_turn's per-step check.
            response_text = decision.response_text
            fast_safe, fast_refusal = self.fast_content_filter.check(response_text)
            if not fast_safe:
                logger.warning(f"🚨 Fast content filter blocked an answer-evaluation response: '{response_text[:80]}'")
                response_text = fast_refusal

            signals = self.interaction_signal_engine.analyze(user_text, attempts_so_far)
            delivery = self.expression_controller.for_action(
                decision.action, repeated_struggle=signals.repeated_struggle, hesitation=signals.hesitation
            )
            if self.ui_signals:
                self.ui_signals.set_expression.emit(delivery.expression.value)

            self._speak_fixed_phrase(response_text, continuous_vad, output_device_index, delivery=delivery)
            self.conversation_manager.add_assistant_message(response_text)
            self._last_bot_response = response_text

            if decision.action == TutorAction.HINT:
                # One more attempt at the SAME question.
                self._pending_teaching_check = {
                    "concept": concept, "question": question, "attempts": attempts_so_far + 1,
                }
            # "reteach"/"continue"/"challenge" all leave pending state
            # cleared (already done above): reteach stops re-testing this
            # question rather than looping on it, continue/challenge are done.

            # Retrieval practice: once per conversation, right after a
            # genuine success, briefly revisit the student's weakest older
            # concept instead of only ever moving forward - the REMEMBER
            # step the original tutoring loop was missing entirely. Reuses
            # this exact same pending-check/_handle_teaching_answer
            # machinery, so a wrong answer here still gets a hint/reteach
            # normally. Excludes concepts already covered this session so
            # it never re-asks about what was just taught. Skipped when
            # the student sounded low-confidence about THIS answer even
            # though it was right ("um, maybe 20?") - piling a second,
            # unrelated question onto genuine uncertainty doesn't serve
            # them; let them consolidate this one first.
            if (
                decision.is_correct
                and not signals.low_confidence
                and profile_id
                and self.current_session
                and not self.current_session.retrieval_practice_offered
            ):
                review = self.review_scheduler.get_review(
                    profile_id, exclude=set(self.current_session.concepts_covered)
                )
                if review:
                    self.current_session.retrieval_practice_offered = True
                    self.current_session.record_concept_covered(review["concept"])
                    bridge = f"Before we move on, let's check something from before. {review['question']}"
                    self._speak_fixed_phrase(bridge, continuous_vad, output_device_index)
                    self.conversation_manager.add_assistant_message(bridge)
                    self._pending_teaching_check = {
                        "concept": review["concept"], "question": review["question"], "attempts": 0,
                    }

            continuous_vad.reset_idle_timer()

        except Exception as e:
            logger.error(f"Error handling teaching answer: {e}")
            # Don't leave the student in silence, and don't leave a stale
            # pending check hanging around for an unrelated future turn
            # (already cleared above, before the try block).
            try:
                self._speak_fixed_phrase(
                    "Sorry, I had trouble with that. Let's try a different question.",
                    continuous_vad, output_device_index
                )
            except Exception:
                pass

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
                # Reject finalized text that matches the bot's last response
                if is_final and self._last_bot_response:
                    sim = self._text_similarity(text, self._last_bot_response)
                    if sim > SIMILARITY_THRESHOLD:
                        logger.info(f"🔇 Echo rejected (similarity={sim:.0%}): '{text[:60]}...'")
                        continue
                
                # ── FINISHED UTTERANCE - Queue for processing ──
                if is_final:
                    logger.info(f"📌 User finalized: '{text}'")
                    self._interruption_event.clear()
                    self._barge_in_detected.clear()
                    request_queue.put(text)
                    
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
                    user_text = self._request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if user_text is None: # Termination sentinel
                    break
                    
                # Signal we are busy
                self._speaker_busy.set()
                self._interruption_event.clear()
                self._barge_in_detected.clear()

                # Profile commands, name-recovery, onboarding-answers, and
                # math/teaching turns are all dispatched here, before the
                # normal LLM path. Wrapped in one try/except so a bug in any
                # of these can't kill the whole speaker thread for the rest
                # of the conversation - confirmed live: an uncaught "no such
                # column: grade" error in _handle_onboarding_answer did
                # exactly that, silencing Jarvis for the remainder of the
                # session (the normal-turn path below already has its own
                # try/except and doesn't need this - only these four didn't).
                try:
                    # "Jarvis Voice Recognition <name>", "go back to your
                    # last session" - never reach the LLM, not real
                    # tutoring questions.
                    if self._try_handle_profile_command(user_text, continuous_vad, output_device_index):
                        self._speaker_busy.clear()
                        continuous_vad.reset_idle_timer()
                        continue

                    # This turn is a reply to the "what's your name?"/"is
                    # that you?" identity check-in run at conversation
                    # start when there was no obvious active profile - see
                    # _start_identity_checkin/_handle_identity_checkin_step.
                    if self._pending_identity_checkin is not None:
                        self._handle_identity_checkin_step(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continuous_vad.reset_idle_timer()
                        continue

                    # Recovering a name after "I heard Voice Recognition but
                    # didn't catch a name" - this turn IS the name, not a
                    # tutoring question, regardless of what it contains.
                    if self._pending_name_capture:
                        self._pending_name_capture = False
                        self._switch_or_create_profile(user_text.strip(), continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continuous_vad.reset_idle_timer()
                        continue

                    # Mid-onboarding: this turn is an answer to the current
                    # onboarding question, not a real tutoring question either.
                    if self._pending_onboarding is not None:
                        self._handle_onboarding_answer(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continuous_vad.reset_idle_timer()
                        continue

                    # This turn is the student's answer to a teaching
                    # turn's comprehension-check question, not a fresh
                    # question - see _run_teaching_turn/_handle_teaching_answer.
                    if self._pending_teaching_check is not None:
                        self._handle_teaching_answer(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continuous_vad.reset_idle_timer()
                        continue

                    # This turn is the student's reply to one half of the
                    # warm-up check-in - see _start_warmup_checkin/_handle_warmup_step.
                    if self._pending_warmup_checkin is not None:
                        self._handle_warmup_step(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continue

                    # This turn is a yes/no reply to a proactive concept
                    # suggestion - see _handle_proactive_suggestion_reply.
                    if self._pending_proactive_suggestion is not None:
                        self._handle_proactive_suggestion_reply(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continue

                    # Math/geometry questions get a teaching plan (speech
                    # steps + synchronized whiteboard diagram) instead of a
                    # plain answer.
                    if self.is_math_query(user_text):
                        self._run_teaching_turn(user_text, continuous_vad, output_device_index)
                        self._speaker_busy.clear()
                        continue
                except Exception as dispatch_err:
                    logger.error(f"Error handling turn dispatch (recovering, not ending the conversation): {dispatch_err}")
                    self._speaker_busy.clear()
                    continuous_vad.reset_idle_timer()
                    continue

                try:
                    logger.info(f"📝 Processing Turn: {user_text}")
                    turn_start = time.perf_counter()
                    if self.current_session:
                        self.current_session.new_turn()

                    # Anonymize and prepare history
                    anonymized_text = self.privacy_manager.anonymize(user_text)
                    self.conversation_manager.add_user_message(anonymized_text)

                    # LLM Generation
                    messages = self.conversation_manager.get_messages()

                    # Start audio playback (callback mode)
                    # AEC: Provide reference audio back to VAD
                    self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
                    if self.ui_signals:
                        self.ui_signals.show_face_fullscreen.emit()
                        self.ui_signals.start_talking.emit()
                    self.audio_player.start_streaming(output_device_index=output_device_index)

                    response_chunks = []
                    latency: Dict[str, float] = {}
                    first_audio_chunk = True

                    # Helper to collect text while streaming, timestamping
                    # the first chunk for the LLM-TTFT latency figure.
                    def text_collector(stream):
                        for chunk in stream:
                            if self._interruption_event.is_set():
                                break
                            if not response_chunks:
                                latency["llm_ttft_ms"] = round((time.perf_counter() - turn_start) * 1000, 1)
                            response_chunks.append(chunk)
                            yield chunk

                    # Helper to timestamp the first audio chunk for the
                    # TTS-first-byte latency figure.
                    def audio_timing(stream):
                        for i, chunk in enumerate(stream):
                            if i == 0:
                                latency["tts_first_byte_ms"] = round((time.perf_counter() - turn_start) * 1000, 1)
                            yield chunk

                    # Pipeline: LLM (with tool-calling) -> TTS -> AudioPlayer
                    llm_stream = self.llm_client.generate_response_with_tools(
                        messages, tools=[CURRICULUM_SEARCH_TOOL, VOLUME_CONTROL_TOOL], tool_executor=self._execute_tool,
                        student_context=self._build_student_context()
                    )
                    collected_stream = text_collector(llm_stream)
                    tts_stream = audio_timing(self.tts_client.synthesize_stream(collected_stream))

                    for audio_chunk in tts_stream:
                        if self._interruption_event.is_set():
                            logger.warning("🛑 Speaker aborted due to interruption event")
                            break

                        # ── ECHO GUARD: Close the STT gate only once real audio is
                        # about to play — not when the turn starts processing. Gating
                        # earlier mutes the STT pipe for the entire LLM+TTS-first-chunk
                        # latency (1-2+ seconds), silently dropping any speech the user
                        # makes during that "thinking" gap instead of transcribing it.
                        if first_audio_chunk:
                            self._bot_is_speaking = True
                            continuous_vad.set_playback_state(True)
                            first_audio_chunk = False

                        # Defensive check: ensure streaming is still active
                        if self.audio_player._is_playing:
                            try:
                                self.audio_player.queue_audio(audio_chunk)
                            except RuntimeError as e:
                                logger.warning(f"⚠️ Playback queueing failed (likely stopped): {e}")
                                break
                        else:
                            break

                    # Wait for playback to finish naturally (if not interrupted)
                    if not self._interruption_event.is_set():
                        # Kick off the output safety check now, before the
                        # blocking wait below - the response is already
                        # synthesized and queued/playing, so this runs
                        # concurrently with the actual speech instead of
                        # delaying it. See _run_output_safety_check.
                        if self.guardrails_manager.is_enabled:
                            full_response_so_far = "".join(response_chunks)
                            if full_response_so_far:
                                turn_token = object()
                                self._active_turn_token = turn_token
                                threading.Thread(
                                    target=self._run_output_safety_check,
                                    args=(full_response_so_far, turn_token, continuous_vad, output_device_index),
                                    daemon=True,
                                ).start()

                        self.audio_player.stop_streaming(immediate=False)

                        # Add full response to history only if NOT interrupted
                        full_response = "".join(response_chunks)
                        if full_response:
                            self._last_bot_response = full_response
                            self.conversation_manager.add_assistant_message(full_response)
                            logger.info(f"🤖 Bot: {full_response}")
                    else:
                        # Interrupted — still store partial response for similarity filter
                        partial = "".join(response_chunks)
                        if partial:
                            self._last_bot_response = partial

                    latency["total_ms"] = round((time.perf_counter() - turn_start) * 1000, 1)
                    logger.info(
                        f"⏱️  LLM TTFT: {latency.get('llm_ttft_ms', '?')}ms | "
                        f"TTS first byte: {latency.get('tts_first_byte_ms', '?')}ms | "
                        f"Total: {latency['total_ms']}ms"
                    )
                    if self.current_session:
                        self.current_session.last_latency_breakdown = latency

                    # Reset idle timer because we just finished a turn
                    continuous_vad.reset_idle_timer()

                except Exception as turn_err:
                    logger.error(f"Error in speaker turn: {turn_err}")
                finally:
                    if self.ui_signals:
                        self.ui_signals.stop_talking.emit()
                    # ── ECHO GUARD: Signal playback ended + start cooldown ──
                    self._bot_is_speaking = False
                    self._playback_ended_time = time.time()
                    continuous_vad.set_playback_state(False)
                    self._barge_in_detected.clear()
                    
                    self._speaker_busy.clear()
                    self.audio_player.stop_streaming(immediate=True) # Ensure closed
                    
            logger.info("🔊 Speaker worker stopped cleanly")
        except Exception as e:
            logger.error(f"Speaker loop error: {e}")

    def _get_pulse_device_index(self, pa, direction: str = 'output') -> int:
        """Find the index of the 'pulse' audio device.

        PulseAudio (Linux/Pi) exposes a single bridging device that handles
        both directions, so a single index normally works for input and
        output alike. Platforms without PulseAudio (e.g. native Windows) have
        no such device, so the fallback must use the correctly-directioned
        env var instead of reusing the output device index for microphone
        input (which has no input channels and silently mis-selects a mic).
        """
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info and 'pulse' in info.get('name', '').lower():
                    logger.info(f"✅ Found PulseAudio device at index {i}: {info['name']}")
                    return i
        except Exception as e:
            logger.warning(f"Error searching for PulseAudio device: {e}")

        env_var = 'AUDIO_INPUT_DEVICE_INDEX' if direction == 'input' else 'AUDIO_OUTPUT_DEVICE_INDEX'
        fallback = int(os.getenv(env_var, 1))
        logger.warning(f"⚠️  PulseAudio not found - falling back to {env_var}={fallback} for {direction}")
        return fallback

    @staticmethod
    def _set_display_power(on: bool) -> None:
        """
        Toggle HDMI display power via vcgencmd (Raspberry Pi only).

        Ported as-is from the old cv2 FaceAnimator.enter_sleep()/wake_up() -
        silently no-ops off-Pi (FileNotFoundError when vcgencmd doesn't
        exist, e.g. this Windows dev machine), so it's safe to call
        unconditionally everywhere.
        """
        try:
            subprocess.run(
                ["vcgencmd", "display_power", "1" if on else "0"],
                check=False, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _enter_sleep(self) -> None:
        """Blank the display and turn off physical HDMI power (Pi only)."""
        self._display_sleeping = True
        self._set_display_power(False)
        if self.ui_signals:
            self.ui_signals.enter_sleep.emit()

    def _wake_up(self) -> None:
        """Restore the display and turn physical HDMI power back on (Pi only)."""
        self._display_sleeping = False
        self._set_display_power(True)
        if self.ui_signals:
            self.ui_signals.wake_up.emit()

    def _speak_system_message(self, phrase: str, continuous_vad, output_device_index: int) -> None:
        """
        Speak a fixed, non-LLM-generated phrase (check-in prompt, goodbye,
        etc). Reuses the same TTS/playback/echo-guard pattern as a normal
        turn, just with fixed text instead of an LLM-generated response.
        """
        if self._speaker_busy.is_set():
            # Speaker thread is mid-turn (race with the idle-timeout/end firing
            # right as a turn starts) - skip this cycle rather than step on it.
            return

        logger.info(f"🗣️  System message: '{phrase}'")

        try:
            self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
            self.audio_player.start_streaming(output_device_index=output_device_index)

            self._bot_is_speaking = True
            continuous_vad.set_playback_state(True)

            for audio_chunk in self.tts_client.synthesize_stream(iter([phrase])):
                if self.audio_player._is_playing:
                    self.audio_player.queue_audio(audio_chunk)
                else:
                    break

            self.audio_player.stop_streaming(immediate=False)
            self._last_bot_response = phrase
        except Exception as e:
            logger.error(f"Error speaking system message: {e}")
        finally:
            self._bot_is_speaking = False
            continuous_vad.set_playback_state(False)
            self.audio_player.stop_streaming(immediate=True)

    def _speak_check_in_prompt(self, continuous_vad, output_device_index: int) -> None:
        """Speak a short "are you still there?" prompt before ending an idle conversation."""
        phrase = os.getenv(
            'CHECK_IN_PROMPT',
            "Are you still there? Let me know if you'd like to keep going!"
        )
        self._speak_system_message(phrase, continuous_vad, output_device_index)

    def _speak_goodbye(self, continuous_vad, output_device_index: int) -> None:
        """
        Speak a farewell message when a conversation is about to end.

        If the conversation included real teaching (current_session.
        concepts_covered is non-empty - see _run_teaching_turn) and no
        explicit GOODBYE_MESSAGE override is configured, this is a genuine
        wrap-up naming what was covered and one real takeaway, not a
        generic farewell - see LLMClient.generate_session_wrapup(), which
        already fails toward a simple templated summary on its own, so no
        extra fallback handling is needed here. Falls back to the plain
        generic goodbye for a casual conversation that never got to any
        math.
        """
        override = os.getenv('GOODBYE_MESSAGE')
        concepts = self.current_session.concepts_covered if self.current_session else []

        if override:
            phrase = override
        elif concepts:
            phrase = self.llm_client.generate_session_wrapup(concepts)
        else:
            phrase = "Okay, talk to you later! Just say Hey Jarvis whenever you want to continue."

        self._speak_system_message(phrase, continuous_vad, output_device_index)

    def _speak_standalone(self, phrase: str, output_device_index: int) -> None:
        """
        Speak a fixed phrase with no active conversation (e.g. the
        going-to-sleep announcement). No echo-guard/continuous_vad needed
        here since wake-word detection uses its own model, not Azure STT.
        """
        logger.info(f"🗣️  {phrase}")
        try:
            self.audio_player.start_streaming(output_device_index=output_device_index)
            for audio_chunk in self.tts_client.synthesize_stream(iter([phrase])):
                if self.audio_player._is_playing:
                    self.audio_player.queue_audio(audio_chunk)
                else:
                    break
            self.audio_player.stop_streaming(immediate=False)
        except Exception as e:
            logger.error(f"Error speaking standalone message: {e}")
        finally:
            self.audio_player.stop_streaming(immediate=True)

    def _handle_wake_word(self):
        """Handle wake word detection - enter continuous conversation mode."""
        # Prevent concurrent interactions
        if not self._interaction_lock.acquire(blocking=False):
            logger.warning("Already in conversation, ignoring wake word")
            return
        
        try:
            logger.info("\n" + "="*60)
            logger.info("🎤 WAKE WORD DETECTED - CONVERSATION MODE ACTIVATED")
            logger.info("="*60)

            # Wake the screen back up immediately if it was asleep
            if self._display_sleeping:
                self._wake_up()

            # Stop wake word detection to free microphone
            self.wake_word_detector.stop()

            # New session for this conversation - a fresh session_id, turn
            # counter, and tool-call audit trail each time the wake word
            # fires (long-term memory lives in ProfileManager, not here).
            self.current_session = Session(profile_id=self.profile_manager.get_active_profile_id())
            logger.info(f"🆕 Session {self.current_session.session_id} started")

            # Flush any stale items (e.g. a leftover None sentinel from the previous
            # conversation's idle-timeout cleanup) so the new speaker thread starts clean.
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except queue.Empty:
                    break

            # Enter continuous conversation mode
            idle_timeout = int(os.getenv('CONVERSATION_IDLE_TIMEOUT_SECONDS', 10))
            check_in_grace = int(os.getenv('CHECK_IN_GRACE_SECONDS', 60))
            logger.info(
                f"⏱️  Will check in after {idle_timeout}s of silence, "
                f"then end after {check_in_grace}s more with no response"
            )

            # DYNAMICALLY FIND PULSE DEVICE (falls back to the correctly-directioned
            # env var per direction when no PulseAudio device exists, e.g. Windows)
            pulse_input_index = self._get_pulse_device_index(self.pa, direction='input')
            pulse_output_index = self._get_pulse_device_index(self.pa, direction='output')

            # Initialize VAD with dedicated INPUT stream. Its own hard idle-stop
            # must cover the check-in grace period too, otherwise it kills the
            # mic feed before we've finished waiting for a response.
            continuous_vad = ContinuousVADCapture(
                idle_timeout_seconds=idle_timeout + check_in_grace,
                pa=self.pa,
                input_device_index=pulse_input_index,
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
                args=(continuous_vad, pulse_output_index),
                name="SpeakerThread"
            )
            
            listener_thread.start()
            speaker_thread.start()

            logger.info("🚀 Full-Duplex engines started")

            # No real profile has been created on this robot yet - invite
            # account creation instead of silently using the placeholder
            # "Guest" profile as if it were a real person. Otherwise greet
            # the student right away instead of waiting on an LLM
            # round-trip for the first response - threads are already
            # running at this point, so barge-in still works if the
            # student starts talking over it.
            active_id = self.profile_manager.get_active_profile_id()
            if active_id is not None and self.profile_manager.is_default_profile(active_id):
                self._start_identity_checkin(continuous_vad, pulse_output_index)
            else:
                self._start_warmup_checkin(continuous_vad, pulse_output_index)

            # Wait for conversation to end (timeout or manual stop).
            # Two-stage idle handling: after `idle_timeout` of silence, ask if
            # the student is still there instead of ending immediately; only
            # end for real if there's no response within `check_in_grace`.
            checked_in = False
            prompt_finished_at = None
            while self._conversation_active.is_set():
                # Check for fatal errors in audio components and recover
                if self.audio_player.has_fatal_error:
                    logger.warning("♻️  FATAL AUDIO ERROR - Recreating system...")
                    self._recreate_audio_system()

                if not checked_in:
                    idle_duration = time.time() - continuous_vad.last_speech_time
                    if idle_duration >= idle_timeout:
                        logger.info(f"⏱️  {idle_timeout}s idle - checking if student is still there")
                        self._speak_check_in_prompt(continuous_vad, pulse_output_index)
                        checked_in = True
                        prompt_finished_at = time.time()
                else:
                    # last_speech_time keeps advancing on its own while the bot
                    # speaks the prompt (tracked as activity), so only count it
                    # as a response if it moves meaningfully PAST when the
                    # prompt actually finished playing.
                    if continuous_vad.last_speech_time > prompt_finished_at + 0.5:
                        logger.info("✅ Student responded to check-in - resuming conversation")
                        checked_in = False
                    elif time.time() - prompt_finished_at >= check_in_grace:
                        logger.info(f"⏱️  No response {check_in_grace}s after check-in - ending conversation")
                        self._speak_goodbye(continuous_vad, pulse_output_index)
                        self._conversation_active.clear()
                        break

                time.sleep(0.5)
            
            logger.info(f"\n👋 Conversation ended")
            # for audio_data in continuous_vad.listen_continuous():
            #     if audio_data is None:
            #         # Timeout reached
            #         break
                
            #     turn_count += 1
            #     logger.info(f"\n💬 Turn {turn_count}")
                
            #     # Process this turn
            #     self.end_time = time.time()
            #     if turn_count == 1:
            #         logger.info(f"⏱️ latency to start STT upon detection of wake word : {(self.end_time - self.start_time):.3f}s")
            #     # interrupted = self._process_turn(audio_data, continuous_vad)
            #     self._process_turn(audio_data, continuous_vad)
            #     # if interrupted:
            #     #     logger.info("🛑 Conversation interrupted by wake word")
            #     #     break
            
            # logger.info(f"\n👋 Conversation ended ({turn_count} turns)")
            # logger.info(f"📊 {self.conversation_manager}")
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        
        finally:
            # 1. SIGNAL WORKERS TO STOP
            self._conversation_active.clear()
            self._request_queue.put(None) # Sentinel for speaker
            
            # 2. CLEANUP WORKERS - ensure they stop before releasing PA
            try:
                if 'listener_thread' in locals() and listener_thread.is_alive():
                    listener_thread.join(timeout=1.0)
                if 'speaker_thread' in locals() and speaker_thread.is_alive():
                    speaker_thread.join(timeout=1.0)
            except:
                pass

            # 3. Explicitly cleanup continuous VAD to free microphone
            try:
                if 'continuous_vad' in locals():
                    continuous_vad.stop_background_monitoring()
                    continuous_vad.cleanup()
            except:
                pass
            
            # 4. Always restart wake word detection
            logger.info("▶️  Resuming wake word detection...")
            self._standby_since = time.time()
            self._restart_wake_word()
            self._interaction_lock.release()
    
    def _process_turn(self, audio_data: bytes, continuous_vad: ContinuousVADCapture) -> bool:
        """
        Process one turn of the conversation.
        
        Args:
            audio_data: Captured audio data
            continuous_vad: Continuous VAD instance (to reset idle timer)
            
        Returns:
            True if interrupted, False otherwise
        """
        try:
            # Step 1: Convert speech to text
            logger.info("☁️  Converting speech to text...")
            user_text = self.stt_client.recognize_from_audio_data(audio_data)
            
            if not user_text:
                logger.warning("No speech recognized")
                return False
            
            logger.info(f"📝 Student: {user_text}")
            
            # Step 2: Anonymize PII
            anonymized_text = self.privacy_manager.anonymize(user_text)
            
            # Step 3: Add to conversation history
            self.conversation_manager.add_user_message(anonymized_text)
            
            # Step 4: Generate LLM response with streaming
            logger.info("🧠 Generating response...")
            messages = self.conversation_manager.get_messages()
            
            # Start audio player streaming
            self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
            self.audio_player.start_streaming()
            
            # Collect response text chunks as they stream
            response_chunks = []
            try:
                # Create a wrapper that collects chunks while streaming
                def text_chunk_collector(llm_stream):
                    """Collect text chunks while passing them through."""

                    for chunk in llm_stream:
                        response_chunks.append(chunk)
                        if len(response_chunks) == 1:
                            logger.info(f"⏱️  LLM TTFT: {(time.time() - llm_start):.3f}s")

                        yield chunk
                
                # Stream LLM output through collector to TTS
                llm_start = time.time()
                llm_stream = self.llm_client.generate_response_stream(messages)

                # Now synthesize TTS from collected text
                tts_start = time.perf_counter()
                response_text_stream = text_chunk_collector(llm_stream)  # Iterator over chunks
                tts_stream = self.tts_client.synthesize_stream(response_text_stream)

                # Stream audio to player
                for audio_chunk in tts_stream:
                    self.audio_player.queue_audio(audio_chunk)
                
                # Wait for playback to complete
                self.audio_player.stop_streaming()
                
                # Combine collected chunks into full response
                response_text = ''.join(response_chunks)
                logger.info(f"🤖 Jarvis: {response_text}")
                
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)
                
                # Reset idle timer after our response
                continuous_vad.reset_idle_timer()
                
            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
                self.audio_player.stop_streaming()
                raise
            
            return False  # No interruption possible in continuous mode
        
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return False
    
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
            if self.ui_signals:
                self.ui_signals.thinking.emit()
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
                if self.ui_signals:
                    self.ui_signals.start_talking.emit()
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
                logger.info(f"🤖 Chippy: {response_text}")
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)
                
                # Reset idle timer AFTER bot finishes speaking
                # This ensures we don't timeout while bot is generating/speaking
                continuous_vad.reset_idle_timer()
                
                if self.ui_signals:
                    self.ui_signals.listening.emit()
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
    
    def run(self):
        """Start Jarvis and run the main loop."""
        # MainWindow.__init__ already puts the face in idle state on construction,
        # and show_idle_mode() there is only reachable via signals from this
        # thread once running - nothing to emit here at startup.
        logger.info("\n" + "🤖 "*20)
        logger.info("Jarvis TUTORING ROBOT STARTED")
        logger.info("🤖 "*20 + "\n")
        logger.info("Listening for wake word: 'Hey Jarvis'")
        logger.info("Press Ctrl+C to stop\n")
        
        self._is_running = True

        try:
            # First blocking wake word listen. When a wake word fires, _handle_wake_word()
            # is called inline, stops the detector, runs the conversation, then calls
            # _restart_wake_word() which takes over in a daemon thread. After that,
            # start() returns here — but we must NOT call stop() at that point because
            # the daemon thread and all shared resources (self.pa, TTS) are still needed.
            self.wake_word_detector.start(self._handle_wake_word)

            # Keep run() alive while wake-word/conversation cycles continue in daemon threads.
            # Only exit when _is_running is cleared by stop() or a signal arrives.
            display_sleep_timeout = int(os.getenv('DISPLAY_SLEEP_TIMEOUT', 0))
            while self._is_running:
                # After enough standby idle time (no conversation, no new wake
                # word), announce it and put the physical display to sleep.
                if (not self._display_sleeping
                        and display_sleep_timeout > 0
                        and not self._conversation_active.is_set()
                        and (time.time() - self._standby_since) >= display_sleep_timeout):
                    logger.info(f"💤 {display_sleep_timeout}s idle - going to sleep")
                    sleep_output_index = self._get_pulse_device_index(self.pa, direction='output')
                    self._speak_standalone(
                        os.getenv('SLEEP_MESSAGE', "I'm going to sleep now. Just say Hey Jarvis to wake me up!"),
                        sleep_output_index
                    )
                    self._enter_sleep()

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
    import sys

    is_windows = sys.platform == "win32"

    # ── Display configuration ──────────────────────────────────────────────────
    # FACE_ENABLED=true          → show the face/teaching-canvas window
    # FACE_ENABLED=false         → headless, no GUI (default when DISPLAY not set)
    # unset                      → auto: on Windows, on (normal desktop session,
    #                              Qt opens its own native window, no DISPLAY
    #                              needed); on Linux/Pi, on only if DISPLAY/
    #                              WAYLAND_DISPLAY is already set (matches a
    #                              headless-SSH-session default)
    #
    # DISPLAY_BACKEND=physical   → HDMI monitor          (DISPLAY=:0)
    # DISPLAY_BACKEND=vnc        → TigerVNC session      (DISPLAY=:1)
    # DISPLAY_BACKEND=auto       → pick first available X11 socket (:1 then :0)
    #
    # You can also skip DISPLAY_BACKEND and set DISPLAY directly, e.g. DISPLAY=:0
    #
    # None of the above (DISPLAY/XAUTHORITY/X11 sockets) is a Linux/X11 concept
    # that applies on native Windows — Qt opens a normal Win32 window directly
    # there, so Windows gets its own simpler default/path below. On Linux, Qt
    # still needs a real X11 DISPLAY the same way the old cv2-based face did.
    # ──────────────────────────────────────────────────────────────────────────
    face_env = os.getenv("FACE_ENABLED", "").strip().lower()
    if face_env in ("true", "1", "yes"):
        face_enabled = True
    elif face_env in ("false", "0", "no"):
        face_enabled = False
    elif is_windows:
        # No DISPLAY-style signal exists on Windows to auto-detect from -
        # a normal interactive session always has a GUI available.
        face_enabled = True
    else:
        # Linux/Pi auto: enable face only when DISPLAY is already set
        face_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if face_enabled and is_windows:
        logger.info("Face animation enabled (native Windows GUI, no X11 needed)")
    elif face_enabled:
        # DISPLAY_BACKEND always wins when explicitly set — it overrides whatever
        # DISPLAY was loaded from .env so that a single knob controls the display.
        # Priority: DISPLAY_BACKEND (explicit) > DISPLAY (env/shell) > default :0
        backend = os.getenv("DISPLAY_BACKEND", "").strip().lower()

        if backend == "vnc":
            os.environ["DISPLAY"] = ":1"
        elif backend == "physical":
            os.environ["DISPLAY"] = ":0"
        elif backend == "auto":
            # Pick the first X11 socket that actually exists
            for candidate in (":1", ":0"):
                if os.path.exists(f"/tmp/.X11-unix/X{candidate[1:]}"):
                    os.environ["DISPLAY"] = candidate
                    break
            else:
                os.environ["DISPLAY"] = ":0"
        elif not os.environ.get("DISPLAY"):
            # No DISPLAY_BACKEND and no DISPLAY — last resort default
            os.environ["DISPLAY"] = ":0"

        # Ensure X authentication is available.
        # TigerVNC stores its cookie in the same ~/.Xauthority file as the
        # physical display, just under a different display entry (:1 vs :0).
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

    app = None
    ui_signals = None
    if face_enabled:
        try:
            from PyQt5.QtWidgets import QApplication
            # QApplication must exist before any other Qt object (UISignals,
            # MainWindow) is constructed, and must live on the main thread.
            app = QApplication(sys.argv)
            ui_signals = UISignals()

            # Resolve relative to this file's own location, not the process's
            # CWD - "./visuals/faces" only works if launched as `cd src &&
            # python main.py`, but breaks under `python3 src/main.py` from
            # the repo root (the convention actually used to run this on the
            # Pi), silently falling back to headless instead of erroring loudly.
            faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visuals", "faces")
            window = MainWindow(ui_signals, faces_dir, fullscreen=not is_windows)
            window.show()

            logger.info(
                "Face animation enabled (native Windows GUI)" if is_windows
                else f"Face animation enabled on display {os.environ.get('DISPLAY', '')}"
            )
        except Exception as e:
            logger.warning(f"Failed to init face animation: {e}. Falling back to headless.")
            app = None
            ui_signals = None

    jarvis = JarvisBot(ui_signals)

    if app:
        worker = threading.Thread(target=jarvis.run, daemon=True)
        worker.start()
        # Qt's event loop must run on the main thread
        sys.exit(app.exec_())
    else:
        jarvis.run()

if __name__ == "__main__":
    main()
