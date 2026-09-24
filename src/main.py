import os
# Allow the OS to use its default display and QT backend, rather than hardcoding.

import re
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

    # Asked in order right after a brand new profile is created.
    ONBOARDING_QUESTIONS = [
        "What grade are you in?",
        "What's your favorite subject to learn about?",
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

        # ── Current conversation session (see session.py) ──
        # None outside of an active conversation; a fresh Session is created
        # each time the wake word fires.
        self.current_session: Optional[Session] = None

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

    def _speak_fixed_phrase(self, phrase: str, continuous_vad, output_device_index: int) -> None:
        """
        Speak a fixed, non-LLM-generated phrase (profile switch/creation
        confirmations). Same TTS/playback/echo-guard pattern as a normal
        turn in _speaker_loop, just with fixed text.
        """
        logger.info(f"🗣️  {phrase}")
        try:
            self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
            if self.ui_signals:
                self.ui_signals.start_talking.emit()
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
            logger.error(f"Error speaking fixed phrase: {e}")
        finally:
            if self.ui_signals:
                self.ui_signals.stop_talking.emit()
            self._bot_is_speaking = False
            self._playback_ended_time = time.time()
            continuous_vad.set_playback_state(False)
            self.audio_player.stop_streaming(immediate=True)

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
                name = self.profile_manager.get_profile_name(self.profile_manager.get_active_profile_id())
                self._speak_fixed_phrase(f"Okay, switching back to {name}'s session.", continuous_vad, output_device_index)
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

        # question_index == 1 means this answer is responding to
        # ONBOARDING_QUESTIONS[0] ("What grade are you in?") - parse and
        # store it so RAG retrieval (search_curriculum tool) can scope
        # results to this student's actual grade level.
        if next_index == 1:
            grade = normalize_grade(user_text)
            if grade:
                self.profile_manager.set_grade(self._pending_onboarding["profile_id"], grade)
                logger.info(f"📓 Stored grade '{grade}' for profile {self._pending_onboarding['profile_id']}")
            else:
                logger.info(f"📓 Could not parse a grade from: '{user_text}'")

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
            "area", "perimeter", "volume", "surface area",
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

        return bool(re.search(r"\d", t) and re.search(r"[\+\-\*/=]", t))

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

            plan = self.llm_client.generate_teaching_plan(messages)
            visuals = plan.get("visuals", [])

            for step in plan.get("speech", []):
                if self._interruption_event.is_set():
                    break

                step_text = step.get("text", "").strip()
                if not step_text:
                    continue

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

            if not self._interruption_event.is_set():
                self.audio_player.stop_streaming(immediate=False)
                full_response = " ".join(response_parts)
                if full_response:
                    self._last_bot_response = full_response
                    self.conversation_manager.add_assistant_message(full_response)
                    logger.info(f"🤖 Bot (teaching): {full_response}")
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

                    # ── ECHO GUARD: Signal that bot is now speaking ──
                    self._bot_is_speaking = True
                    continuous_vad.set_playback_state(True)

                    response_chunks = []
                    latency: Dict[str, float] = {}

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
                        messages, tools=[CURRICULUM_SEARCH_TOOL, VOLUME_CONTROL_TOOL], tool_executor=self._execute_tool
                    )
                    collected_stream = text_collector(llm_stream)
                    tts_stream = audio_timing(self.tts_client.synthesize_stream(collected_stream))

                    for audio_chunk in tts_stream:
                        if self._interruption_event.is_set():
                            logger.warning("🛑 Speaker aborted due to interruption event")
                            break

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

    def _get_pulse_device_index(self, pa) -> int:
        """Find the index of the 'pulse' audio device."""
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info and 'pulse' in info.get('name', '').lower():
                    logger.info(f"✅ Found PulseAudio device at index {i}: {info['name']}")
                    return i
        except Exception as e:
            logger.warning(f"Error searching for PulseAudio device: {e}")
        
        # Fallback to env var or default 1 (but log warning)
        fallback = int(os.getenv('AUDIO_OUTPUT_DEVICE_INDEX', 1))
        logger.warning(f"⚠️  PulseAudio not found - falling back to index {fallback}")
        return fallback

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
            logger.info(f"⏱️  Will end after 10 seconds of silence")
            
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
            
            # DYNAMICALLY FIND PULSE DEVICE
            pulse_index = self._get_pulse_device_index(self.pa)
            
            # Initialize VAD with dedicated INPUT stream
            continuous_vad = ContinuousVADCapture(
                idle_timeout_seconds=idle_timeout,
                pa=self.pa,
                input_device_index=pulse_index,
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

            # No real profile has been created on this robot yet - invite
            # account creation instead of silently using the placeholder
            # "Guest" profile as if it were a real person.
            active_id = self.profile_manager.get_active_profile_id()
            if active_id is not None and self.profile_manager.is_default_profile(active_id):
                self._speak_fixed_phrase(
                    "Please create your account before you start learning! "
                    "Just say Voice Recognition, then say your name, and I'll remember you from now on.",
                    continuous_vad, pulse_index
                )

            # Wait for conversation to end (timeout or manual stop)
            while self._conversation_active.is_set():
                # Check for fatal errors in audio components and recover
                if self.audio_player.has_fatal_error:
                    logger.warning("♻️  FATAL AUDIO ERROR - Recreating system...")
                    self._recreate_audio_system()
                
                # Check if idle timeout exceeded
                idle_duration = time.time() - continuous_vad.last_speech_time
                if idle_duration >= idle_timeout:
                    logger.info(f"⏱️  {idle_timeout}s idle timeout - ending")
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
                logger.info(f"🤖 Chippy: {response_text}")
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
    # FACE_ENABLED=true   → show the face/teaching-canvas window
    # FACE_ENABLED=false  → headless, no GUI
    # unset               → auto: on Windows, on (normal desktop session,
    #                       Qt opens its own native window, no DISPLAY needed);
    #                       on Linux/Pi, on only if DISPLAY/WAYLAND_DISPLAY is
    #                       already set (matches a headless-SSH-session default)
    # ──────────────────────────────────────────────────────────────────────────
    face_env = os.getenv("FACE_ENABLED", "").strip().lower()
    if face_env in ("true", "1", "yes"):
        face_enabled = True
    elif face_env in ("false", "0", "no"):
        face_enabled = False
    elif is_windows:
        face_enabled = True
    else:
        face_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

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
            # CWD - same reasoning as the SYSTEM_PROMPT_PATH/face-path fixes
            # elsewhere in this codebase (CWD depends on how you launch main.py).
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
