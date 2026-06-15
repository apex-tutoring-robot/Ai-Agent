import os
# Allow the OS to use its default display and QT backend, rather than hardcoding.

import threading
import time
import queue
from queue import Queue
from typing import Optional
from dotenv import load_dotenv
from PyQt5.QtWidgets import QApplication
import sys

from visuals.ui.main_window import MainWindow
from visuals.ui.ui_signals import UISignals

from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient, SynthesisBlockedError
from conversation.state_manager import ConversationStateManager
from privacy.privacy_manager import PrivacyManager
from vision.camera import Camera
from guardrails.guardrails_manager import GuardrailsManager
from user_profile import get_user_id, get_user_name, set_user_name
import logging

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

    def __init__(self, ui_signals=None):
        """Initialize Jarvis with all components."""
        logger.info("Initializing Jarvis...")
        self.ui_signals = ui_signals

        self.user_id = get_user_id()
        self._awaiting_name: bool = False

        # Initialize shared PyAudio instance
        import pyaudio
        self.pa = pyaudio.PyAudio()

        # Initialize components with shared PyAudio
        input_idx = os.getenv("AUDIO_INPUT_DEVICE_INDEX")
        input_idx = int(input_idx) if input_idx not in (None, "") else None

        self.wake_word_detector = WakeWordDetector(
            pa=self.pa,
            input_device_index=input_idx
        )
        self.audio_player = AudioPlayer(
            pa=self.pa,
            on_level=(lambda level: self.ui_signals.mouth_level.emit(level)) if self.ui_signals else None
        )
        self.stt_client = SpeechToTextClient(user_id=self.user_id)
        self.llm_client = LLMClient(user_id=self.user_id)
        user_name = get_user_name()
        if user_name:
            self._apply_user_name_to_prompt(user_name)
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
            import pyaudio
            self.pa = pyaudio.PyAudio()

            # Re-init components with shared callbacks
            self.continuous_vad = ContinuousVADCapture(pa=self.pa)
            self.audio_player = AudioPlayer(
                pa=self.pa,
            )
            self.wake_word_detector = WakeWordDetector(pa=self.pa)

            logger.info("✅ Audio system fully recreated and re-wired")
        except Exception as e:
            logger.error(f"Failed to recreate audio system: {e}")

    # ── Display power management ───────────────────────────────────────────────

    def _set_display_power(self, on: bool) -> None:
        if self.ui_signals is None:
            return
        os.system(f"vcgencmd display_power {'1' if on else '0'}")
        self._display_on = on
        logger.info(f"🖥️  Display power {'on' if on else 'off'}")

    def _on_display_sleep(self) -> None:
        logger.info("💤 Display sleep timeout — turning off display")
        self._set_display_power(False)

    def _schedule_display_sleep(self) -> None:
        if self.ui_signals is None:
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
        return len(overlap) / min(len(words_a), len(words_b))

    def run_teaching_plan(self, plan, output_device_index, continuous_vad):
        import time

        speech_steps = plan.get("speech", [])
        visual_steps = plan.get("visuals", [])

        def visuals_for(step_id):
            return [v for v in visual_steps if v.get("speech_id") == step_id]

        self.audio_player.start_streaming(output_device_index=output_device_index)
        logger.info("DEBUG: run_teaching_plan started")

        try:
            if self.ui_signals:
                self.ui_signals.start_talking.emit()

            for step in speech_steps:
                if self._interruption_event.is_set():
                    break

                step_id = step.get("id")
                step_text = step.get("text", "").strip()

                if not step_text:
                    continue

                actions = visuals_for(step["id"])

                if self.ui_signals and actions:
                    logger.info(f"DEBUG: emitting draw actions for step {step_id}: {actions}")
                    self.ui_signals.draw_actions.emit(actions)

                # time.sleep(0.15)

                audio_bytes = self.tts_client.synthesize_to_audio(step_text)
                if audio_bytes and self.audio_player._is_playing:
                    self.audio_player.queue_audio(audio_bytes)

            self.audio_player.stop_streaming(immediate=False)

        finally:
            if self.ui_signals:
                self.ui_signals.stop_talking.emit()

    def is_math_query(self, text: str) -> bool:
        import re

        t = text.lower().strip()

        math_keywords = [
            "solve", "equations", "add", "subtract", "multiply", "divide", "area", "perimeter", "radius", "diameter", "rectangle", "circle", "triangle", "fraction", "algebra", "geometry", "graph"
        ]

        if any(word in t for word in math_keywords):
            return True

        return bool(re.search(r"\d", t) and re.search(r"[\+\-\*/=]", t))

    _CAMERA_TRIGGERS = {
        "scan", "photo", "picture", "camera",
        "take a photo", "take a picture", "scan my homework",
        "look at this", "check this", "show you",
    }

    @staticmethod
    def _is_camera_trigger(text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in JarvisBot._CAMERA_TRIGGERS)

    def force_shape_if_missing(self, plan, user_text):
        import re
        text = user_text.lower()
        visuals = plan.get("visuals", [])

        shape_map = {
            "triangle": 3,
            "pentagon": 5,
            "hexagon": 6,
            "heptagon": 7,
            "octagon": 8,
            "nonagon": 9,
            "decagon": 10,
        }

        sides = None
        for word, n in shape_map.items():
            if word in text:
                sides = n
                break

        match = re.search(r"(\d+)\s*[- ]?sided|(\d+)\s+sides", text)
        if match:
            sides = int(match.group(1) or match.group(2))

        if not sides:
            plan["visuals"] = visuals
            return plan

        visuals = [
            v for v in visuals
            if v.get("action") not in ["draw_regular_polygon", "draw_polygon"]
        ]

        speech = plan.get("speech", [])
        speech_id = speech[0].get("id", 1) if speech else 1

        has_clear = any(v.get("action") == "clear" for v in visuals)
        if not has_clear:
            visuals.insert(0, {"speech_id": speech_id, "action": "clear"})

            visuals.insert(1, {
                "speech_id": speech_id,
                "action": "draw_regular_polygon",
                "sides": max(3, min(12, sides)),
                "cx": 700,
                "cy": 270,
                "radius": 125
            })

            visuals.insert(2, {
                "speech_id": speech_id,
                "action": "draw_text",
                "text": f"{sides}-sided shape",
                "sides": max(3, min(12, sides)),
                "x": 610,
                "y": 430
            })

        plan["visuals"] = visuals
        return plan

    def _apply_user_name_to_prompt(self, name: str) -> None:
        self.llm_client.system_prompt += (
            f"\n\nThe student's name is {name}. "
            "Use their name occasionally to make the interaction feel personal."
        )

    @staticmethod
    def _extract_name(text: str) -> str:
        """Extract a first name from phrases like 'My name is John' or 'I'm John'."""
        t = text.strip()
        for prefix in ("my name is ", "i'm ", "i am ", "it's ", "its ", "call me ", "name is "):
            if t.lower().startswith(prefix):
                t = t[len(prefix):]
                break
        words = t.split()[:2]
        return " ".join(w.capitalize() for w in words) if words else text.strip().title()

    def _listener_loop(self, continuous_vad, request_queue):
        """Producer: Always listening, detecting interruptions in real-time.

        Three-layer echo defense:
          L1 – Discard ALL recognitions while bot is speaking (unless barge-in detected)
          L2 – Text similarity filter rejects echo transcriptions that match bot's last response
          L3 – Echo cooldown after playback ends (500ms grace period)
        """
        logger.info("📡 Listener worker started")
        ECHO_COOLDOWN_S = float(os.getenv('ECHO_COOLDOWN_S', 0.5))
        SIMILARITY_THRESHOLD = float(os.getenv('ECHO_SIMILARITY_THRESHOLD', 0.55))
        SIMILARITY_WINDOW_S = float(os.getenv('ECHO_SIMILARITY_WINDOW_S', 3.0))

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
                    if self.ui_signals:
                        self.ui_signals.listening.emit()
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
                    final_text = None

                    # ── Greeting / name-collection (bypass normal pipeline) ────
                    if user_text == "__GREET__":
                        name = get_user_name()
                        if name:
                            final_text = f"Welcome back, {name}! What would you like to learn today?"
                        else:
                            self._awaiting_name = True
                            final_text = "Hi there! I am Jarvis, your personal tutor. What is your name?"
                    elif self._awaiting_name:
                        if was_barge_in:
                            # Echo triggered while bot was asking for the name — discard.
                            logger.info(f"🔇 Ignoring barge-in during name prompt: '{user_text}'")
                            continuous_vad.reset_idle_timer()
                            continue
                        name = self._extract_name(user_text)
                        set_user_name(name)
                        self._apply_user_name_to_prompt(name)
                        self._awaiting_name = False
                        final_text = f"Nice to meet you, {name}! I am here to help you learn. What would you like to study today?"
                    else:
                        # ── Camera Vision ──────────────────────────────────────
                        if self._is_camera_trigger(user_text):
                            try:
                                saved_path = self.camera.capture_and_save()
                                logger.info(f"📷 Image saved to {saved_path}")

                                import base64 as _b64
                                with open(saved_path, "rb") as _f:
                                    data_url = f"data:image/jpeg;base64,{_b64.b64encode(_f.read()).decode()}"

                                try:
                                    extracted = self.llm_client.extract_image_content(data_url)
                                    continuous_vad.reset_idle_timer()  # extraction can take 5-10s
                                    combined = f"{user_text}\n\n[Scanned homework content:\n{extracted}]"
                                    anonymized_text = self.privacy_manager.anonymize(combined)
                                    logger.info("📷 Image extracted and stored as text")
                                except Exception as extract_err:
                                    logger.error(f"📷 Image extraction failed: {extract_err}")
                                    error_audio = self.tts_client.synthesize_to_audio(
                                        "Sorry, I had trouble reading the image. Please try again."
                                    )
                                    self.audio_player.start_streaming(output_device_index=output_device_index)
                                    self.audio_player.queue_audio(error_audio)
                                    self.audio_player.stop_streaming(immediate=False)
                                    continue
                            except Exception as cam_err:
                                logger.error(f"📷 Camera failed to capture: {cam_err} — falling back to text-only")
                                combined = f"{user_text}\n\nCamera is not working, please talk to me!"
                                anonymized_text = self.privacy_manager.anonymize(combined)
                        else:
                            anonymized_text = self.privacy_manager.anonymize(user_text)
                        # ──────────────────────────────────────────────────────

                    # ── Guardrails + LLM (skipped for greeting/name turns) ────
                    if final_text is None:
                        allowed, refusal_msg = self.guardrails.check_input(anonymized_text)
                        continuous_vad.reset_idle_timer()  # guardrails LLM call can take 5-10s

                        if allowed:
                            self.conversation_manager.add_user_message(anonymized_text)
                            messages = self.conversation_manager.get_messages()

                            # ── Teaching Plan Path (math/visual queries) ──────
                            if self.is_math_query(user_text):
                                logger.info("DEBUG: math query detected - generating teaching plan")
                                if self.ui_signals:
                                    self.ui_signals.thinking.emit()
                                try:
                                    plan = self.llm_client.generate_teaching_plan(messages)
                                    plan = self.force_shape_if_missing(plan, user_text)
                                    if self.ui_signals:
                                        self.ui_signals.show_teaching_layout.emit()
                                    self.run_teaching_plan(plan, output_device_index, continuous_vad)
                                    continuous_vad.reset_idle_timer()
                                    continue
                                except Exception as e:
                                    logger.error(f"Teaching plan failed, falling back to normal response: {e}")
                                    # Fall through to regular LLM path

                            # ── Regular LLM Path (streaming LLM → TTS) ────────
                            # synthesize_stream handles sentence detection and calls
                            # check_output_fast before each sentence is synthesized.
                            # LLM generation and TTS synthesis overlap sentence-by-sentence.
                            if self.ui_signals:
                                self.ui_signals.thinking.emit()

                            response_chunks = []

                            def _llm_stream():
                                for chunk in self.llm_client.generate_response_stream(messages):
                                    if self._interruption_event.is_set():
                                        break
                                    response_chunks.append(chunk)
                                    yield chunk

                            self.audio_player.start_streaming(output_device_index=output_device_index)
                            self._bot_is_speaking = True
                            continuous_vad.set_playback_state(True)

                            blocked_refusal = None
                            first_audio = True
                            try:
                                for audio_chunk in self.tts_client.synthesize_stream(
                                    _llm_stream(),
                                    pre_synthesis_check=self.guardrails.check_output_fast
                                ):
                                    if self._interruption_event.is_set():
                                        logger.warning("🛑 Speaker aborted due to interruption event")
                                        break
                                    if first_audio:
                                        if self.ui_signals:
                                            self.ui_signals.start_talking.emit()
                                        first_audio = False
                                    if self.audio_player._is_playing:
                                        try:
                                            self.audio_player.queue_audio(audio_chunk)
                                        except RuntimeError as e:
                                            logger.warning(f"⚠️ Playback queueing failed: {e}")
                                            break
                                    else:
                                        break
                            except SynthesisBlockedError as e:
                                blocked_refusal = e.refusal_text

                            if blocked_refusal or self._interruption_event.is_set():
                                self.audio_player.stop_streaming(immediate=True)
                            else:
                                self.audio_player.stop_streaming(immediate=False)

                            full_response = "".join(response_chunks)

                            if blocked_refusal:
                                logger.warning("🛡️ Output check blocked LLM response — speaking refusal")
                                refusal_audio = self.tts_client.synthesize_to_audio(blocked_refusal)
                                if refusal_audio:
                                    self.audio_player.start_streaming(output_device_index=output_device_index)
                                    self._bot_is_speaking = True
                                    continuous_vad.set_playback_state(True)
                                    if self.ui_signals:
                                        self.ui_signals.start_talking.emit()
                                    self.audio_player.queue_audio(refusal_audio)
                                    self.audio_player.stop_streaming(immediate=False)
                                self._last_bot_response = blocked_refusal
                            elif full_response and not self._interruption_event.is_set():
                                self.conversation_manager.add_assistant_message(full_response)
                                self._last_bot_response = full_response
                                logger.info(f"Bot: {full_response}")
                            elif full_response:
                                self._last_bot_response = full_response  # echo detection only

                            continuous_vad.reset_idle_timer()
                            continue  # finally block still runs for echo-guard cleanup
                        else:
                            logger.info("🛡️ Guardrails blocked input — speaking refusal")
                            final_text = refusal_msg

                    if not final_text:
                        continuous_vad.reset_idle_timer()
                        continue

                    # ── TTS + Playback ────────────────────────────────────────
                    if self.ui_signals:
                        self.ui_signals.start_talking.emit()
                    self.audio_player.start_streaming(output_device_index=output_device_index)

                    # ── ECHO GUARD: Signal that bot is now speaking ──
                    self._bot_is_speaking = True
                    continuous_vad.set_playback_state(True)

                    # Stream final_text through TTS to audio player
                    tts_stream = self.tts_client.synthesize_stream(iter([final_text]))

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
                    self.audio_player.stop_streaming(immediate=True)  # Ensure closed

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
        """Return the PyAudio input device index to use for the VAD mic stream.

        When PipeWire AEC is active (pulse device found), input goes through the
        pulse device so PipeWire routes it to echo-cancel-source.  If the user has
        set AUDIO_INPUT_DEVICE_INDEX explicitly AND pulse is unavailable, honour
        that env var as a direct hw fallback.
        """
        env_idx = os.getenv('AUDIO_INPUT_DEVICE_INDEX')
        if pulse_index is not None:
            # Pulse device found — always use it for input so AEC is in the path.
            logger.info(f"🎤 Input routed through PipeWire (device {pulse_index}) — AEC active")
            return pulse_index
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

            # Greet by name if known, otherwise ask for it
            self._request_queue.put("__GREET__")

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
            self._awaiting_name = False    # Reset if conversation timed out mid-onboarding

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

            if self.ui_signals:
                self.ui_signals.show_face_fullscreen.emit()

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
                logger.info(f"Chippy: {response_text}")
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)

                # Reset idle timer AFTER bot finishes speaking
                # This ensures we don't timeout while bot is generating/speaking
                continuous_vad.reset_idle_timer()

                if self.ui_signals:
                    self.ui_signals.stop_talking.emit()
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
        if self.ui_signals:
            self.ui_signals.stop_talking.emit()
        logger.info("\n" + "🤖 "*20)
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
    # FACE_ENABLED=true/false to force; otherwise auto-detect via Wayland socket.
    # On Raspberry Pi OS with labwc the socket is at /run/user/<uid>/wayland-0.
    # wayvnc mirrors this Wayland session to VNC viewers — no separate X11 needed.
    face_env = os.getenv("FACE_ENABLED", "").strip().lower()
    if face_env in ("true", "1", "yes"):
        face_enabled = True
    elif face_env in ("false", "0", "no"):
        face_enabled = False
    else:
        uid = os.getuid()
        face_enabled = any(
            os.path.exists(f"/run/user/{uid}/{wl}")
            for wl in ("wayland-0", "wayland-1")
        )

    if face_enabled:
        if not os.environ.get("WAYLAND_DISPLAY"):
            uid = os.getuid()
            for _wl in ("wayland-0", "wayland-1"):
                if os.path.exists(f"/run/user/{uid}/{_wl}"):
                    os.environ["WAYLAND_DISPLAY"] = _wl
                    break

        if os.environ.get("WAYLAND_DISPLAY"):
            if not os.environ.get("QT_QPA_PLATFORM"):
                os.environ["QT_QPA_PLATFORM"] = "wayland"
            logger.info(f"Running UI on Wayland ({os.environ['WAYLAND_DISPLAY']})")
        else:
            logger.warning("Wayland compositor not found, switching to headless")
            face_enabled = False

    if not face_enabled:
        logger.info("Running headless mode")

    if face_enabled:
        app = QApplication(sys.argv)
        ui_signals = UISignals()
        window = MainWindow(ui_signals)
        window.show()
        jarvis = JarvisBot(ui_signals=ui_signals)
        worker = threading.Thread(target=jarvis.run, daemon=True)
        worker.start()
        sys.exit(app.exec_())
    else:
        jarvis = JarvisBot(ui_signals=None)
        jarvis.run()

if __name__ == "__main__":
    main()
