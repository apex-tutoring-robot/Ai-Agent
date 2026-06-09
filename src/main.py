import os

# ── Qt / cv2 plugin conflict fix ──────────────────────────────────────────────
# cv2 (OpenCV) ships its own bundled Qt and registers its plugin path via
# QT_QPA_PLATFORM_PLUGIN_PATH. This causes PyQt5 to load cv2's incompatible
# xcb plugin instead of the system one, aborting at startup on Raspberry Pi.
# We must clear / override these env vars BEFORE any cv2 or Qt import occurs.
import sys as _sys

# 1. Point Qt at the system PyQt5 plugins, not cv2's bundled ones.
import sysconfig as _sc
_pyqt5_plugins = os.path.join(
    _sc.get_path("platlib"), "PyQt5", "Qt5", "plugins"
)
if os.path.isdir(_pyqt5_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _pyqt5_plugins

# 2. On a physical display with no compositor, eglfs or linuxfb are more
#    stable than xcb on Pi 4/5.  Let xcb stay as the default when DISPLAY
#    is set (X11 session), otherwise force eglfs for direct framebuffer.
#    You can override this with QT_QPA_PLATFORM=xcb in your .env if needed.
if not os.environ.get("QT_QPA_PLATFORM"):
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    else:
        os.environ["QT_QPA_PLATFORM"] = "eglfs"

# 3. Suppress Qt debug noise that pollutes logs.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
# ──────────────────────────────────────────────────────────────────────────────

# Allow the OS to use its default display and QT backend, rather than hardcoding.

import threading
import time
import queue
from queue import Queue
from typing import Optional
from difflib import SequenceMatcher
from dotenv import load_dotenv
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

from visuals.ui.main_window import MainWindow
from visuals.ui.ui_signals import UISignals

from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from conversation.state_manager import ConversationStateManager
from privacy.privacy_manager import PrivacyManager
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JarvisBot:
    """Main orchestrator for Jarvis tutoring robot."""
    
    def __init__(self, ui_signals=None):
        """Initialize Jarvis with all components."""
        logger.info("Initializing Jarvis...")
        self.ui_signals = ui_signals        
        
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
        self.stt_client = SpeechToTextClient()
        self.llm_client = LLMClient()
        self.tts_client = TextToSpeechClient()
        self.privacy_manager = PrivacyManager()
        self.conversation_manager = ConversationStateManager(
            max_history=int(os.getenv('MAX_CONVERSATION_HISTORY', 20))
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

                time.sleep(0.15)

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
            # arithmetic / algebra
            "solve", "equation", "equations", "add", "subtract", "multiply", "divide",
            "fraction", "algebra", "calculate", "compute", "simplify", "evaluate",
            "factor", "expand", "expression", "variable", "coefficient",
            # geometry — shapes
            "area", "perimeter", "volume", "surface area",
            "radius", "diameter", "circumference",
            "rectangle", "square", "circle", "triangle", "polygon",
            "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
            "trapezoid", "trapezium", "parallelogram", "rhombus", "kite",
            "ellipse", "oval", "sector", "segment",
            "sided", "sides", "shape", "diagonal", "hypotenuse",
            # geometry — measurements
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
                    if self.ui_signals:
                        self.ui_signals.listening.emit()
                    request_queue.put(text)
                    
            logger.info("📡 Listener worker stopped cleanly")
        except Exception as e:
            logger.error(f"Listener error: {e}")
            self._conversation_active.clear()
            
    def force_shape_if_missing(self, plan, user_text):
        """
        Guarantee that a shape diagram exists in the visual plan.

        Strategy:
          1. Detect what shape the user asked about (by name or N-sided).
          2. Check whether the LLM already drew that shape type.
          3. Only inject a fallback shape when the LLM forgot it entirely.
          4. Never strip visuals that the LLM correctly generated — only add missing ones.
          5. Always ensure labels go OUTSIDE the shape (above/below, never on edges).
        """
        import re, math

        text = user_text.lower()
        visuals = plan.get("visuals", [])
        speech  = plan.get("speech", [])
        speech_id = speech[0].get("id", 1) if speech else 1

        # ── Map shape names → sides (for regular polygons) ──
        poly_map = {
            "triangle": 3, "pentagon": 5, "hexagon": 6,
            "heptagon": 7, "octagon": 8, "nonagon": 9, "decagon": 10,
        }

        # ── Map shape names → draw action we expect ──
        draw_action_for = {
            "rectangle": "draw_rect", "square": "draw_rect",
            "circle": "draw_circle", "ellipse": "draw_circle",
            "triangle": "draw_polygon",
            "pentagon": "draw_regular_polygon", "hexagon": "draw_regular_polygon",
            "heptagon": "draw_regular_polygon", "octagon": "draw_regular_polygon",
            "nonagon": "draw_regular_polygon",  "decagon": "draw_regular_polygon",
            "trapezoid": "draw_polygon", "trapezium": "draw_polygon",
            "parallelogram": "draw_polygon", "rhombus": "draw_polygon",
        }

        # Detect the shape name in user text
        detected_shape = None
        for name in draw_action_for:
            if name in text:
                detected_shape = name
                break

        # Detect N-sided polygon
        sides = None
        if detected_shape in poly_map:
            sides = poly_map[detected_shape]
        m = re.search(r"(\d+)\s*[- ]?sided|(\d+)\s+sides", text)
        if m:
            sides = int(m.group(1) or m.group(2))
            if not detected_shape:
                detected_shape = f"{sides}-sided"

        if not detected_shape:
            # Nothing geometry-specific detected — return unchanged
            return plan

        # ── Check whether LLM already drew the expected shape ──
        expected_actions = {draw_action_for.get(detected_shape, "draw_regular_polygon")}
        # Also accept draw_line clusters as triangle proxies
        if detected_shape == "triangle":
            expected_actions.add("draw_line")

        existing_draw_actions = {v.get("action") for v in visuals}
        shape_already_drawn = bool(expected_actions & existing_draw_actions - {"draw_text", "draw_line", "clear"})

        # For triangle: need at least 3 draw_line actions
        if detected_shape == "triangle":
            line_count = sum(1 for v in visuals if v.get("action") == "draw_line")
            shape_already_drawn = line_count >= 3

        if shape_already_drawn:
            # LLM drew the shape — only fix label positions if they overlap the shape
            plan["visuals"] = self._fix_label_positions(visuals, detected_shape)
            return plan

        # ── LLM forgot the shape — inject it ──
        logger.warning(f"[force_shape] LLM omitted diagram for '{detected_shape}' — injecting fallback")

        has_clear = any(v.get("action") == "clear" for v in visuals)
        insert_at = 0
        if not has_clear:
            visuals.insert(0, {"speech_id": speech_id, "action": "clear"})
            insert_at = 1

        if detected_shape in ("rectangle", "square"):
            visuals.insert(insert_at, {"speech_id": speech_id, "action": "draw_rect",
                                        "x": 500, "y": 160, "w": 220, "h": 160})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "shape", "x": 580, "y": 145})

        elif detected_shape == "circle":
            visuals.insert(insert_at, {"speech_id": speech_id, "action": "draw_circle",
                                        "x": 620, "y": 270, "r": 100})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "r", "x": 635, "y": 260})

        elif detected_shape == "triangle":
            # Right triangle pointing right
            pts = [[500, 380], [700, 380], [500, 160]]
            visuals.insert(insert_at,     {"speech_id": speech_id, "action": "draw_line",
                                            "x1": 500, "y1": 380, "x2": 700, "y2": 380})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_line",
                                            "x1": 700, "y1": 380, "x2": 500, "y2": 160})
            visuals.insert(insert_at + 2, {"speech_id": speech_id, "action": "draw_line",
                                            "x1": 500, "y1": 160, "x2": 500, "y2": 380})
            # Labels: base below bottom edge, height left of vertical edge
            visuals.insert(insert_at + 3, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "base", "x": 575, "y": 410})
            visuals.insert(insert_at + 4, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "height", "x": 430, "y": 275})

        elif detected_shape in ("trapezoid", "trapezium"):
            visuals.insert(insert_at,     {"speech_id": speech_id, "action": "draw_polygon",
                                            "points": [[520,170],[680,170],[720,360],[480,360]]})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "a (top)", "x": 560, "y": 155})
            visuals.insert(insert_at + 2, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "b (bottom)", "x": 540, "y": 390})

        elif detected_shape in ("parallelogram", "rhombus"):
            visuals.insert(insert_at,     {"speech_id": speech_id, "action": "draw_polygon",
                                            "points": [[540,170],[740,170],[680,360],[480,360]]})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_text",
                                            "text": "base", "x": 575, "y": 390})

        elif sides:  # Regular polygon fallback
            clamped = max(3, min(12, sides))
            visuals.insert(insert_at,     {"speech_id": speech_id, "action": "draw_regular_polygon",
                                            "sides": clamped, "cx": 620, "cy": 270, "radius": 110})
            visuals.insert(insert_at + 1, {"speech_id": speech_id, "action": "draw_text",
                                            "text": f"s = side length", "x": 545, "y": 405})

        plan["visuals"] = visuals
        return plan

    def _fix_label_positions(self, visuals, shape_name):
        """
        Post-process visuals: push any draw_text that overlaps the diagram zone
        to safe positions above or below the shape.
        Diagram zone x: 480-750. Safe label y: above=130, below=430.
        """
        DIAGRAM_X_MIN = 480
        DIAGRAM_X_MAX = 750
        LABEL_Y_ABOVE = 130   # safe y for labels above shape
        LABEL_Y_BELOW = 430   # safe y for labels below shape

        fixed = []
        for v in visuals:
            if v.get("action") == "draw_text":
                x = v.get("x", 0)
                y = v.get("y", 0)
                # Only adjust labels that are in the diagram zone
                if DIAGRAM_X_MIN <= x <= DIAGRAM_X_MAX:
                    # If y is in the middle of the shape area (160-420), push to below
                    if 160 <= y <= 420:
                        v = dict(v)  # don't mutate original
                        v["y"] = LABEL_Y_BELOW
                        LABEL_Y_BELOW += 30  # stack multiple labels
            fixed.append(v)
        return fixed

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
                
                try:
                    logger.info(f"📝 Processing Turn: {user_text}")
                    
                    if self.is_math_query(user_text):
                        logger.info("DEBUG: math query detected - generating teaching plan")
                        if self.ui_signals:
                            self.ui_signals.thinking.emit()
                            
                        anonymized_text = self.privacy_manager.anonymize(user_text)
                        self.conversation_manager.add_user_message(anonymized_text)
                    
                        try:
                            # LLM Generation
                            messages = self.conversation_manager.get_messages()
                            plan = self.llm_client.generate_teaching_plan(messages)
                            plan = self.force_shape_if_missing(plan, user_text)
                            if self.ui_signals:
                                self.ui_signals.show_teaching_layout.emit()
                            self.run_teaching_plan(plan, output_device_index, continuous_vad)
                            continue
                        
                        except Exception as e:
                            logger.error(f"Teaching plan failed, falling back to normal response: {e}")
                        
                    if self.ui_signals:
                        self.ui_signals.thinking.emit()
                    # Anonymize and prepare history
                    anonymized_text = self.privacy_manager.anonymize(user_text)
                    self.conversation_manager.add_user_message(anonymized_text)
                    
                    # LLM Generation
                    messages = self.conversation_manager.get_messages()
                    llm_start = time.perf_counter()
                    
                    # Start audio playback (callback mode)
                    # AEC: Provide reference audio back to VAD
                    self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
                    if self.ui_signals:
                        self.ui_signals.start_talking.emit()
                    self.audio_player.start_streaming(output_device_index=output_device_index)
                    
                    # ── ECHO GUARD: Signal that bot is now speaking ──
                    self._bot_is_speaking = True
                    continuous_vad.set_playback_state(True)
                    
                    response_chunks = []
                    
                    # Helper to collect text while streaming
                    def text_collector(stream):
                        for chunk in stream:
                            if self._interruption_event.is_set():
                                break
                            response_chunks.append(chunk)
                            yield chunk
                            
                    # Pipeline: LLM -> TTS -> AudioPlayer
                    llm_stream = self.llm_client.generate_response_stream(messages)
                    collected_stream = text_collector(llm_stream)
                    tts_stream = self.tts_client.synthesize_stream(collected_stream)
                    
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
                        if self.ui_signals:
                            self.ui_signals.teaching_text.emit(full_response)
                        if full_response:
                            self._last_bot_response = full_response
                            self.conversation_manager.add_assistant_message(full_response)
                            logger.info(f"🤖 Bot: {full_response}")
                    else:
                        # Interrupted — still store partial response for similarity filter
                        partial = "".join(response_chunks)
                        if partial:
                            self._last_bot_response = partial
                    
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
        logger.warning("⚠️ PulseAudio not found - using default output device")
        return None

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
                
            if self.ui_signals:
                self.ui_signals.show_face_fullscreen.emit()
            
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
    
    def run(self):
        """Start Jarvis and run the main loop."""
        if self.ui_signals:
            self.ui_signals.stop_talking.emit()
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
    import os

    # -----------------------------
    # DISPLAY CONFIG (KEEP THIS)
    # -----------------------------
    face_env = os.getenv("FACE_ENABLED", "").strip().lower()
    if face_env in ("true", "1", "yes"):
        face_enabled = True
    elif face_env in ("false", "0", "no"):
        face_enabled = False
    else:
        face_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if face_enabled:
        backend = os.getenv("DISPLAY_BACKEND", "").strip().lower()

        if backend == "vnc":
            os.environ["DISPLAY"] = ":1"
        elif backend == "physical":
            os.environ["DISPLAY"] = ":0"
        elif backend == "auto":
            for candidate in (":1", ":0"):
                if os.path.exists(f"/tmp/.X11-unix/X{candidate[1:]}"):
                    os.environ["DISPLAY"] = candidate
                    break
            else:
                os.environ["DISPLAY"] = ":0"
        elif not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = ":0"

        # XAUTHORITY fix
        if not os.environ.get("XAUTHORITY"):
            xauth_path = os.path.expanduser("~/.Xauthority")
            if os.path.exists(xauth_path):
                os.environ["XAUTHORITY"] = xauth_path

        display = os.environ.get("DISPLAY", "")
        display_num = display.lstrip(":").split(".")[0]
        socket_path = f"/tmp/.X11-unix/X{display_num}"

        if not os.path.exists(socket_path):
            logger.warning("Display not available, switching to headless")
            face_enabled = False
        else:
            logger.info(f"Running UI on display {display}")

    else:
        logger.info("Running headless mode")

    # -----------------------------
    # QT APP
    # -----------------------------
    if face_enabled:
        # QApplication MUST be the first Qt object — before UISignals, MainWindow,
        # or any object that inherits QObject (including JarvisBot's PyAudio threads).
        app = QApplication.instance() or QApplication(sys.argv)

        # Hide the mouse cursor for kiosk/touchscreen deployments
        # (remove this line if you want a visible cursor)
        # from PyQt5.QtCore import Qt as _Qt
        # app.setOverrideCursor(_Qt.BlankCursor)

        ui_signals = UISignals()
        window = MainWindow(ui_signals)
        window.show()

        jarvis = JarvisBot(ui_signals=ui_signals)

        worker = threading.Thread(target=jarvis.run, daemon=True)
        worker.start()

        sys.exit(app.exec_())

    else:
        # Headless fallback
        jarvis = JarvisBot(ui_signals=None)
        jarvis.run()

if __name__ == "__main__":
    main()
