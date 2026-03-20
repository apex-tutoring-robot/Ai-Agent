import os
# Allow the OS to use its default display and QT backend, rather than hardcoding.

import threading
import time
import queue
from queue import Queue
from typing import Optional
from difflib import SequenceMatcher
from dotenv import load_dotenv
from visuals.faces.face_animator import FaceAnimator
import numpy as np
import logging

from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from conversation.state_manager import ConversationStateManager
from privacy.privacy_manager import PrivacyManager
from vision.camera import Camera
from guardrails.guardrails_manager import GuardrailsManager

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
        self.tts_client = TextToSpeechClient()
        self.privacy_manager = PrivacyManager()
        self.guardrails = GuardrailsManager(
            config_path=os.getenv('GUARDRAILS_CONFIG_PATH', 'config/guardrails')
        )
        self.camera = Camera()
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

    _CAMERA_TRIGGERS = {
        "scan", "photo", "picture", "camera",
        "take a photo", "take a picture", "scan my homework",
        "look at this", "check this", "show you",
    }

    @staticmethod
    def _is_camera_trigger(text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in JarvisBot._CAMERA_TRIGGERS)

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

                # Reset idle timer immediately so long camera/LLM processing doesn't trigger timeout
                continuous_vad.reset_idle_timer()

                # Signal we are busy
                self._speaker_busy.set()
                self._interruption_event.clear()
                self._barge_in_detected.clear()
                
                try:
                    logger.info(f"📝 Processing Turn: {user_text}")

                    # Anonymize
                    # anonymized_text = self.privacy_manager.anonymize(user_text)

                    # ── Camera Vision ──────────────────────────────────────────
                    # If the user's phrase is a camera trigger, capture an image,
                    # upload it to Azure Blob Storage, and store the SAS URL in
                    # conversation history so follow-up questions can reference it.
                    if self._is_camera_trigger(user_text):
                        if self.face:
                            self.face.start_scanning()
                        try:
                            saved_path = self.camera.capture_and_save()
                            logger.info(f"📷 Image saved to {saved_path}")

                            import base64 as _b64
                            with open(saved_path, "rb") as _f:
                                data_url = f"data:image/jpeg;base64,{_b64.b64encode(_f.read()).decode()}"

                            try:
                                extracted = self.llm_client.extract_image_content(data_url)
                                combined = f"{user_text}\n\n[Scanned homework content:\n{extracted}]"
                                anonymized_text = self.privacy_manager.anonymize(combined)
                                # self.conversation_manager.add_user_message(combined)
                                # messages = self.conversation_manager.get_messages()
                                logger.info("📷 Image extracted and stored as text")
                            except Exception as extract_err:
                                logger.error(f"📷 Image extraction failed: {extract_err}")
                                if self.face:
                                    self.face.start_talking()
                                error_audio = self.tts_client.synthesize_to_audio(
                                    "Sorry, I had trouble reading the image. Please try again."
                                )
                                self.audio_player.start_streaming(output_device_index=output_device_index)
                                self.audio_player.queue_audio(error_audio)
                                self.audio_player.stop_streaming(immediate=False)
                                if self.face:
                                    self.face.start_idle()
                                continue
                        except Exception as cam_err:
                            logger.error(f"📷 Camera failed to capture: {cam_err} — falling back to text-only")
                            combined = f"{user_text}\n\nCamera is not working, please talk to me!"
                            anonymized_text = self.privacy_manager.anonymize(combined)
                            # self.conversation_manager.add_user_message(anonymized_text)
                            # messages = self.conversation_manager.get_messages()
                    else:
                        anonymized_text = self.privacy_manager.anonymize(user_text)
                        # self.conversation_manager.add_user_message(anonymized_text)
                        # messages = self.conversation_manager.get_messages()
                    # ──────────────────────────────────────────────────────────
                    allowed, refusal_msg = self.guardrails.check_input(anonymized_text)

                    if allowed:
                        # Add to conversation history only after input passes
                        self.conversation_manager.add_user_message(anonymized_text)
                        messages = self.conversation_manager.get_messages()

                        # Buffer the FULL LLM response before starting TTS.
                        # This is required so the output guardrail can check the
                        # complete response before any audio is produced.
                        llm_start = time.perf_counter()
                        if self.face:
                            self.face.start_thinking()
                        response_chunks = []
                        for chunk in self.llm_client.generate_response_stream(messages):
                            if self._interruption_event.is_set():
                                break
                            response_chunks.append(chunk)

                        # Handle interruption during LLM generation
                        if self._interruption_event.is_set():
                            partial = "".join(response_chunks)
                            if partial:
                                self._last_bot_response = partial  # for similarity filter
                            continuous_vad.reset_idle_timer()
                            continue

                        full_response = "".join(response_chunks)
                        if not full_response:
                            continuous_vad.reset_idle_timer()
                            continue

                        # ── GUARDRAILS: OUTPUT CHECK ─────────────────────────
                        # Checks complete LLM response before any audio plays.
                        safe, final_text = self.guardrails.check_output(full_response)
                        if safe:
                            self.conversation_manager.add_assistant_message(final_text)
                            self._last_bot_response = final_text
                            logger.info(f"🤖 Bot: {final_text}")
                        else:
                            logger.warning("🛡️ Guardrails blocked LLM output — substituting refusal")
                            # final_text is already the kid-friendly refusal; history not updated
                    else:
                        # Input blocked — speak refusal, skip LLM entirely
                        logger.info("🛡️ Guardrails blocked input — speaking refusal")
                        final_text = refusal_msg

                    # LLM Generation
                    # llm_start = time.perf_counter()
                    # Start audio playback (callback mode)
                    # AEC: Provide reference audio back to VAD
                    self.audio_player.on_audio_played = continuous_vad.provide_reference_audio
                    if self.face:
                        self.face.start_talking()
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
                    if self.face:
                        self.face.start_idle()
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
        if self.face:
            self.face.start_idle()
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
