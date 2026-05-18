"""
Continuous Voice Activity Detection for conversation mode.
Listens continuously with idle timeout for multi-turn conversations.
Supports multiple utterances in a conversation session
"""

import numpy as np
import time
import os
import logging
from collections import deque
import queue
import threading
from typing import Generator, Optional
from dotenv import load_dotenv
import webrtcvad
import pyaudio
import traceback

# Must be imported before PyAudio initialises to suppress ALSA noise
from audio import suppress_alsa  # noqa: F401

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

try:
    from utils import imp_shim
    imp_shim.install_shim()
    import speexdsp
    from speexdsp import EchoCanceller
    try:
        from speexdsp import Preprocessor
        HAS_PREPROCESSOR = True
    except ImportError:
        logger.warning("⚠️  SpeexDSP 'Preprocessor' not found - Noise Suppression will be disabled")
        HAS_PREPROCESSOR = False
        Preprocessor = None
    
    HAS_SPEEX = True
except Exception:
    logger.warning("SpeexDSP import failed (AEC will be disabled):")
    logger.warning(traceback.format_exc())
    HAS_SPEEX = False
    HAS_PREPROCESSOR = False
    EchoCanceller = None
    Preprocessor = None

# Debug: Track if AEC is available
if HAS_SPEEX:
    if HAS_PREPROCESSOR:
        logger.info("✅ SpeexDSP loaded - AEC and Preprocessor available")
    else:
        logger.info("✅ SpeexDSP loaded - AEC active (Preprocessor/NS missing)")
else:
    logger.warning("⚠️  SpeexDSP NOT found - using Digital Ducking fallback for interruptions")


class ContinuousVADCapture:
    """Continuous voice activity detection with idle timeout for conversation mode."""
    
    def __init__(
        self,
        sample_rate: int = None,
        frame_duration_ms: int = None,
        vad_aggressiveness: int = None,
        silence_timeout_ms: int = None,
        idle_timeout_seconds: int = None,
        input_device_index: Optional[int] = None,
        audio_stream = None,  # Existing audio stream to reuse
        pa = None,  # Existing PyAudio instance to reuse
        player: Optional[object] = None # AudioPlayer instance for real-time state
    ):
        """
        Initialize continuous VAD capture.
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (must be 10, 20, or 30)
            vad_aggressiveness: VAD aggressiveness (0-3, 3 is most aggressive)
            silence_timeout_ms: Milliseconds of silence before ending one utterance
            idle_timeout_seconds: Seconds of no speech before ending conversation
            input_device_index: Index of audio input device
            audio_stream: Existing PyAudio stream to reuse (optional)
            pa: Existing PyAudio instance to reuse (optional)
        """
        self.sample_rate = sample_rate or int(os.getenv('SAMPLE_RATE', 16000))
        self.frame_duration_ms = frame_duration_ms or int(os.getenv('VAD_FRAME_DURATION_MS', 20))
        self.vad_aggressiveness = vad_aggressiveness or int(os.getenv('VAD_AGGRESSIVENESS', 2))
        self.silence_timeout_ms = silence_timeout_ms or int(os.getenv('SILENCE_TIMEOUT_MS', 2000))
        self.idle_timeout_seconds = idle_timeout_seconds or int(os.getenv('CONVERSATION_IDLE_TIMEOUT_SECONDS', 10))
        env_input = os.getenv('AUDIO_INPUT_DEVICE_INDEX')
        self.input_device_index = (
            input_device_index
            if input_device_index is not None
            else (int(env_input) if env_input not in (None, "") else None)
        )   
        
        # Validate sample rate for WebRTC VAD
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000. Got {self.sample_rate}")
        
        # Validate frame duration for WebRTC VAD
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms. Got {self.frame_duration_ms}")
        
        # Calculate chunk size based on frame duration
        self.chunk_size = int((self.sample_rate * self.frame_duration_ms) / 1000)
        
        if audio_stream and pa:
            logger.info(f"Continuous VAD initialized with reused audio stream: {self.sample_rate}Hz, {self.frame_duration_ms}ms frames, "
                       f"idle timeout={self.idle_timeout_seconds}s")
        else:
            logger.info(f"Continuous VAD initialized: {self.sample_rate}Hz, {self.frame_duration_ms}ms frames, "
                       f"idle timeout={self.idle_timeout_seconds}s")
        
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.pa = pa  # Use provided instance or None
        self.audio_stream = audio_stream  # Use provided stream or None
        self.last_speech_time = time.time() # Initialize immediately to prevent NoneType error
        self._owns_pa = (pa is None)  # Track if we should terminate PyAudio
        self._owns_stream = (audio_stream is None)  # Track if we should close stream
        
        # Latency tracking
        self.speech_start_time = None  # When user starts speaking
        self.silence_detected_time = None  # When user stops speaking
        
        self.echo_canceller = None
        self.reference_queue = queue.Queue(maxsize=100)
        self.reference_buffer = b"" # Temporary buffer for incomplete chunks
        
        # Real-time state from AudioPlayer
        self.player = player
        
        self._is_monitoring = False
        self._interruption_callback = None
        self._interrupted_this_turn = False
        self.interruption_buffer = []
        
        # ── Playback-aware gating (mute-the-pipe) ──
        self._is_bot_playing = False           # Set by main.py via set_playback_state()
        self._barge_in_event = None            # threading.Event from JarvisBot (set on barge-in)
        self._barge_in_energy_threshold = int(os.getenv('BARGE_IN_ENERGY_THRESHOLD', '1500'))
        self._barge_in_consecutive_needed = int(os.getenv('BARGE_IN_CHUNKS_NEEDED', '3'))
        # AEC Ring Buffer (Hardware-Aligned Reference Audio)
        # 1-second circular buffer for reference audio
        self.ref_ring_buffer = np.zeros(self.sample_rate, dtype=np.int16)
        self.ref_ring_pos = 0
        self.ref_ring_timestamp = 0.0 # DAC Time of the last sample in ring
        self.last_reference_time = 0.0 # Wall-clock time of last reference update
        self.ref_lock = threading.Lock()
        self.echo_canceller = None
        self.preprocessor = None
        
        # AEC Delay Compensation & Stabilization
        self.mic_delay_buffer = deque(maxlen=int(os.getenv('AEC_DELAY_CHUNKS', '12')))
        self._last_telemetry_time = 0
        self._detected_lags = deque(maxlen=50) # Stable median
        self._last_correlation_time = 0
        self._aec_locked_offset = None # Permanent lock per session
        
        self.preprocessor = None
        #if HAS_SPEEX:
            # Speex echo canceller needs (frame_size, filter_length)
            # frame_size must match chunk_size (320 for 20ms at 16kHz)
            # filter_length is typically 2000-4000
         #   try:
               # 4096 taps = 256ms of tail length. Better for Pi rooms.
          #     self.echo_canceller = EchoCanceller(self.chunk_size, 4096, self.sample_rate)
           # except Exception as e:
            #    if "No constructor defined" in str(e) or "abstract" in str(e).lower():
             #       logger.info("ℹ️  Using EchoCanceller_create factory (SWIG abstract class workaround)")
              #      import speexdsp
               #     self.echo_canceller = speexdsp.EchoCanceller_create(self.chunk_size, 2048, self.sample_rate)
                #else:
                 #   raise
            
            # Initialize Preprocessor (Denoise + AGC)
            # Control via env variable (default: True)
            #enable_ns = os.getenv('ENABLE_SPEEX_NOISE_SUPPRESSION', 'true').lower() == 'true'
            
            #if enable_ns and HAS_PREPROCESSOR:
             #   try:
              #      self.preprocessor = Preprocessor(self.chunk_size, self.sample_rate)
               #     self.preprocessor.denoise = True
                #    self.preprocessor.agc = True
                 #   self.preprocessor.dereverb = True
                  #  self.preprocessor.agc_level = 8000
                   # logger.info("✅ Speex Preprocessor (Denoise/AGC) initialized")
                #except Exception as e:
                 #   logger.warning(f"Failed to init Speex Preprocessor: {e}")
            #elif enable_ns and not HAS_PREPROCESSOR:
             #   logger.warning("ℹ️  Speex Preprocessor requested but not available in this version")
            #else:
             #   logger.info("ℹ️  Speex Preprocessor disabled via env var")
            
            #logger.info("✅ Speex Echo Canceller initialized")
            
    def on_audio_played(self, audio_data: bytes) -> None:
        """Alias for provide_reference_audio to match AudioPlayer callback signature."""
        self.provide_reference_audio(audio_data)

    def set_playback_state(self, is_playing: bool) -> None:
        """Called by main.py to inform VAD whether the bot is currently playing audio.
        
        When is_playing=True, always_streaming will feed silence to STT
        instead of mic audio, preventing the bot from hearing itself.
        Barge-in detection runs separately and can open the gate.
        """
        self._is_bot_playing = is_playing
        if is_playing:
            logger.info("🔇 VAD: Playback started — muting STT pipe (silence gate ON)")
        else:
            logger.info("🔊 VAD: Playback ended — STT pipe open")

    def set_barge_in_event(self, event) -> None:
        """Register the threading.Event that will be set when barge-in is detected."""
        self._barge_in_event = event
    
    def listen_continuous(self) -> Generator[bytes, None, None]:
        """
        Listen continuously for speech, yielding audio segments.
        Ends after idle_timeout_seconds of no speech.
        
        Yields:
            Raw audio bytes of captured speech
        """
        try:
            # Only create stream if not already provided
            # Only create stream if not already provided
            if not self.audio_stream:
                self.ensure_stream_open()
            else:
                logger.info("♻️  Reusing existing audio stream from wake word detector")
                # Restart the stream if it was stopped
                if not self.audio_stream.is_active():
                    self.audio_stream.start_stream()
                    logger.info("▶️  Restarted paused audio stream")
            
            logger.info("🎤 CONVERSATION MODE - Listening continuously...")
            logger.info(f"⏱️  Will end if there is no speech for {self.idle_timeout_seconds} seconds")
            
            self.last_speech_time = time.time()
            
            while True:
                # Check if idle timeout exceeded
                idle_duration = time.time() - self.last_speech_time
                if idle_duration >= self.idle_timeout_seconds:
                    logger.info(f"⏱️  {self.idle_timeout_seconds}s idle timeout reached - ending conversation")
                    return
                
                # Capture one utterance
                audio_data = self._capture_one_utterance()
                
                if audio_data:
                    # Reset idle timer - we got speech!
                    self.last_speech_time = time.time()
                    yield audio_data
        
        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
            raise
        
        finally:
            # Ensure cleanup happens
            logger.info("Cleaning up continuous VAD...")
            self.cleanup()
    
    def _capture_one_utterance(self) -> Optional[bytes]:
        """
        Capture one speech utterance.
        Returns None if no speech detected within a reasonable time.
        """
        try:
            frames = []
            speech_started = False
            silence_start = None
            
            # Two separate timeouts for different purposes
            initial_speech_timeout = 0.5  # Wait 1.5s for user to START speaking
            post_speech_silence_timeout_ms = 800  # End utterance after 800ms of silence
            
            listen_start = time.time()
            
            while True:
                # Give up if no speech detected within initial timeout
                if not speech_started and (time.time() - listen_start) > initial_speech_timeout:
                    return None
                
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Mix down to mono if device is stereo
                if getattr(self, '_actual_channels', 1) == 2:
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_chunk = audio_array.reshape(-1, 2)[:, 0].tobytes()
                
                # Apply Acoustic Echo Cancellation (AEC)
                if HAS_SPEEX and self.echo_canceller:
                    try:
                        ref_chunk = self.reference_queue.get_nowait()
                        audio_chunk = self.echo_canceller.run(audio_chunk, ref_chunk)
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.warning(f"AEC Error in capture: {e}")

                # Apply Speex Preprocessor (Denoise + AGC + Dereverb)
                if self.preprocessor:
                    try:
                        audio_chunk = self.preprocessor.process(audio_chunk)
                    except:
                        pass
                
                # Check if speech is present
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    if not speech_started:
                        logger.info("💬 Speech detected...")
                        speech_started = True
                    
                    frames.append(audio_chunk)
                    silence_start = None  # Reset silence timer
                
                elif speech_started:
                    # We were recording, now silence
                    frames.append(audio_chunk)  # Include some silence for natural cutoff
                    
                    if silence_start is None:
                        silence_start = time.time()
                    
                    # Check if post-speech silence timeout exceeded
                    silence_duration_ms = (time.time() - silence_start) * 1000
                    if silence_duration_ms >= post_speech_silence_timeout_ms:
                        logger.info(f"✓ Utterance complete (silence: {silence_duration_ms:.0f}ms)")
                        break
            
            # Combine all frames
            audio_data = b''.join(frames)
            logger.info(f"Captured {len(audio_data)} bytes of audio")
            
            return audio_data if audio_data else None
        
        except Exception as e:
            logger.error(f"Error capturing utterance: {e}")
            return None
    
    def stream_audio_chunks(self) -> Generator[bytes, None, None]:
        """
        Stream audio chunks in real-time for streaming STT.
        Yields audio chunks immediately as they're captured.
        Signals end of utterance with None when user stops speaking.
        
        Yields:
            Audio chunks (20ms each) or None to signal end of utterance
        """
        try:
            # Only create stream if not already provided
            # Only create stream if not already provided
            if not self.audio_stream:
                # Cleanup any existing resources first
                if self.audio_stream:
                    self.cleanup()
                
                self.ensure_stream_open()
            else:
                logger.info("♻️  Reusing existing audio stream from wake word detector")
                # Restart the stream if it was stopped
                if not self.audio_stream.is_active():
                    self.audio_stream.start_stream()
                    logger.info("▶️  Restarted paused audio stream")

            speech_started = False
            silence_start = None
            
            # Two separate timeouts for different purposes
            initial_speech_timeout = 1.5  # Wait 1.5s for user to START speaking
            post_speech_silence_timeout = 0.8  # End utterance after 800ms of silence
            
            # -----------------------------------------------------------------
            # ACTIVE MIC DRAIN: After playback ends, the ALSA hardware ring-buffer
            # still holds frames contaminated with speaker echo. 
            # 
            # SPECIAL CASE: If this turn was triggered by an interruption, we SKIP 
            # the drain because we want the user's interruption speech immediately.
            # -----------------------------------------------------------------
            if self._interrupted_this_turn and self.interruption_buffer:
                logger.info(f"⚡ Handoff: Yielding {len(self.interruption_buffer)} interruption chunks...")
                for chunk in self.interruption_buffer:
                    yield chunk
                
                # Reset state but keep speech_started=True so we don't time out
                self._interrupted_this_turn = False
                self.interruption_buffer = []
                speech_started = True 
                listen_start = time.time() # Reset timeout
            else:
                drain_duration = float(os.getenv('POST_PLAYBACK_DRAIN_MS', '400')) / 1000.0
                drain_end = time.time() + drain_duration
                drained_chunks = 0
                if drain_duration > 0:
                    while time.time() < drain_end:
                        try:
                            self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                            drained_chunks += 1
                        except Exception:
                            break
                    logger.debug(f"Drained {drained_chunks} mic buffer chunks ({drain_duration*1000:.0f}ms) to clear echo tail")
            # -----------------------------------------------------------------
            
            listen_start = time.time()
            
            logger.info("🎤 Streaming audio chunks...")
            
            chunk_count = 0
            speech_chunk_count = 0
            silence_chunk_count = 0
            
            # Debouncing: count consecutive speech chunks to avoid noise triggering
            consecutive_speech_chunks = 0
            min_speech_chunks_to_cancel_silence = 25  # 25 chunks (500ms) to cancel mid-speech silence
            # Require 8 consecutive VAD-positive chunks (~160ms) before we declare speech START.
            # This prevents echo blips that survived the drain from triggering the pipeline.
            min_consecutive_to_start = int(os.getenv('MIN_SPEECH_CHUNKS_TO_START', '8'))
            pending_speech_chunks = []  # Buffer chunks while waiting to confirm speech start
            
            # Reset latency tracking for this turn
            self.silence_detected_time = None
            
            # PRE-BUFFER: Keep recent chunks BEFORE confirmed speech detected.
            from collections import deque
            pre_buffer = deque(maxlen=7)  # 7 chunks * 20ms = 140ms


            
            while True:
                # Give up if no speech detected within initial timeout
                if not speech_started and (time.time() - listen_start) > initial_speech_timeout:
                    logger.info("No speech detected, ending stream")
                    yield None  # Signal end
                    return
                
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                chunk_count += 1
                
                # Mix down to mono if device is stereo
                if getattr(self, '_actual_channels', 1) == 2:
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_chunk = audio_array.reshape(-1, 2)[:, 0].tobytes()
                
                # Apply Acoustic Echo Cancellation (AEC)
                # Ensure we scrub the bot's voice from the microphone input
                if HAS_SPEEX and self.echo_canceller:
                    try:
                        # Try to get matching reference chunk
                        ref_chunk = self.reference_queue.get_nowait()
                        
                        try:
                            # Standard process(mic, ref) - confirmed via SpeexInspect
                            audio_chunk = self.echo_canceller.process(audio_chunk, ref_chunk)
                        except (AttributeError, Exception):
                            # Fallback just in case
                            audio_chunk = self.echo_canceller.run(audio_chunk, ref_chunk)
                    except queue.Empty:
                        pass
                    except Exception as e:
                        if not self._aec_debug_done:
                            logger.error(f"AEC Error (Stream): {e}")
                            self._aec_debug_done = True
                        
                        # Emergency Fallback to Ducking
                        is_bot_playing = (self.player and hasattr(self.player, 'last_heartbeat') and 
                                          time.time() - self.player.last_heartbeat < 0.25)
                        if is_bot_playing:
                            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                            audio_np *= 0.1 
                            audio_chunk = audio_np.astype(np.int16).tobytes()

                # Apply Speex Preprocessor (Denoise + AGC + Dereverb)
                if self.preprocessor:
                    try:
                        audio_chunk = self.preprocessor.process(audio_chunk)
                    except Exception as e:
                        # Log once to avoid spamming
                        if chunk_count % 100 == 0:
                            logger.warning(f"Preprocessor error: {e}")
                
                # Check if speech is present
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    speech_chunk_count += 1
                    consecutive_speech_chunks += 1
                    
                    if not speech_started:
                        # Accumulate chunks until we're sure this is real speech
                        # (not an echo blip or noise burst)
                        pending_speech_chunks.append(audio_chunk)
                        
                        if len(pending_speech_chunks) >= min_consecutive_to_start:
                            # Confirmed real speech - commit to speech mode
                            self.speech_start_time = time.perf_counter()
                            pre_chunks = list(pre_buffer)
                            logger.info(f"💬 Speech confirmed - yielding {len(pre_chunks)} pre-buffer + {len(pending_speech_chunks)} pending chunks...")
                            speech_started = True
                            silence_start = None
                            
                            # Yield pre-buffer first (captures the very start of speech)
                            for buffered_chunk in pre_chunks:
                                yield buffered_chunk
                            # Yield all pending (confirmed speech) chunks
                            for pending_chunk in pending_speech_chunks:
                                yield pending_chunk
                            pending_speech_chunks = []
                    elif silence_start is None:
                        # Normal speech, not in silence period
                        yield audio_chunk
                    else:
                        # We're in silence period, but detected speech
                        # Only cancel silence if we get sustained speech (debouncing)
                        if consecutive_speech_chunks >= min_speech_chunks_to_cancel_silence:
                            logger.info(f"🔊 Sustained speech detected during silence ({consecutive_speech_chunks} chunks), canceling silence timer")
                            silence_start = None  # Reset silence timer
                            silence_chunk_count = 0
                        else:
                            logger.debug(f"Brief speech blip #{consecutive_speech_chunks} during silence, ignoring...")
                        yield audio_chunk

                
                else:
                    # Not speech (silence)
                    consecutive_speech_chunks = 0  # Reset consecutive counter
                    
                    if not speech_started:
                        # Discard any partially accumulated pending chunks on silence break
                        # (ensures the 8-chunk threshold is truly CONTIGUOUS)
                        if pending_speech_chunks:
                            pending_speech_chunks = []
                        # Store in pre-buffer (circular, auto-discards old chunks)
                        pre_buffer.append(audio_chunk)
                    elif speech_started:
                        # We were recording, now silence
                        silence_chunk_count += 1
                        if silence_start is None:
                            silence_start = time.time()
                            logger.info(f"🔇 Silence started (after {speech_chunk_count} speech chunks), waiting for {post_speech_silence_timeout}s...")

                        yield audio_chunk  # Include silence for natural cutoff
                        
                        # Check if post-speech silence timeout exceeded
                        silence_duration = time.time() - silence_start
                        
                        if silence_duration >= post_speech_silence_timeout:
                            # LATENCY: Track when user stops speaking (for End-to-TTFT)
                            self.silence_detected_time = time.perf_counter()
                            logger.info(f"✓ End of utterance detected (silence: {silence_duration:.2f}s, {silence_chunk_count} silence chunks)")
                            logger.info(f"📊 Total chunks: {chunk_count} (speech: {speech_chunk_count}, silence: {silence_chunk_count})")
                            yield None  # Signal end of stream
                            break
        
        except Exception as e:
            logger.error(f"Error streaming audio chunks: {e}")
            yield None  # Signal end on error
        
        finally:
            # Do NOT cleanup here - we want to keep the stream open for full duplex
            # The lifecycle is managed by the ContinuousVADCapture instance
            pass
            
    def always_streaming(self) -> Generator[bytes, None, None]:
        """
        Continuous generator that yields audio chunks indefinitely 
        until explicitly stopped or idle timeout is reached.
        
        ECHO DEFENSE (mute-the-pipe):
        While bot is playing back audio, this yields SILENCE to STT.
        Only switches to real mic audio when a barge-in is detected
        (high energy for N consecutive chunks after AEC processing).
        """
        self.ensure_stream_open()
        self._is_monitoring = True
        self.last_speech_time = time.time()
        
        # Silence chunk for muting (same size as one frame)
        silence_chunk = b'\x00' * (self.chunk_size * 2)  # int16 = 2 bytes per sample
        
        # Barge-in detection state
        barge_in_consecutive = 0
        barge_in_active = False  # Once true, stays true until playback ends
        _barge_in_peak_rms = 0.0        # Track peak RMS this playback session
        _barge_in_log_time = 0.0        # Throttle RMS telemetry to once/sec
        # After barge-in activates, drain this many frames as silence before opening STT.
        # Gives PipeWire AEC time to suppress the ongoing bot speech (default 40ms = 2 frames).
        _barge_in_echo_drain = int(os.getenv('BARGE_IN_ECHO_DRAIN_FRAMES', '2'))
        _barge_in_drain_remaining = 0
        
        logger.info("📡 Continuous audio streaming started")
        
        while self._is_monitoring:
            # Idle timeout check
            if (time.time() - self.last_speech_time) > self.idle_timeout_seconds:
                logger.info("⏱️  Idle timeout in always_streaming — stopping")
                break
                
            try:
                # Read audio chunk - safety check for closed stream
                if not self.audio_stream or self.audio_stream.is_stopped():
                    break
                    
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Mix down to mono if device is stereo
                if getattr(self, '_actual_channels', 1) == 2:
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_chunk = audio_array.reshape(-1, 2)[:, 0].tobytes()
                
                # Apply AEC if enabled (always run AEC regardless of gate state)
                if HAS_SPEEX and self.echo_canceller:
                    try:
                        # 1. Get current wall-clock timestamp (ADC time)
                        adc_time = time.time()
                        
                        # 2. Add MIC chunk to its own delay buffer (lookahead for AEC)
                        self.mic_delay_buffer.append(audio_chunk)
                        
                        # 3. Only pull once buffer is full (compensates for hardware lag)
                        if len(self.mic_delay_buffer) >= self.mic_delay_buffer.maxlen:
                            delayed_mic = self.mic_delay_buffer[0]
                            
                            with self.ref_lock:
                                if (time.time() - self.last_reference_time < 0.2):
                                    mic_time = adc_time - (self.mic_delay_buffer.maxlen * self.frame_duration_ms / 1000.0)
                                    time_diff = self.ref_ring_timestamp - mic_time
                                    
                                    # Periodic cross-correlation for offset locking
                                    curr_time = time.time()
                                    if curr_time - self._last_correlation_time > 2.0:
                                        try:
                                            search_size = int(0.5 * self.sample_rate) 
                                            start_window = (self.ref_ring_pos - search_size) % len(self.ref_ring_buffer)
                                            if start_window + search_size <= len(self.ref_ring_buffer):
                                                ref_window = self.ref_ring_buffer[start_window : start_window + search_size]
                                            else:
                                                part1 = self.ref_ring_buffer[start_window:]
                                                part2 = self.ref_ring_buffer[:search_size - len(part1)]
                                                ref_window = np.concatenate([part1, part2])
                                            
                                            if len(ref_window) >= self.chunk_size:
                                                mic_np = np.frombuffer(delayed_mic, dtype=np.int16).astype(np.float32)
                                                ref_np = ref_window.astype(np.float32)
                                                corr = np.correlate(ref_np, mic_np, mode='valid')
                                                peak_idx = np.argmax(corr)
                                                detected_lag = len(ref_np) - peak_idx
                                                peak_val = corr[peak_idx]
                                                energy = np.sum(mic_np**2)
                                                if peak_val > 0.5 * energy:
                                                    self._detected_lags.append(detected_lag)
                                                    self._aec_locked_offset = int(np.median(self._detected_lags))
                                                    self._last_correlation_time = curr_time
                                        except Exception as ce:
                                            logger.debug(f"Correlation failed: {ce}")

                                    idx_offset = self._aec_locked_offset
                                    if idx_offset is None:
                                        hw_latency = float(os.getenv('AEC_HARDWARE_LATENCY_MS', '150')) / 1000.0
                                        idx_offset = int((time_diff + hw_latency) * self.sample_rate)
                                    
                                    if 0 <= idx_offset <= len(self.ref_ring_buffer) - self.chunk_size:
                                        start_idx = (self.ref_ring_pos - idx_offset) % len(self.ref_ring_buffer)
                                        if start_idx + self.chunk_size <= len(self.ref_ring_buffer):
                                            ref_chunk_np = self.ref_ring_buffer[start_idx:start_idx+self.chunk_size]
                                        else:
                                            part1 = self.ref_ring_buffer[start_idx:]
                                            part2 = self.ref_ring_buffer[:self.chunk_size - len(part1)]
                                            ref_chunk_np = np.concatenate([part1, part2])
                                        
                                        ref_chunk = ref_chunk_np.astype(np.int16).tobytes()
                                        audio_chunk = self.echo_canceller.process(delayed_mic, ref_chunk)
                                        
                                        # Telemetry
                                        if curr_time - self._last_telemetry_time > 5.0:
                                            logger.info(f"🎯 AEC Locked: offset={idx_offset} smp (~{idx_offset/self.sample_rate*1000:.1f}ms)")
                                            self._last_telemetry_time = curr_time
                                    else:
                                        audio_chunk = delayed_mic
                                else:
                                    audio_chunk = delayed_mic

                    except Exception as e:
                        logger.error(f"AEC Error: {e}")

                # ════════════════════════════════════════════════════════
                # MUTE-THE-PIPE: Decide whether to yield real audio or silence
                # ════════════════════════════════════════════════════════
                if self._is_bot_playing:
                    # Bot speaking counts as activity — keep conversation alive
                    self.last_speech_time = time.time()
                    
                    # Measure mic energy AFTER AEC to detect user speech above residual echo
                    mic_samples = np.frombuffer(audio_chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(mic_samples.astype(np.float32)**2))
                    
                    if rms > _barge_in_peak_rms:
                        _barge_in_peak_rms = rms
                    _now = time.time()
                    if _now - _barge_in_log_time >= 1.0:
                        logger.debug(f"🔉 Barge-in RMS: {rms:.0f} (peak={_barge_in_peak_rms:.0f}, threshold={self._barge_in_energy_threshold})")
                        _barge_in_log_time = _now

                    if rms > self._barge_in_energy_threshold:
                        barge_in_consecutive += 1
                    else:
                        barge_in_consecutive = 0

                    if barge_in_consecutive >= self._barge_in_consecutive_needed and not barge_in_active:
                        # BARGE-IN DETECTED! Open the gate.
                        barge_in_active = True
                        _barge_in_drain_remaining = _barge_in_echo_drain
                        logger.info(f"🎤🔥 BARGE-IN detected! RMS={rms:.0f} for {barge_in_consecutive} chunks — opening STT gate")
                        if self._barge_in_event:
                            self._barge_in_event.set()

                    if barge_in_active:
                        if _barge_in_drain_remaining > 0:
                            # Drain echo tail before feeding real audio to STT
                            _barge_in_drain_remaining -= 1
                            yield silence_chunk
                        else:
                            yield audio_chunk
                    else:
                        # Gate is closed — feed silence to prevent echo transcription
                        yield silence_chunk
                
                else:
                    # Bot is NOT playing — normal operation
                    # Reset barge-in state for next playback session
                    if barge_in_active or barge_in_consecutive > 0 or _barge_in_peak_rms > 0:
                        if _barge_in_peak_rms > 0:
                            logger.debug(f"🔉 Barge-in session ended: peak RMS={_barge_in_peak_rms:.0f}, threshold={self._barge_in_energy_threshold}")
                        barge_in_active = False
                        barge_in_consecutive = 0
                        _barge_in_peak_rms = 0.0
                    
                    # VAD check to update idle timer
                    try:
                        is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                        if is_speech:
                            self.last_speech_time = time.time()
                    except:
                        pass
                    
                    yield audio_chunk
                
            except Exception as e:
                logger.error(f"Error in always_streaming: {e}")
                time.sleep(0.01)
                
        logger.info("📡 Continuous audio streaming stopped")
        self._is_monitoring = False
    
    def start_background_monitoring(self, callback) -> None:
        """
        Start monitoring for speech in a non-blocking background thread.
        Triggers callback if speech is detected.
        # ... (rest of method unchanged)
        """
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._interruption_callback = callback
        self._monitoring_start_time = time.time()
        
        # We need a separate thread for this
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="VADMonitoringThread"
        )
        self._monitoring_thread.start()
        logger.info("📡 Background speech monitoring started")

    def stop_background_monitoring(self) -> None:
        """Stop background speech monitoring."""
        self._is_monitoring = False
        if hasattr(self, '_monitoring_thread') and self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
            self._monitoring_thread = None
        logger.info("📡 Background speech monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Internal loop for background monitoring."""
        try:
            # Re-ensure stream is open
            if not self.audio_stream:
                self.ensure_stream_open()
            
            # Sensitivity Tuning
            # -----------------------------------------------------------------
            # Silence: User must speak for ~200ms (10 chunks) to start a turn.
            # Playing: User must speak for ~300ms (15 chunks) to interrupt the bot.
            base_threshold = 10    
            playing_threshold = 15 
            
            # Global Energy threshold: RMS value must be > 500 to be considered speech.
            # This filters out mic hiss while allowing natural speech.
            energy_threshold_base = int(os.getenv('INTERRUPTION_ENERGY_THRESHOLD', '500'))
            
            # Buffer 500ms (25 chunks) to ensure we don't clipped the start of interruptions
            monitoring_pre_buffer = deque(maxlen=25) 

            
            while self._is_monitoring:
                try:
                    if not self.audio_stream or not self._is_monitoring:
                        break

                    # Capture audio
                    try:
                        audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    except Exception as e:
                        if self._is_monitoring:
                            logger.warning(f"Audio read error in monitoring: {e}")
                        break
                    
                    # Mix down to mono if device is stereo
                    if getattr(self, '_actual_channels', 1) == 2:
                        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_chunk = audio_array.reshape(-1, 2)[:, 0].tobytes()
                    
                    # Determine current threshold based on whether bot is playing.
                    # Use real-time heartbeat from the hardware-backed callback.
                    is_bot_playing = False
                    if self.player and hasattr(self.player, 'last_heartbeat'):
                        # If heartbeat is within 250ms, the bot is definitely speaking
                        is_bot_playing = (time.time() - self.player.last_heartbeat < 0.25)
                    
                    current_threshold = playing_threshold if is_bot_playing else base_threshold
                    
                    # ---------------------------------------------------------
                    # FALLBACK: Digital Ducking
                    # If we don't have hardware AEC, we digitally lower the 
                    # microphone volume when the bot is speaking. This helps
                    # the VAD ignore the echo of the bot's own voice.
                    # ---------------------------------------------------------
                    vad_chunk = audio_chunk
                    if not (HAS_SPEEX and self.echo_canceller) and is_bot_playing:
                        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                        audio_array *= 0.1 # Aggressive Ducking: Reduce sensitivity to 10%
                        vad_chunk = audio_array.astype(np.int16).tobytes()
                    
                    # AEC (Hardware/Library)
                    elif HAS_SPEEX and self.echo_canceller:
                        try:
                            ref_chunk = self.reference_queue.get_nowait()
                            try:
                                # Identified as .process() via SpeexInspect
                                vad_chunk = self.echo_canceller.process(audio_chunk, ref_chunk)
                            except (AttributeError, Exception):
                                vad_chunk = self.echo_canceller.run(audio_chunk, ref_chunk)
                        except queue.Empty:
                            pass
                        except Exception as e:
                            if not self._aec_debug_done:
                                logger.error(f"AEC Error (Monitoring): {e}")
                                self._aec_debug_done = True
                            
                            if is_bot_playing:
                                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                                audio_np *= 0.1
                                vad_chunk = audio_np.astype(np.int16).tobytes()
                    
                    is_speech = self.vad.is_speech(vad_chunk, self.sample_rate)
                    
                    if is_speech:
                        consecutive_speech_chunks += 1
                        
                        # Energy check: ensure the speech isn't just low-level noise
                        samples = np.frombuffer(vad_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
                        
                        # Dynamic Energy Threshold: 
                        # If bot is playing, we need slightly higher signal to cut through residual echo.
                        # If AEC is active, we only need 1.2x. If ducking (no AEC), we need 1.5x.
                        has_aec = (HAS_SPEEX and self.echo_canceller)
                        multiplier = 1.2 if has_aec else 1.5
                        current_energy_req = energy_threshold_base * (multiplier if is_bot_playing else 1.0)
                        
                        if rms < current_energy_req:
                            consecutive_speech_chunks = 0
                        
                        if consecutive_speech_chunks >= current_threshold:
                            logger.info(f"🎤 Interruption detected! (After {consecutive_speech_chunks} chunks, threshold={current_threshold}, RMS={rms:.0f})")
                            
                            # FINAL SAFETY: If we are ducking echo (no AEC), ensure the handoff 
                            # is actually loud enough to be human speech, not just echo leakage.
                            if is_bot_playing and not (HAS_SPEEX and self.echo_canceller) and rms < 1200:
                                logger.warning(f"🔇 Interruption rejected - RMS {rms:.0f} too low for non-AEC handoff")
                                consecutive_speech_chunks = 0
                                continue

                            # Stop monitoring IMMEDIATELY 
                            self._is_monitoring = False
                            
                            # CATCH THE SPEECH: Save the pre-buffer and current segment
                            # If bot was playing, we skip the first few chunks of the pre-buffer
                            # as they likely contain bot echo that AEC is still clearing.
                            skip_chunks = 5 if is_bot_playing else 0
                            handoff_list = list(monitoring_pre_buffer)[skip_chunks:]
                            self.interruption_buffer = handoff_list + [vad_chunk]
                            
                            self._interrupted_this_turn = True
                            
                            if self._interruption_callback:
                                self._interruption_callback()
                            consecutive_speech_chunks = 0 # Reset
                    else:
                        consecutive_speech_chunks = 0
                        monitoring_pre_buffer.append(vad_chunk)
                        
                except Exception as e:
                    # Don't spam errors for small glitches, but exit if fatal
                    if "Unanticipated host error" in str(e):
                        logger.error(f"Fatal ALSA error in monitoring: {e}")
                        break
                    time.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self._is_monitoring = False

    def provide_reference_audio(self, audio_data: bytes) -> None:
        """
        Provide reference audio (what's playing) for echo cancellation.
        Accumulates data until it hits chunk_size for AEC.
        """
        # Trigger immediate correlation check if this is the start of a turn
        if time.time() - self.last_reference_time > 1.0:
            self._last_correlation_time = 0 # Force immediate scan in always_streaming
            
        self.last_reference_time = time.time()
        
        if not self.echo_canceller:
            return
            
        # 1. Add to Ring Buffer with Wall-Clock Timestamp
        with self.ref_lock:
            # Shared time baseline with VAD
            self.ref_ring_timestamp = time.time()
            
            # Efficiently write to circular buffer
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            n = len(audio_np)
            if self.ref_ring_pos + n <= len(self.ref_ring_buffer):
                self.ref_ring_buffer[self.ref_ring_pos:self.ref_ring_pos + n] = audio_np
            else:
                # Wrap around
                space = len(self.ref_ring_buffer) - self.ref_ring_pos
                self.ref_ring_buffer[self.ref_ring_pos:] = audio_np[:space]
                self.ref_ring_buffer[:n - space] = audio_np[space:]
            
            self.ref_ring_pos = (self.ref_ring_pos + n) % len(self.ref_ring_buffer)

    def ensure_stream_open(self) -> None:
        """Helper to open or restart the audio stream."""
        if not self.pa:
            self.pa = pyaudio.PyAudio()
            self._owns_pa = True
            
        if not self.audio_stream:
            # Multi-Stream Config: INPUT ONLY
            # PulseAudio handles mixing with the separate Output stream
            configs = [
                (1, self.input_device_index),
                (1, None)
            ]
            
            self._actual_channels = 1
            for channels, in_idx in configs:
                try:
                    self.audio_stream = self.pa.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size,
                        input_device_index=in_idx
                    )
                    self._actual_channels = channels
                    logger.info(f"🎤 Input stream opened: {channels} ch, in={in_idx}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to open input stream (ch={channels}, in={in_idx}): {e}")
                    continue
            
            if not self.audio_stream:
                raise RuntimeError("Failed to open INPUT audio stream.")
            self._owns_stream = True
        elif not self.audio_stream.is_active():
            self.audio_stream.start_stream()
    
    def reset_idle_timer(self):
        """Reset the idle timer (e.g., when bot is speaking)."""
        self.last_speech_time = time.time()
    def pause(self) -> None:
        """Alias for release_device to maintain API compatibility."""
        self.release_device()
        
    def release_device(self) -> None:
        """Fully close the audio stream to release hardware locks."""
        if self.audio_stream:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                logger.info("🎤 Microphone stream closed to free device")
            except Exception as e:
                logger.warning(f"Error closing microphone stream: {e}")
            self.audio_stream = None
            
            # Settle time
            time.sleep(0.3)

    def cleanup(self) -> None:
        """Clean up audio resources. Ensure device is released."""
        self.release_device()

        if self._owns_pa and self.pa:
            try:
                self.pa.terminate()
                logger.info("PyAudio terminated (owned by VAD)")
            except:
                pass
            self.pa = None
        
        elif not self._owns_pa:
            logger.info("♻️  Keeping shared PyAudio instance alive")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


if __name__ == "__main__":
    # Test continuous VAD
    print("Testing Continuous VAD...")
    print("Have a multi-turn conversation. Will end after 10s of silence.\n")
    
    with ContinuousVADCapture(idle_timeout_seconds=10) as continuous_vad:
        turn = 1
        for audio_data in continuous_vad.listen_continuous():
            if audio_data:
                print(f"\nTurn {turn}: Captured {len(audio_data)} bytes")
                turn += 1
        
        print("\nConversation ended!")
