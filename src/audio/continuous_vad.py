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
import threading
from typing import Generator, Optional
from dotenv import load_dotenv
import webrtcvad
import pyaudio

# Must be imported before PyAudio initialises to suppress ALSA noise
from audio import suppress_alsa  # noqa: F401

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from utils import imp_shim
    imp_shim.install_shim()
    from speexdsp import Preprocessor
    HAS_PREPROCESSOR = True
    logger.info("✅ SpeexDSP Preprocessor loaded (Denoise/AGC/Dereverb)")
except Exception:
    HAS_PREPROCESSOR = False
    Preprocessor = None
    logger.warning("⚠️  SpeexDSP Preprocessor not found — noise suppression disabled")


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
        self.input_device_index = input_device_index
        
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
        
        # Real-time state from AudioPlayer
        self.player = player
        
        self._is_monitoring = False
        self._interruption_callback = None
        self._interrupted_this_turn = False
        self.interruption_buffer = []
        
        # ── Playback-aware gating (mute-the-pipe) ──
        self._is_bot_playing = False           # Set by main.py via set_playback_state()
        self._playback_start_time = 0.0        # Reset each time playback starts
        self._barge_in_event = None            # threading.Event from JarvisBot (set on barge-in)
        self._barge_in_energy_threshold = int(os.getenv('BARGE_IN_ENERGY_THRESHOLD', '1500'))
        self._barge_in_consecutive_needed = int(os.getenv('BARGE_IN_CHUNKS_NEEDED', '3'))
        # WebRTC AEC needs ~1-2s to converge on a new playback session — block barge-in until then.
        self._barge_in_grace_s = float(os.getenv('BARGE_IN_GRACE_S', '2.0'))
        self._post_playback_drain_s = float(os.getenv('POST_PLAYBACK_DRAIN_MS', '400')) / 1000.0
        self._min_consecutive_to_start = int(os.getenv('MIN_SPEECH_CHUNKS_TO_START', '8'))
        self._barge_in_echo_drain_frames = int(os.getenv('BARGE_IN_ECHO_DRAIN_FRAMES', '2'))

        self.preprocessor = None

        enable_ns = os.getenv('ENABLE_SPEEX_NOISE_SUPPRESSION', 'true').lower() == 'true'
        if enable_ns and HAS_PREPROCESSOR:
            try:
                self.preprocessor = Preprocessor(self.chunk_size, self.sample_rate)
                self.preprocessor.denoise = True
                self.preprocessor.agc = True
                self.preprocessor.dereverb = True
                self.preprocessor.agc_level = 8000
                logger.info("✅ Speex Preprocessor (Denoise/AGC/Dereverb) initialized")
            except Exception as e:
                logger.warning(f"Failed to init Speex Preprocessor: {e}")
            
    def _mix_to_mono(self, chunk: bytes) -> bytes:
        if getattr(self, '_actual_channels', 1) == 2:
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            return audio_array.reshape(-1, 2)[:, 0].tobytes()
        return chunk

    def set_playback_state(self, is_playing: bool) -> None:
        """Called by main.py to inform VAD whether the bot is currently playing audio.

        When is_playing=True, always_streaming will feed silence to STT
        instead of mic audio, preventing the bot from hearing itself.
        Barge-in detection runs separately and can open the gate.
        """
        self._is_bot_playing = is_playing
        if is_playing:
            self._playback_start_time = time.time()
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
                audio_chunk = self._mix_to_mono(audio_chunk)

                # Apply Speex Preprocessor (Denoise + AGC + Dereverb)
                if self.preprocessor:
                    try:
                        audio_chunk = self.preprocessor.process(audio_chunk)
                    except Exception:
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
            if not self.audio_stream:
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
            else:
                drain_end = time.time() + self._post_playback_drain_s
                drained_chunks = 0
                if self._post_playback_drain_s > 0:
                    while time.time() < drain_end:
                        try:
                            self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                            drained_chunks += 1
                        except Exception:
                            break
                    logger.debug(f"Drained {drained_chunks} mic buffer chunks ({self._post_playback_drain_s*1000:.0f}ms) to clear echo tail")
            # -----------------------------------------------------------------
            
            listen_start = time.time()
            
            logger.info("🎤 Streaming audio chunks...")
            
            chunk_count = 0
            speech_chunk_count = 0
            silence_chunk_count = 0
            
            # Debouncing: count consecutive speech chunks to avoid noise triggering
            consecutive_speech_chunks = 0
            min_speech_chunks_to_cancel_silence = 25  # 25 chunks (500ms) to cancel mid-speech silence
            # Require consecutive VAD-positive chunks before declaring speech START.
            # Prevents echo blips that survived the drain from triggering the pipeline.
            min_consecutive_to_start = self._min_consecutive_to_start
            pending_speech_chunks = []  # Buffer chunks while waiting to confirm speech start
            
            # Reset latency tracking for this turn
            self.silence_detected_time = None
            
            # PRE-BUFFER: Keep recent chunks BEFORE confirmed speech detected.
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
                audio_chunk = self._mix_to_mono(audio_chunk)

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
                        # (ensures the consecutive-chunk threshold is truly CONTIGUOUS)
                        pending_speech_chunks = []
                        # Store in pre-buffer (circular, auto-discards old chunks)
                        pre_buffer.append(audio_chunk)
                    else:
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
        _barge_in_echo_drain = self._barge_in_echo_drain_frames
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
                audio_chunk = self._mix_to_mono(audio_chunk)

                # ════════════════════════════════════════════════════════
                # MUTE-THE-PIPE: Decide whether to yield real audio or silence
                # ════════════════════════════════════════════════════════
                if self._is_bot_playing:
                    # Bot speaking counts as activity — keep conversation alive
                    self.last_speech_time = time.time()

                    # Grace period: AEC needs time to converge on a new playback
                    # session.  Suppress barge-in completely until it has.
                    elapsed = time.time() - self._playback_start_time
                    if elapsed < self._barge_in_grace_s:
                        yield silence_chunk
                        continue

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
            consecutive_speech_chunks = 0

            while self._is_monitoring:
                try:
                    if not self.audio_stream:
                        break

                    # Capture audio
                    try:
                        audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    except Exception as e:
                        if self._is_monitoring:
                            logger.warning(f"Audio read error in monitoring: {e}")
                        break
                    
                    audio_chunk = self._mix_to_mono(audio_chunk)

                    # Determine current threshold based on whether bot is playing.
                    # Use real-time heartbeat from the hardware-backed callback.
                    is_bot_playing = False
                    if self.player and hasattr(self.player, 'last_heartbeat'):
                        # If heartbeat is within 250ms, the bot is definitely speaking
                        is_bot_playing = (time.time() - self.player.last_heartbeat < 0.25)
                    
                    current_threshold = playing_threshold if is_bot_playing else base_threshold
                    
                    is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)

                    if is_speech:
                        consecutive_speech_chunks += 1

                        # Energy check: ensure the speech isn't just low-level noise
                        samples = np.frombuffer(audio_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))

                        current_energy_req = energy_threshold_base * (1.2 if is_bot_playing else 1.0)

                        if rms < current_energy_req:
                            consecutive_speech_chunks = 0

                        if consecutive_speech_chunks >= current_threshold:
                            logger.info(f"🎤 Interruption detected! (After {consecutive_speech_chunks} chunks, threshold={current_threshold}, RMS={rms:.0f})")

                            # Stop monitoring IMMEDIATELY
                            self._is_monitoring = False

                            # CATCH THE SPEECH: Save the pre-buffer and current segment
                            # If bot was playing, we skip the first few chunks of the pre-buffer
                            # as they likely contain bot echo that AEC is still clearing.
                            skip_chunks = 5 if is_bot_playing else 0
                            handoff_list = list(monitoring_pre_buffer)[skip_chunks:]
                            self.interruption_buffer = handoff_list + [audio_chunk]
                            
                            self._interrupted_this_turn = True
                            
                            if self._interruption_callback:
                                self._interruption_callback()
                            consecutive_speech_chunks = 0 # Reset
                    else:
                        consecutive_speech_chunks = 0
                        monitoring_pre_buffer.append(audio_chunk)
                        
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
