"""
Audio playback module for Raspberry Pi using sounddevice.
PipeWire / PulseAudio compatible.
"""

import os
import logging
import time
import queue
import threading
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# Must be imported before PyAudio initialises to suppress ALSA noise
from audio import suppress_alsa  # noqa: F401

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Thread-safe audio player with streaming support."""
    
    def __init__(
        self,
        sample_rate: int = 16000,  # Azure TTS outputs 16kHz audio
        channels: int = None,
        output_device_index: Optional[int] = None,
        pa: Optional['pyaudio.PyAudio'] = None,
        on_audio_played: Optional[callable] = None,
        on_level: Optional[callable] = None
    ):
        """
        Initialize audio player.
        
        Args:
            sample_rate: Audio sample rate (default 16000 for Azure TTS)
            channels: Number of audio channels
            output_device_index: Index of audio output device
            pa: Optional shared PyAudio instance
            on_audio_played: Optional callback for AEC reference audio
            on_level: Optional callback for lip sync RMS level (0.0 to 1.0)
        """
        self.sample_rate = sample_rate  # Use 16000 to match Azure TTS
        self.channels = channels or int(os.getenv('CHANNELS', 1))
        self.output_device_index = output_device_index
        self.on_level = on_level
        
        logger.info(f"AudioPlayer initialized: {self.sample_rate}Hz, {self.channels} channel(s), device {self.output_device_index}")
        
        self.pa = pa
        self._owns_pa = (pa is None)
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self._is_playing = False
        self._stop_event = threading.Event()
        self.on_audio_played = on_audio_played
        self.has_fatal_error = False
        self._owns_stream = True
        
        # Latency tracking and real-time state
        self.first_token_time = None
        self.first_audio_played = False
        self.last_heartbeat = 0.0  # Real-time timestamp of local playback

    def play_audio(self, audio_data: bytes) -> None:
        try:
            import pyaudio
            if not self.pa:
                self.pa = pyaudio.PyAudio()
            
            # If we don't have a stream, open a managed one
            if not self.audio_stream:
                self.audio_stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_index
                )
                self._owns_stream = True
            
            # Play audio
            self.audio_stream.write(audio_data)
        
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise
    
    # Frames per callback buffer: 320 frames @ 16kHz = 20ms
    # Matches VAD and Speex AEC frame size for perfect synchronization.
    _FRAMES_PER_BUFFER = 320

    def start_streaming(self, output_device_index: int = None) -> None:
        """
        Start the audio stream for playback using PortAudio callback mode.
        No blocking write() calls — PortAudio calls _audio_callback every 64ms.
        """
        if self._is_playing:
            logger.warning("Streaming already active")
            return

        self._stop_event.clear()
        self._is_playing = True
        self.first_audio_played = False
        self.has_fatal_error = False
        self._callback_buf = bytearray()  # leftover bytes between callback calls

        try:
            import pyaudio
            if not self.pa:
                self.pa = pyaudio.PyAudio()
                self._owns_pa = True

            target_device_index = output_device_index if output_device_index is not None else self.output_device_index

            # Close any stale stream (safe here — no worker thread is running yet)
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

            # Open in CALLBACK mode — PortAudio drives timing, no Python write() blocks
            self.audio_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=target_device_index,
                frames_per_buffer=self._FRAMES_PER_BUFFER,
                stream_callback=self._audio_callback,
            )
            self.audio_stream.start_stream()
            self.playback_thread = None  # No worker thread — callback drives everything

            logger.info(f"🔊 Audio stream opened (callback mode): {self.sample_rate}Hz, "
                        f"buffer={self._FRAMES_PER_BUFFER}, device={target_device_index}")
            logger.info("Streaming playback started")

        except Exception as e:
            logger.error(f"Error starting streaming playback: {e}")
            self._is_playing = False
            if self.audio_stream:
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PortAudio callback — runs in PortAudio's internal C thread every 64ms.

        Rules:
        - Must be fast and non-blocking (no Python I/O, no locks that might block)
        - Queue.get_nowait() is safe (very brief lock, never sleeps)
        - Returns (audio_bytes, flag) where flag is paContinue / paComplete / paAbort

        Stop semantics:
        - immediate=True  → _stop_event is set → return paAbort  (silent, instant)
        - graceful        → None sentinel in queue → return paComplete (drain then stop)
        """
        import pyaudio
        required = frame_count * 2  # mono int16: 2 bytes per sample

        # Immediate abort path — check first, fastest exit
        if self._stop_event.is_set():
            self._is_playing = False
            return (b'\x00' * required, pyaudio.paAbort)

        # Update heartbeat: we are actively processing audio for the speaker
        self.last_heartbeat = time.time()

        output = self._callback_buf  # pick up leftover from previous call

        while len(output) < required:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                # No data yet
                if not self._is_playing:
                    # Done — fill remaining with silence and complete
                    output.extend(b'\x00' * (required - len(output)))
                    return (bytes(output), pyaudio.paComplete)
                # Still playing but queue is momentarily empty — output silence (underrun guard)
                output.extend(b'\x00' * (required - len(output)))
                self._callback_buf = bytearray()
                return (bytes(output), pyaudio.paContinue)

            if chunk is None:
                # Graceful-stop sentinel
                self._is_playing = False
                output.extend(b'\x00' * (required - len(output)))
                self._callback_buf = bytearray()
                return (bytes(output), pyaudio.paComplete)

            # TTFAS latency tracking
            if not self.first_audio_played and self.first_token_time:
                ttfas = time.perf_counter() - self.first_token_time
                logger.info(f"⏱️  TTFAS (Time To First Audio Spoken): {ttfas:.3f}s")
                self.first_audio_played = True

            output.extend(chunk)

        # We have at least `required` bytes — save the overflow for next call
        original_output = bytes(output[:required])
        self._callback_buf = output[required:]

        # CRITICAL AEC FIX: Notify VAD ONLY about the EXACT samples being played NOW.
        # This ensures the Reference Buffer in AEC is perfectly aligned with the Speakers.
        if self.on_audio_played and original_output != b'\x00' * len(original_output):
            try:
                self.on_audio_played(original_output)
            except Exception as e:
                # Use a flag to avoid log spamming if VAD is not ready
                pass
                
        # --- LIP SYNC RMS ---
        if self.on_level:
            if original_output == b'\x00' * len(original_output):
                self.on_level(0.0)
            else:
                try:
                    import numpy as np
                    a = np.frombuffer(original_output, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(a * a)) / 32768.0
                    if rms < 0.02:
                        rms = 0.0
                    level = min(rms * 8.0, 1.0)
                    self.on_level(level)
                except Exception as e:
                    logger.error(f"Error calculating RMS: {e}")

        return (original_output, pyaudio.paContinue)

    def queue_audio(self, audio_chunk: bytes) -> None:
        if not self._is_playing:
            raise RuntimeError("Streaming not started.")
        self.audio_queue.put(audio_chunk)
    
    def stop_streaming(self, immediate: bool = False) -> bool:
        """
        Stop streaming playback.

        immediate=True  → paAbort via _stop_event; audio cuts off within one callback cycle (~64ms)
        immediate=False → None sentinel into queue; PortAudio drains naturally then paComplete

        Returns True if stream stopped cleanly.
        """
        if not self._is_playing and not self.audio_stream:
            return True

        logger.info(f"Stopping playback (immediate={immediate})...")

        if immediate:
            self._is_playing = False
            # 1. Discard all queued audio
            logger.info("Clearing playback queue for immediate stop")
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            # 2. Signal callback to return paAbort on its next invocation
            self._stop_event.set()
        else:
            # Graceful: push sentinel so callback returns paComplete after draining.
            # _is_playing must remain True until the callback consumes the sentinel,
            # otherwise the callback's queue.Empty handler exits early mid-playback.
            self.audio_queue.put(None)

        # Wait for PortAudio to acknowledge the stop (polls is_active())
        # In callback mode there is NO Python blocking write() here — completely safe
        # Immediate: short timeout — paAbort fires within one callback cycle (~64ms)
        # Graceful: no deadline — the None sentinel drives paComplete naturally;
        #           forcing a timeout here would cut off long audio mid-playback.
        stream = self.audio_stream  # local ref — safe to read, we're the only closer
        if stream:
            if immediate:
                deadline = time.time() + 2.0
                while stream.is_active() and time.time() < deadline:
                    time.sleep(0.02)
            else:
                while stream.is_active():
                    time.sleep(0.02)

        # Close the stream safely
        if self.audio_stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
                # On Pi 5 / ALSA, closing the stream inside a callback-triggered
                # path can cause assertion failures. We'll stop it here and 
                # let start_streaming or cleanup handle the full close.
                # self.audio_stream.close() 
            except Exception:
                pass
            # Set to None so we know it needs reopening, but don't close() yet
            # self.audio_stream = None

        self.playback_thread = None
        if self.on_level is not None:
            self.on_level(0.0)
        logger.info("✅ Streaming playback stopped")
        return True

    def cleanup(self) -> None:
        """Clean up audio stream (but keep PyAudio instance for reuse)."""
        # Clean up stream
        try:
            if self.audio_stream:
                try:
                    if self.audio_stream.is_active():
                        self.audio_stream.stop_stream()
                except:
                    pass
                try:
                    self.audio_stream.close()
                except:
                    pass
                self.audio_stream = None
                
                # CRITICAL: Give ALSA time to fully release the device
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error cleaning up audio stream: {e}")
        
        # DON'T terminate PyAudio - reuse it for next conversation
        # Only terminate in __exit__ or explicit shutdown
    
    def shutdown(self) -> None:
        self.cleanup()
        
        # Now terminate PyAudio only if we own it
        try:
            if self._owns_pa and self.pa:
                self.pa.terminate()
                self.pa = None
                logger.info("PyAudio terminated (owned by AudioPlayer)")
            elif not self._owns_pa:
                logger.info("♻️  Keeping shared PyAudio instance alive")
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()
        self.shutdown()
