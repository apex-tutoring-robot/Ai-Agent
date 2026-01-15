"""
Continuous Voice Activity Detection for conversation mode.
Listens continuously with idle timeout for multi-turn conversations.
Supports multiple utterances in a conversation session
"""

import os
import logging
import time
from typing import Optional, Generator
import pyaudio
import webrtcvad
from dotenv import load_dotenv

# Suppress ALSA warnings
try:
    from . import suppress_alsa
except ImportError:
    pass

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class ContinuousVADCapture:
    """Continuous voice activity detection with idle timeout for conversation mode."""
    
    def __init__(
        self,
        sample_rate: int = None,
        frame_duration_ms: int = None,
        vad_aggressiveness: int = None,
        silence_timeout_ms: int = None,
        idle_timeout_seconds: int = None,
        input_device_index: Optional[int] = None
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
        """
        self.sample_rate = sample_rate or int(os.getenv('SAMPLE_RATE', 16000))
        self.frame_duration_ms = frame_duration_ms or int(os.getenv('VAD_FRAME_DURATION_MS', 20))
        self.vad_aggressiveness = vad_aggressiveness or int(os.getenv('VAD_AGGRESSIVENESS', 2))
        self.silence_timeout_ms = silence_timeout_ms or int(os.getenv('SILENCE_TIMEOUT_MS', 2000))
        self.idle_timeout_seconds = idle_timeout_seconds or int(os.getenv('CONVERSATION_IDLE_TIMEOUT_SECONDS', 10))
        self.input_device_index = input_device_index or int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', 1))
        
        # Validate sample rate for WebRTC VAD
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000. Got {self.sample_rate}")
        
        # Validate frame duration for WebRTC VAD
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms. Got {self.frame_duration_ms}")
        
        # Calculate chunk size based on frame duration
        self.chunk_size = int((self.sample_rate * self.frame_duration_ms) / 1000)
        
        logger.info(f"Continuous VAD initialized: {self.sample_rate}Hz, {self.frame_duration_ms}ms frames, "
                   f"idle timeout={self.idle_timeout_seconds}s")
        
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.pa = None
        self.audio_stream = None
        self.last_speech_time = None
        
        # Latency tracking
        self.speech_start_time = None  # When user starts speaking
        self.silence_detected_time = None  # When user stops speaking
    
    def listen_continuous(self) -> Generator[bytes, None, None]:
        """
        Listen continuously for speech, yielding audio segments.
        Ends after idle_timeout_seconds of no speech.
        
        Yields:
            Raw audio bytes of captured speech
        """
        try:
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pa.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index
            )
            
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
            initial_speech_timeout = 1.5  # Wait 1.5s for user to START speaking
            post_speech_silence_timeout_ms = 800  # End utterance after 800ms of silence
            
            listen_start = time.time()
            
            while True:
                # Give up if no speech detected within initial timeout
                if not speech_started and (time.time() - listen_start) > initial_speech_timeout:
                    return None
                
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
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
            # Cleanup any existing resources first
            if self.audio_stream:
                self.cleanup()
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pa.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index
            )

            speech_started = False
            silence_start = None
            
            # Two separate timeouts for different purposes
            initial_speech_timeout = 1.5  # Wait 1.5s for user to START speaking
            post_speech_silence_timeout = 0.8  # End utterance after 800ms of silence
            
            listen_start = time.time()
            
            logger.info("🎤 Streaming audio chunks...")
            
            chunk_count = 0
            speech_chunk_count = 0
            silence_chunk_count = 0
            
            # Debouncing: count consecutive speech chunks to avoid noise triggering
            consecutive_speech_chunks = 0
            min_speech_chunks_to_cancel_silence = 10  # Need 10 consecutive chunks (200ms) to cancel silence
            
            while True:
                # Give up if no speech detected within initial timeout
                if not speech_started and (time.time() - listen_start) > initial_speech_timeout:
                    logger.info("No speech detected, ending stream")
                    yield None  # Signal end
                    return
                
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                chunk_count += 1
                
                # Check if speech is present
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    speech_chunk_count += 1
                    consecutive_speech_chunks += 1
                    
                    if not speech_started:
                        # LATENCY: Track when user starts speaking (for STT latency)
                        self.speech_start_time = time.perf_counter()
                        logger.info("💬 Speech detected - streaming chunks...")
                        speech_started = True
                        yield audio_chunk
                        silence_start = None
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
                    
                    if speech_started:
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
            # Always cleanup resources
            self.cleanup()
    
    def reset_idle_timer(self):
        """Reset the idle timer (e.g., when bot is speaking)."""
        self.last_speech_time = time.time()
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.pa:
            self.pa.terminate()
            self.pa = None
    
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
