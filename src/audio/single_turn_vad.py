"""
Voice Activity Detection (VAD) based audio capture.
Captures student speech after wake word detection using WebRTC VAD.
Captures ONE utterance after a wake word is detected
"""

import os
import logging
import time
from typing import Optional
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


class VADAudioCapture:
    """Voice Activity Detection based audio capture."""
    
    def __init__(
        self,
        sample_rate: int = None,
        frame_duration_ms: int = None,
        vad_aggressiveness: int = None,
        silence_timeout_ms: int = None,
        input_device_index: Optional[int] = None
    ):
        """
        Initialize VAD audio capture.
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (must be 10, 20, or 30)
            vad_aggressiveness: VAD aggressiveness (0-3, 3 is most aggressive)
            silence_timeout_ms: Milliseconds of silence before stopping capture
            input_device_index: Index of audio input device
        """
        self.sample_rate = sample_rate or int(os.getenv('SAMPLE_RATE', 16000))
        self.frame_duration_ms = frame_duration_ms or int(os.getenv('VAD_FRAME_DURATION_MS', 20))
        self.vad_aggressiveness = vad_aggressiveness or int(os.getenv('VAD_AGGRESSIVENESS', 3))
        self.silence_timeout_ms = silence_timeout_ms or int(os.getenv('SILENCE_TIMEOUT_MS', 1500))
        self.input_device_index = input_device_index or int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', 1))
        
        # Validate sample rate for WebRTC VAD
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000. Got {self.sample_rate}")
        
        # Validate frame duration for WebRTC VAD
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms. Got {self.frame_duration_ms}")
        
        # Calculate chunk size based on frame duration
        # WebRTC VAD requires: samples = (sample_rate * frame_duration_ms) / 1000
        self.chunk_size = int((self.sample_rate * self.frame_duration_ms) / 1000)
        
        logger.info(f"VAD initialized: {self.sample_rate}Hz, {self.frame_duration_ms}ms frames, "
                   f"chunk_size={self.chunk_size} samples ({self.chunk_size * 2} bytes)")
        
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.pa = None
        self.audio_stream = None
    
    def capture_speech(self) -> bytes:
        """
        Capture speech from microphone using VAD.
        
        Returns:
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
            
            logger.info("Listening for speech...")
            
            frames = []
            speech_started = False
            silence_start = None
            
            while True:
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Check if speech is present
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    if not speech_started:
                        logger.info("Speech detected, recording...")
                        speech_started = True
                    
                    frames.append(audio_chunk)
                    silence_start = None  # Reset silence timer
                
                elif speech_started:
                    # We were recording, now silence
                    frames.append(audio_chunk)  # Include some silence for natural cutoff
                    
                    if silence_start is None:
                        silence_start = time.time()
                    
                    # Check if silence timeout exceeded
                    silence_duration_ms = (time.time() - silence_start) * 1000
                    if silence_duration_ms >= self.silence_timeout_ms:
                        logger.info(f"Speech ended (silence: {silence_duration_ms:.0f}ms)")
                        break
            
            # Combine all frames
            audio_data = b''.join(frames)
            logger.info(f"Captured {len(audio_data)} bytes of audio")
            
            return audio_data
        
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            raise
        
        finally:
            self.cleanup()
    
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
    # Test VAD capture
    print("Testing VAD audio capture...")
    print("Speak after the prompt, then pause for silence detection.\n")
    
    input("Press Enter to start recording...")
    
    with VADAudioCapture() as vad_capture:
        audio_data = vad_capture.capture_speech()
        print(f"\nCaptured {len(audio_data)} bytes of audio")
        
        # Optionally save to file
        save = input("\nSave audio to file? (y/n): ")
        if save.lower() == 'y':
            with open('test_capture.raw', 'wb') as f:
                f.write(audio_data)
            print("Saved to test_capture.raw")
