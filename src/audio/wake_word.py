"""
Wake Word Detection using Porcupine.
Continuously listens for "Hey CHIPPY" wake word with low CPU usage.
"""

import os
import struct
import logging
import threading
from typing import Callable, Optional
import pyaudio
import pvporcupine
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Porcupine-based wake word detector for 'Hey CHIPPY'."""
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        keyword_path: Optional[str] = None,
        sensitivity: float = 0.5,
        input_device_index: Optional[int] = None
    ):
        """
        Initialize wake word detector.
        
        Args:
            access_key: Picovoice access key
            keyword_path: Path to .ppn wake word model file
            sensitivity: Detection sensitivity (0.0 to 1.0)
            input_device_index: Index of audio input device
        """
        self.access_key = access_key or os.getenv('PICOVOICE_ACCESS_KEY')
        self.keyword_path = keyword_path or os.getenv('WAKE_WORD_MODEL_PATH')
        self.sensitivity = sensitivity
        self.input_device_index = input_device_index or int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', 1))
        
        self.porcupine = None
        self.audio_stream = None
        self.pa = None
        self._is_running = False
        
        if not self.access_key:
            raise ValueError("Picovoice access key not provided")
        if not self.keyword_path or not os.path.exists(self.keyword_path):
            logger.warning(f"Wake word model not found at {self.keyword_path}. "
                          "Using built-in 'porcupine' keyword for testing.")
            self.keyword_path = None
    
    def start(self, callback: Callable[[], None]) -> None:
        """
        Start listening for wake word.
        
        Args:
            callback: Function to call when wake word is detected
        """
        try:
            # Initialize Porcupine
            if self.keyword_path:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.keyword_path],
                    sensitivities=[self.sensitivity]
                )
            else:
                # Fallback to built-in keyword for testing
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=['porcupine'],
                    sensitivities=[self.sensitivity]
                )
            
            logger.info(f"Porcupine initialized with frame length: {self.porcupine.frame_length}, "
                       f"sample rate: {self.porcupine.sample_rate}")
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                input_device_index=self.input_device_index
            )
            
            logger.info(f"Wake word detection started (device index: {self.input_device_index})")
            self._is_running = True
            
            # Listen loop
            while self._is_running:
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    logger.info("Wake word detected!")
                    callback()
        
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop wake word detection and clean up resources."""
        self._is_running = False
        
        if self.audio_stream:
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.pa:
            self.pa.terminate()
            self.pa = None
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        logger.info("Wake word detection stopped")
    
    def is_running(self) -> bool:
        """Check if detector is currently running."""
        return self._is_running


def list_audio_devices():
    """Helper function to list available audio devices."""
    pa = pyaudio.PyAudio()
    print("\nAvailable Audio Devices:")
    print("-" * 60)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']}")
        print(f"   Max Input Channels: {info['maxInputChannels']}")
        print(f"   Max Output Channels: {info['maxOutputChannels']}")
        print(f"   Default Sample Rate: {info['defaultSampleRate']}")
        print()
    pa.terminate()


if __name__ == "__main__":
    # List available devices
    list_audio_devices()
    
    # Test wake word detection
    def on_wake_word():
        print(">>> Wake word detected! <<<")
    
    detector = WakeWordDetector()
    try:
        print("\nListening for wake word... (Press Ctrl+C to stop)")
        detector.start(on_wake_word)
    except KeyboardInterrupt:
        print("\nStopping...")
        detector.stop()
