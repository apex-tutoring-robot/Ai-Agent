"""
Wake Word Detection using openWakeWord.
Continuously listens for wake words with low CPU usage.
No API key required - fully open-source.
"""

import os
import logging
import time
import numpy as np
from typing import Callable, Optional
import pyaudio
from openwakeword.model import Model
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class WakeWordDetector:
    """openWakeWord-based wake word detector."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: float = None,
        vad_threshold: float = None,
        input_device_index: Optional[int] = None,
        pa: Optional[pyaudio.PyAudio] = None
    ):
        """
        Initialize wake word detector.

        Args:
            model_name: Name of pre-trained model to use (e.g., 'alexa', 'hey_jarvis')
                       If not specified, loads all pre-trained models
            threshold: Detection threshold 0.0-1.0 (default: 0.5, lower = more sensitive)
            vad_threshold: VAD threshold 0.0-1.0 for noise reduction (default: 0, disabled)
            input_device_index: PyAudio input device index (None = system default)
        """
        self.model_name = model_name or os.getenv('WAKE_WORD_MODEL', '')
        self.threshold = threshold if threshold is not None else float(os.getenv('WAKE_WORD_THRESHOLD', 0.5))
        self.vad_threshold = vad_threshold if vad_threshold is not None else float(os.getenv('WAKE_WORD_VAD_THRESHOLD', 0))
        self.input_device_index = input_device_index
        
        self.model = None
        self.audio_stream = None
        self._is_running = False
        self.last_detection_time = 0  # For debouncing multiple detections
        
        # openWakeWord requirements: 16kHz, 16-bit PCM, mono
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms frames (optimal for openWakeWord)
        
        logger.info(f"Wake word detector initialized: model='{self.model_name or 'all'}', "
                   f"threshold={self.threshold}, vad={self.vad_threshold}")
    
    def start(self, callback: Callable[[], None]) -> None:
        """
        Start listening for wake word.
        
        Args:
            callback: Function to call when wake word is detected
        """
        try:
            logger.info("Loading openWakeWord model...")

            # Load all built-in models, then filter to self.model_name later
            self.model = Model(
                wakeword_model_paths=[],
                enable_speex_noise_suppression=False,
                vad_threshold=self.vad_threshold
            )
            logger.info(f"✓ Models loaded: {list(self.model.models.keys())}")

            if self.model_name and self.model_name not in self.model.models:
                logger.warning(
                    f"Model '{self.model_name}' not found. Available: {list(self.model.models.keys())}"
                )
                self.model_name = None

            # Always use a dedicated PyAudio instance here
            self.pa = pyaudio.PyAudio()
            self._owns_pa = True
            logger.info("PyAudio initialized (owned by WakeWordDetector)")

            # Open input stream, fall back to default device on failure
            stream_kwargs = dict(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            try:
                self.audio_stream = self.pa.open(**stream_kwargs, input_device_index=self.input_device_index)
            except Exception as e:
                logger.warning(f"Failed to open with device index {self.input_device_index}, using default mic")
                self.audio_stream = self.pa.open(**stream_kwargs)
            logger.info(f"✓ Audio stream opened on input device index {self.input_device_index}")

            self._is_running = True
            logger.info("Wake word detection started")

            while self._is_running:
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                predictions = self.model.predict(audio_array)
    
                current_time = time.time()
                cooldown_period = 2.0

                if self.model_name:
                    if self.model_name in predictions:
                        score = predictions[self.model_name]
                        if score >= self.threshold and (current_time - self.last_detection_time) >= cooldown_period:
                            logger.info(
                                f"✓ Wake word detected! (model: {self.model_name}, score: {score:.3f})"
                            )
                            self.last_detection_time = current_time
                            callback()
                else:
                    for model_name, score in predictions.items():
                        if score >= self.threshold and (current_time - self.last_detection_time) >= cooldown_period:
                            logger.info(
                                f"✓ Wake word detected! (model: {model_name}, score: {score:.3f})"
                            )
                            self.last_detection_time = current_time
                            callback()
                            break

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            raise
        finally:
            self.stop()
    
    def pause(self) -> None:
        """
        Pause wake word detection but keep audio stream alive.
        This allows the stream to be reused by continuous VAD without recreation.
        """
        self._is_running = False
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                logger.info("Wake word detection paused (stream stopped but kept alive)")
            except Exception as e:
                logger.warning(f"Error stopping stream during pause: {e}")
    
    def stop(self) -> None:
        """Stop wake word detection and clean up resources."""
        self._is_running = False
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pa:
            try:
                if self._owns_pa:
                    self.pa.terminate()
                    logger.info("PyAudio terminated (owned by WakeWordDetector)")
                else:
                    logger.info("♻️  Keeping shared PyAudio instance alive")
            except:
                pass
            self.pa = None
        
        if self.model:
            # openWakeWord doesn't have explicit cleanup, but we can release the reference
            self.model = None
        
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
    # list_audio_devices()
    
    # Test wake word detection
    def on_wake_word():
        print(">>> Wake word detected! <<<")
    
    detector = WakeWordDetector()
    try:
        print("\nListening for wake word... (Press Ctrl+C to stop)")
        print("Will respond to any pre-trained wake word\n")
        detector.start(on_wake_word)
    except KeyboardInterrupt:
        print("\nStopping...")
        detector.stop()
