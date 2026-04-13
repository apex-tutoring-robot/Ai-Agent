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
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class WakeWordDetector:
    """openWakeWord-based wake word detector."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        threshold: float = None,
        vad_threshold: float = None,
        enable_speex_noise_suppression: bool = True,
        input_device_index: Optional[int] = None,
        pa: Optional[pyaudio.PyAudio] = None
    ):
        """
        Initialize wake word detector.
        
        Args:
            model_path: Path to custom .tflite or .onnx model file (optional)
            model_name: Name of pre-trained model to use (e.g., 'alexa', 'hey_jarvis')
                       If not specified and model_path is None, loads all pre-trained models
            threshold: Detection threshold 0.0-1.0 (default: 0.5, lower = more sensitive)
            vad_threshold: VAD threshold 0.0-1.0 for noise reduction (default: 0, disabled)
            enable_speex_noise_suppression: Enable Speex noise suppression (Raspberry Pi)
            input_device_index: Index of audio input device
        """
        self.model_path = model_path or os.getenv('WAKE_WORD_MODEL_PATH', '')
        self.model_name = model_name or os.getenv('WAKE_WORD_MODEL', '')
        self.threshold = threshold if threshold is not None else float(os.getenv('WAKE_WORD_THRESHOLD', 0.5))
        self.vad_threshold = vad_threshold if vad_threshold is not None else float(os.getenv('WAKE_WORD_VAD_THRESHOLD', 0))
        self.enable_speex = enable_speex_noise_suppression
        self.input_device_index = input_device_index if input_device_index is not None else int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', 1))
        
        self.model = None
        self.audio_stream = None
        self.pa = pa
        self._owns_pa = (pa is None)
        self._is_running = False
        self.last_detection_time = 0  # For debouncing multiple detections
        
        # openWakeWord requirements: 16kHz, 16-bit PCM, mono
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms frames (optimal for openWakeWord)
        
        logger.info(f"Wake word detector initialized: model='{self.model_name or 'all'}', "
                   f"threshold={self.threshold}, vad={self.vad_threshold}, speex={self.enable_speex}")
    
    def start(self, callback: Callable[[], None]) -> None:
        """
        Start listening for wake word.
        
        Args:
            callback: Function to call when wake word is detected
        """
        try:
            # Initialize openWakeWord model
            logger.info("Loading openWakeWord model...")
            
            # Determine model paths
            if self.model_path and os.path.exists(self.model_path):
                # Load specific custom model file
                model_paths = [self.model_path]
                logger.info(f"Loading custom model: {self.model_path}")
            else:
                # Load all pre-trained models (will filter by name during prediction)
                model_paths = []
                logger.info("Loading all pre-trained models...")
            
            # Try to enable Speex if requested (only works on Linux with speexdsp_ns installed)
            try:
                self.model = Model(
                    wakeword_model_paths=model_paths,
                    # inference_framework='onnx',  # Use ONNX (works on both Windows and Pi)
                    enable_speex_noise_suppression=self.enable_speex,
                    vad_threshold=self.vad_threshold
                )
                if self.enable_speex:
                    logger.info("✓ Speex noise suppression enabled")
            except ModuleNotFoundError as e:
                if 'speexdsp_ns' in str(e):
                    logger.warning("Speex not installed, falling back to basic mode")
                    self.model = Model(
                        wakeword_model_paths=model_paths,
                        # inference_framework='onnx',
                        enable_speex_noise_suppression=False,
                        vad_threshold=self.vad_threshold
                    )
                else:
                    raise
            
            logger.info(f"✓ Models loaded: {list(self.model.models.keys())}")
            
            # Validate model name if using pre-trained
            if self.model_name and self.model_name not in self.model.models:
                logger.warning(f"Model '{self.model_name}' not found. "
                             f"Available: {list(self.model.models.keys())}")
                logger.info(f"Will respond to any wake word")
                self.model_name = None
            
            # Initialize PyAudio if not provided
            if not self.pa:
                self.pa = pyaudio.PyAudio()
                self._owns_pa = True
                logger.info("PyAudio initialized (owned by WakeWordDetector)")
            else:
                logger.info("♻️  Reusing shared PyAudio instance in WakeWordDetector")
            
            # Open audio stream with robust fallback (Channels 1 -> 2 -> Default Device)
            self.audio_stream = None
            self._actual_channels = 1
            
            # List of (channels, device_index) to try
            configs_to_try = [
                (1, self.input_device_index),
                (2, self.input_device_index),
                (1, None), # System Default
                (2, None)  # System Default Stereo
            ]
            
            for channels, dev_index in configs_to_try:
                try:
                    self.audio_stream = self.pa.open(
                        rate=self.sample_rate,
                        channels=channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.chunk_size,
                        input_device_index=dev_index
                    )
                    self._actual_channels = channels
                    logger.info(f"✓ Audio stream opened: {channels} channels, device index: {dev_index}")
                    break
                except Exception as e:
                    logger.debug(f"Failed configuration ({channels} channels, index {dev_index}): {e}")
                    continue
            
            if not self.audio_stream:
                raise RuntimeError("Could not open audio input stream after all fallbacks failed.")
            
            logger.info(f"Wake word detection started (device index: {self.input_device_index})")
            if self.model_name:
                logger.info(f"Listening for '{self.model_name}' (threshold: {self.threshold})")
            else:
                logger.info(f"Listening for any wake word (threshold: {self.threshold})")
            
            if self.vad_threshold > 0:
                logger.info(f"VAD filtering enabled (threshold: {self.vad_threshold})")
            
            self._is_running = True
            
            # Listen loop
            while self._is_running:
                # Read audio chunk
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert bytes to numpy array (int16)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Mix down to mono if device is stereo
                if self._actual_channels == 2:
                    audio_array = audio_array.reshape(-1, 2)[:, 0]
                
                # Get predictions from model
                predictions = self.model.predict(audio_array)
                
                # Filter by model name if specified
                current_time = time.time()
                cooldown_period = 2.0  # Don't trigger again within 2 seconds
                
                if self.model_name:
                    # Check only the specified model
                    if self.model_name in predictions:
                        score = predictions[self.model_name]
                        if score >= self.threshold:
                            # Check cooldown to prevent multiple detections from same utterance
                            if current_time - self.last_detection_time >= cooldown_period:
                                logger.info(f"✓ Wake word detected! (model: {self.model_name}, score: {score:.3f})")
                                self.last_detection_time = current_time
                                callback()
                else:
                    # Check all models if no specific model name
                    for model_name, score in predictions.items():
                        if score >= self.threshold:
                            # Check cooldown to prevent multiple detections from same utterance
                            if current_time - self.last_detection_time >= cooldown_period:
                                logger.info(f"✓ Wake word detected! (model: {model_name}, score: {score:.3f})")
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
    
    def get_audio_stream(self):
        """Get the active audio stream for reuse."""
        return self.audio_stream
    
    def get_pyaudio_instance(self):
        """Get the PyAudio instance for reuse."""
        return self.pa
    
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
