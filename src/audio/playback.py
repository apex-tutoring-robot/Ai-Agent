"""
Audio playback module for Raspberry Pi.
Supports streaming audio playback with USB audio devices.
"""

import os
import logging
import time
import queue
import threading
from typing import Optional
import pyaudio
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Thread-safe audio player with streaming support."""
    
    def __init__(
        self,
        sample_rate: int = 16000,  # Azure TTS outputs 16kHz audio
        channels: int = None,
        output_device_index: Optional[int] = None
    ):
        """
        Initialize audio player.
        
        Args:
            sample_rate: Audio sample rate (default 16000 for Azure TTS)
            channels: Number of audio channels
            output_device_index: Index of audio output device
        """
        self.sample_rate = sample_rate  # Use 16000 to match Azure TTS
        self.channels = channels or int(os.getenv('CHANNELS', 1))
        self.output_device_index = output_device_index or int(os.getenv('AUDIO_OUTPUT_DEVICE_INDEX', 1))
        
        logger.info(f"AudioPlayer initialized: {self.sample_rate}Hz, {self.channels} channel(s), device {self.output_device_index}")
        
        self.pa = None
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self._is_playing = False
        self._stop_event = threading.Event()
    
    def play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data (blocking).
        
        Args:
            audio_data: Raw audio bytes to play
        """
        try:
            if not self.pa:
                self.pa = pyaudio.PyAudio()
            
            if not self.audio_stream or not self.audio_stream.is_active():
                self.audio_stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_index
                )
            
            # Play audio
            self.audio_stream.write(audio_data)
        
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise
    
    def start_streaming(self) -> None:
        """Start streaming playback thread."""
        if self._is_playing:
            logger.warning("Streaming already active")
            return
        
        self._stop_event.clear()
        self._is_playing = True
        
        try:
            # Initialize PyAudio only once (reuse if exists)
            if not self.pa:
                self.pa = pyaudio.PyAudio()
                logger.info("PyAudio initialized")
            
            # Close old stream if exists
            if self.audio_stream:
                try:
                    if self.audio_stream.is_active():
                        self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception as e:
                    logger.warning(f"Error closing old stream: {e}")
                self.audio_stream = None
            
            # Small delay to let ALSA settle
            time.sleep(0.1)
            
            # Use a larger buffer to prevent underruns
            chunk_size = 1024
            
            # Open new audio stream
            self.audio_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=chunk_size
            )
            
            logger.info(f"Audio stream opened: {self.sample_rate}Hz, buffer={chunk_size}")
            
            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
            
            logger.info("Streaming playback started")
        
        except Exception as e:
            logger.error(f"Error starting streaming playback: {e}")
            self._is_playing = False
            # Don't call cleanup here - preserve PyAudio instance
            if self.audio_stream:
                try:
                    self.audio_stream.close()
                except:
                    pass
                self.audio_stream = None
            raise
    
    def _playback_worker(self) -> None:
        """Worker thread for streaming playback."""
        try:
            while self._is_playing and not self._stop_event.is_set():
                try:
                    # Get audio chunk from queue (timeout to check stop event)
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    if audio_chunk is None:
                        # None signals end of stream
                        break
                    
                    # Play the chunk with error handling
                    try:
                        self.audio_stream.write(audio_chunk)
                        self.audio_queue.task_done()
                    except Exception as write_error:
                        # PyAudio write errors can be fatal, log and continue
                        logger.error(f"Error writing audio chunk: {write_error}")
                        self.audio_queue.task_done()
                        # If it's a critical error, stop playback
                        if "Unanticipated host error" in str(write_error):
                            logger.error("Critical audio error detected, stopping playback")
                            break
                
                except queue.Empty:
                    continue
        
        except Exception as e:
            logger.error(f"Error in playback worker: {e}")
        
        finally:
            logger.info("Playback worker stopped")
    
    def queue_audio(self, audio_chunk: bytes) -> None:
        """
        Add audio chunk to playback queue.
        
        Args:
            audio_chunk: Audio data to queue for playback
        """
        if not self._is_playing:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
        
        self.audio_queue.put(audio_chunk)
    
    def stop_streaming(self) -> None:
        """Stop streaming playback and clean up."""
        if not self._is_playing:
            return
        
        # Wait for the queue to be fully processed before stopping
        logger.info("Waiting for audio queue to finish...")
        
        # Use a loop with timeout instead of indefinite join
        max_wait = 10.0  # Maximum 10 seconds wait
        start_time = time.time()
        
        while not self.audio_queue.empty():
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.warning(f"Audio queue didn't empty after {max_wait}s, forcing cleanup")
                break
            time.sleep(0.1)
        
        # Signal end of stream
        self.audio_queue.put(None)
        self._stop_event.set()
        self._is_playing = False
        
        # Wait for playback thread to finish
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
            if self.playback_thread.is_alive():
                logger.warning("Playback thread didn't stop, continuing anyway")
            self.playback_thread = None
        
        # Clear any remaining items in queue
        cleared = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        
        if cleared > 0:
            logger.warning(f"Cleared {cleared} unprocessed audio chunks")
        
        self.cleanup()
        logger.info("Streaming playback stopped")
    
    def cleanup(self) -> None:
        """Clean up audio stream (but keep PyAudio instance for reuse)."""
        # Clean up stream only
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
        except Exception as e:
            logger.error(f"Error cleaning up audio stream: {e}")
        
        # DON'T terminate PyAudio - reuse it for next conversation
        # Only terminate in __exit__ or explicit shutdown
    
    def shutdown(self) -> None:
        """Complete shutdown including PyAudio termination."""
        self.cleanup()
        
        # Now terminate PyAudio
        try:
            if self.pa:
                self.pa.terminate()
                self.pa = None
                logger.info("PyAudio terminated")
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_streaming()
        self.shutdown()  # Complete shutdown including PyAudio


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
    import struct
    import math
    
    # List available devices
    list_audio_devices()
    
    # Test audio playback with a simple tone
    print("\nTesting audio playback with a 440 Hz tone...")
    
    player = AudioPlayer()
    
    # Generate a 1-second 440 Hz sine wave
    duration = 1.0
    sample_rate = player.sample_rate
    frequency = 440.0
    
    samples = []
    for i in range(int(sample_rate * duration)):
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(struct.pack('h', value))
    
    audio_data = b''.join(samples)
    
    print("Playing tone...")
    player.play_audio(audio_data)
    print("Playback complete!")
    
    player.cleanup()
