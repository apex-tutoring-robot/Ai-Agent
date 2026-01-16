"""
Audio playback module for Raspberry Pi.
Supports streaming audio playback with USB audio devices.
Audio playback the process of reproducing previously recorded sound, converting digital data
into audible sound waves that come out of speakers
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
        
        # Latency tracking for TTFAS
        self.first_token_time = None
        self.first_audio_played = False
    
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
        
        # Reset latency tracking
        self.first_audio_played = False
        
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
            
            # Use a larger buffer to prevent underruns and ALSA issues
            chunk_size = 2048
            
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
            # Keep looping until explicitly told to stop AND queue is empty
            while True:
                try:
                    # Get audio chunk from queue (timeout to check stop event)
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    if audio_chunk is None:
                        # None signals end of stream
                        self.audio_queue.task_done()
                        break
                    
                    # CRITICAL: Check if stream still exists before writing
                    if not self.audio_stream:
                        self.audio_queue.task_done()
                        break
                    
                    # Play the chunk with error handling
                    try:
                        # Double-check we should still be playing (stop could have been called)
                        # But still play this chunk we already dequeued
                        if self._stop_event.is_set() and not self.audio_stream:
                            self.audio_queue.task_done()
                            break
                        
                        # Write audio with exception handling
                        # This can block if ALSA has issues, so wrap tightly
                        try:
                            self.audio_stream.write(audio_chunk, exception_on_underflow=False)
                        except OSError as os_err:
                            # ALSA device errors - log but try to continue
                            logger.warning(f"OSError writing chunk (continuing): {os_err}")
                            # Don't break - maybe next chunk will work
                            # Set flag so we know there were issues
                            self.audio_stream = None  # Mark stream as bad
                            self.audio_queue.task_done()
                            self._is_playing = False
                            break  # Exit on first error to prevent cascade
                        
                        # LATENCY: Track first audio playback for TTFAS
                        if not self.first_audio_played and self.first_token_time:
                            first_audio_time = time.perf_counter()
                            ttfas = first_audio_time - self.first_token_time
                            logger.info(f"⏱️  TTFAS (Time To First Audio Spoken): {ttfas:.3f}s")
                            self.first_audio_played = True
                        
                        self.audio_queue.task_done()
                    except Exception as write_error:
                        # PyAudio write errors can be fatal, log and continue
                        logger.error(f"Error writing audio chunk: {write_error}")
                        self.audio_queue.task_done()
                        # If it's a critical error, stop playback gracefully
                        if "Unanticipated host error" in str(write_error) or "Invalid" in str(write_error):
                            logger.error("Critical audio error detected, stopping playback")
                            self._is_playing = False  # Signal main thread
                            break
                
                except queue.Empty:
                    # Queue is empty - check if we should exit
                    if not self._is_playing or self._stop_event.is_set():
                        # Stop signaled and queue empty, safe to exit
                        break
                    # Otherwise keep looping (waiting for more audio)
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
    
    def stop_streaming(self) -> bool:
        """
        Stop streaming playback and clean up.
        
        Returns:
            True if worker stopped cleanly, False if worker stuck (device may be locked)
        """
        if not self._is_playing:
            return True
        
        logger.info("Stopping playback...")
        
        # Signal stop FIRST
        self._is_playing = False
        self._stop_event.set()
        
        # DON'T put None sentinels - let worker finish all queued chunks
        # Worker will exit naturally after processing everything
        
        # Wait for playback thread to finish all queued audio
        thread_stopped = False
        if self.playback_thread:
            logger.info("Waiting for playback thread to finish...")
            self.playback_thread.join(timeout=20.0)
            thread_stopped = not self.playback_thread.is_alive()
            
            if not thread_stopped:
                logger.error("❌ Worker thread STUCK after 10s")
                logger.error("   CANNOT cleanup - would cause memory corruption crash")
                logger.error("   Device will remain locked until next turn attempts recovery")
                logger.error("   This indicates ALSA/hardware issue with audio device")
                # Don't touch PyAudio or stream - worker thread may still be using them
                # Just mark everything as abandoned
                self.audio_stream = None
                # Don't set pa to None - worker might still be using it
                self.playback_thread = None
                self._is_playing = False
                return False  # Indicate worker didn't stop cleanly
            
            self.playback_thread = None
        
        logger.info("✅ Worker stopped cleanly")
        
        # Clear the queue (should be empty if worker finished properly)
        cleared = 0
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if item is not None:
                    cleared += 1
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        if cleared > 0:
            logger.warning(f"⚠️ Cleared {cleared} chunks from queue after worker stopped")
        
        # ONLY cleanup if thread stopped cleanly
        self.cleanup()
        logger.info("Streaming playback stopped")
        return True  # Worker stopped cleanly
    
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
                
                # CRITICAL: Give ALSA time to fully release the device
                # Without this, the next mic stream open fails with "Unanticipated host error"
                time.sleep(0.3)
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
