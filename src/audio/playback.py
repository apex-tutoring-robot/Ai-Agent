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

import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Thread-safe streaming audio player using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels or int(os.getenv('CHANNELS', 1))

        logger.info(f"AudioPlayer initialized: {self.sample_rate}Hz, {self.channels} channel(s)")

        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self._is_playing = False
        self._stop_event = threading.Event()

        # Latency tracking
        self.first_token_time = None
        self.first_audio_played = False
        sd.default.device = (None, 1)

    # --------------------------------------------------
    # Simple blocking playback
    # --------------------------------------------------
    def play_audio(self, audio_data: bytes) -> None:
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, self.sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise

    # --------------------------------------------------
    # Streaming playback
    # --------------------------------------------------
    def start_streaming(self) -> None:
        if self._is_playing:
            logger.warning("Streaming already active")
            return

        self._stop_event.clear()
        self._is_playing = True
        self.first_audio_played = False

        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()

        logger.info("Streaming playback started")

    def _playback_worker(self) -> None:
        try:
            while True:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)

                    if chunk is None:
                        self.audio_queue.task_done()
                        break

                    audio_array = np.frombuffer(chunk, dtype=np.int16)

                    sd.play(audio_array, self.sample_rate, blocking=True)

                    # Latency tracking
                    if not self.first_audio_played and self.first_token_time:
                        first_audio_time = time.perf_counter()
                        ttfas = first_audio_time - self.first_token_time
                        logger.info(f"⏱️ TTFAS: {ttfas:.3f}s")
                        self.first_audio_played = True

                    self.audio_queue.task_done()

                except queue.Empty:
                    if not self._is_playing or self._stop_event.is_set():
                        break
                    continue

        except Exception as e:
            logger.error(f"Error in playback worker: {e}")

        finally:
            logger.info("Playback worker stopped")

    def queue_audio(self, audio_chunk: bytes) -> None:
        if not self._is_playing:
            raise RuntimeError("Streaming not started.")
        self.audio_queue.put(audio_chunk)

    def stop_streaming(self) -> bool:
        if not self._is_playing:
            return True

        logger.info("Stopping playback...")

        self._is_playing = False
        self._stop_event.set()

        if self.playback_thread:
            self.playback_thread.join(timeout=10.0)
            self.playback_thread = None

        logger.info("Streaming playback stopped")
        return True

    def cleanup(self) -> None:
        sd.stop()

    def shutdown(self) -> None:
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()
        self.shutdown()
