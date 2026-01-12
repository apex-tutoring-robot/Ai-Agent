import threading
import queue
import pyaudio


class AudioPlayer:
    """
    Real-time audio playback for Raspberry Pi using PyAudio.
    Accepts PCM 16-bit mono @ 16kHz.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._running = False
        self._thread = None

        self._pyaudio = pyaudio.PyAudio()
        self._stream = None

        # Audio format (must match Azure TTS output)
        self.rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16

    def start_streaming(self):
        if self._running:
            return

        self._stream = self._pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=1024
        )

        self._running = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def queue_audio(self, audio_chunk: bytes):
        if self._running and audio_chunk:
            self._queue.put(audio_chunk)

    def stop_streaming(self):
        self._running = False

        if self._thread:
            self._thread.join(timeout=1)

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        with self._queue.mutex:
            self._queue.queue.clear()

    def cleanup(self):
        self.stop_streaming()
        self._pyaudio.terminate()

    def _play_loop(self):
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk:
                self._stream.write(chunk)
