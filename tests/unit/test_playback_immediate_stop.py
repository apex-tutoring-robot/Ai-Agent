"""
Tests for AudioPlayer.request_immediate_stop() - the cross-thread-safe
interrupt method added to fix a real concurrency bug found via code review:
a background guardrails safety-check thread was calling the full
stop_streaming(immediate=True) directly, which could race with the speaker
thread's own in-flight stop_streaming(immediate=False) call and reach
PortAudio's stream.stop_stream() from two threads at once (undefined
behavior in a C audio library, not just a logic bug - see playback.py's
"we're the only closer" comment on stop_streaming()).

request_immediate_stop() only touches primitives that are already safe to
use from any thread (threading.Event, queue.Queue) and deliberately does
NOT touch self.audio_stream at all - that stays single-writer, handled by
whichever thread already owns the in-flight stop_streaming()/
start_streaming() call.
"""

import queue
import threading

import pytest

from audio.playback import AudioPlayer


@pytest.fixture
def player():
    return AudioPlayer(sample_rate=16000, channels=1)


class TestRequestImmediateStop:
    def test_sets_stop_event(self, player):
        assert not player._stop_event.is_set()
        player.request_immediate_stop()
        assert player._stop_event.is_set()

    def test_sets_is_playing_false(self, player):
        player._is_playing = True
        player.request_immediate_stop()
        assert player._is_playing is False

    def test_clears_queued_audio(self, player):
        player.audio_queue.put(b"\x00\x01" * 10)
        player.audio_queue.put(b"\x02\x03" * 10)
        assert not player.audio_queue.empty()

        player.request_immediate_stop()
        assert player.audio_queue.empty()

    def test_never_touches_audio_stream_attribute(self, player):
        """
        The whole point: this method must not read or write
        self.audio_stream at all - that object's lifecycle belongs
        exclusively to whichever thread is already inside
        stop_streaming()/start_streaming().
        """
        sentinel = object()
        player.audio_stream = sentinel
        player.request_immediate_stop()
        assert player.audio_stream is sentinel

    def test_safe_to_call_on_an_idle_player(self, player):
        # Must not raise even when nothing is playing.
        player.request_immediate_stop()
        assert player._stop_event.is_set()

    def test_next_callback_invocation_returns_abort(self, player):
        import pyaudio

        player._is_playing = True
        player._callback_buf = bytearray()
        player.audio_queue.put(b"\x00\x01" * 320)  # a full frame's worth

        player.request_immediate_stop()

        output, flag = player._audio_callback(None, 320, None, None)
        assert flag == pyaudio.paAbort

    def test_concurrent_call_from_another_thread_does_not_raise(self, player):
        """
        Not a full reproduction of the original PortAudio-level race (that
        needs real hardware), but confirms the Python-level state mutation
        this method performs is safe to invoke from a genuinely different
        thread while the "owning" thread is concurrently doing normal
        AudioPlayer bookkeeping - the scenario that matters is two threads
        calling the OLD stop_streaming(immediate=True) concurrently; this
        method replaces that call with something callable from anywhere.
        """
        player._is_playing = True
        errors = []

        def background_interrupt():
            try:
                player.request_immediate_stop()
            except Exception as exc:
                errors.append(exc)

        def foreground_activity():
            try:
                for _ in range(50):
                    player.audio_queue.put(b"\x00\x00")
                    player._is_playing = True
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=background_interrupt)
        t2 = threading.Thread(target=foreground_activity)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors
        assert player._stop_event.is_set()
