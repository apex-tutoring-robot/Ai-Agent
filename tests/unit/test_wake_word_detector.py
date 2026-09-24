"""
Tests for WakeWordDetector.stop()'s PyAudio ownership handling.

Regression test for a real bug found via code review: stop() set
self.pa = None unconditionally, even in the branch that explicitly logs
"Keeping shared PyAudio instance alive" because _owns_pa is False. Since
JarvisBot._restart_wake_word() reuses the SAME WakeWordDetector instance
for the app's whole lifetime (stop()/start() cycled repeatedly, never
recreated), nulling the shared reference meant the next start() call's
"if not self.pa" check created a brand-new PyAudio instance instead of
reusing the one shared with AudioPlayer/ContinuousVADCapture - real
resource churn on every wake-word cycle after the first.

WakeWordDetector.__init__ touches no audio hardware (only start() does),
so it's safe to construct directly and drive stop() by hand without a
real microphone/PortAudio device.
"""

import pytest

from audio.wake_word import WakeWordDetector


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudio:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


@pytest.fixture
def detector():
    return WakeWordDetector()


class TestStopWithSharedPyAudio:
    def test_does_not_null_out_a_shared_pa_instance(self, detector):
        shared_pa = FakePyAudio()
        detector.pa = shared_pa
        detector._owns_pa = False

        detector.stop()

        assert detector.pa is shared_pa

    def test_does_not_terminate_a_shared_pa_instance(self, detector):
        shared_pa = FakePyAudio()
        detector.pa = shared_pa
        detector._owns_pa = False

        detector.stop()

        assert shared_pa.terminated is False


class TestStopWithOwnedPyAudio:
    def test_terminates_and_nulls_an_owned_pa_instance(self, detector):
        owned_pa = FakePyAudio()
        detector.pa = owned_pa
        detector._owns_pa = True

        detector.stop()

        assert owned_pa.terminated is True
        assert detector.pa is None


class TestStopClosesAudioStream:
    def test_stops_and_closes_the_audio_stream(self, detector):
        stream = FakeStream()
        detector.audio_stream = stream
        detector.pa = FakePyAudio()
        detector._owns_pa = False

        detector.stop()

        assert stream.stopped is True
        assert stream.closed is True
        assert detector.audio_stream is None

    def test_safe_to_call_with_no_stream_or_pa(self, detector):
        # Must not raise.
        detector.stop()
        assert detector.audio_stream is None


class TestRepeatedStopStartCycleKeepsSharedInstance:
    """Simulates JarvisBot._restart_wake_word()'s actual pattern: the same
    detector instance gets stop() then start()-equivalent reuse across
    multiple wake-word cycles, never recreated."""

    def test_pa_survives_multiple_stop_calls(self, detector):
        shared_pa = FakePyAudio()
        detector.pa = shared_pa
        detector._owns_pa = False

        for _ in range(3):
            detector._is_running = True
            detector.stop()
            assert detector.pa is shared_pa
            assert detector._owns_pa is False
