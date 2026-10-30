"""
Tests for ContinuousVADCapture._is_barge_in_trigger_chunk() - the gate
that decides whether one audio chunk counts toward confirming a
barge-in while the bot is speaking (see always_streaming()).

Root cause this fixes (found via a design review, not live testing this
time): the original barge-in gate was pure energy (RMS) thresholding,
with no check that the loud sound was actually speech - despite this
same class already using webrtcvad.Vad.is_speech() for every other
speech-detection decision in the file. A loud non-speech sound (a
dropped book, a door slam, a dog bark) would have false-positived a
barge-in and cut off the bot's audio for no real reason.

ContinuousVADCapture() opens no real audio device at construction time
(only ensure_stream_open()/always_streaming() etc. do), so it's safe to
construct directly. webrtcvad's own classification of synthetic test
audio isn't something to rely on for a deterministic test (real speech
detection on arbitrary noise isn't predictable), so vad.is_speech is
stubbed here - these tests are about THIS method's combining logic
(loud AND speech, with a fail-safe on a VAD exception), not about
re-validating webrtcvad's own accuracy.
"""

import numpy as np
import pytest

from audio.continuous_vad import ContinuousVADCapture


@pytest.fixture
def vad_capture():
    # input_device_index is irrelevant here - _is_barge_in_trigger_chunk
    # never touches it, and __init__ opens no real audio device anyway.
    return ContinuousVADCapture()


def _pcm16_chunk(num_samples: int, amplitude: int) -> bytes:
    """A chunk of constant-amplitude 16-bit PCM - amplitude controls RMS."""
    return np.full(num_samples, amplitude, dtype=np.int16).tobytes()


class TestIsBargeInTriggerChunk:
    def test_loud_and_speech_triggers(self, vad_capture, monkeypatch):
        monkeypatch.setattr(vad_capture.vad, "is_speech", lambda chunk, rate: True)
        loud = _pcm16_chunk(vad_capture.chunk_size, amplitude=vad_capture._barge_in_energy_threshold + 500)
        triggers, rms = vad_capture._is_barge_in_trigger_chunk(loud)
        assert triggers is True
        assert rms > vad_capture._barge_in_energy_threshold

    def test_loud_but_not_speech_does_not_trigger(self, vad_capture, monkeypatch):
        # The actual bug this fixes: a loud non-speech sound (a dropped
        # book, a door slam) used to count toward a barge-in just from
        # being loud. Now it must also pass the speech classifier.
        monkeypatch.setattr(vad_capture.vad, "is_speech", lambda chunk, rate: False)
        loud = _pcm16_chunk(vad_capture.chunk_size, amplitude=vad_capture._barge_in_energy_threshold + 500)
        triggers, rms = vad_capture._is_barge_in_trigger_chunk(loud)
        assert triggers is False
        assert rms > vad_capture._barge_in_energy_threshold  # confirms it WAS loud enough

    def test_quiet_speech_does_not_trigger(self, vad_capture, monkeypatch):
        # Even if VAD says "speech", it still needs to actually be loud -
        # otherwise ordinary background chatter below the bot's own
        # speaker volume would false-positive too.
        monkeypatch.setattr(vad_capture.vad, "is_speech", lambda chunk, rate: True)
        quiet = _pcm16_chunk(vad_capture.chunk_size, amplitude=10)
        triggers, rms = vad_capture._is_barge_in_trigger_chunk(quiet)
        assert triggers is False
        assert rms < vad_capture._barge_in_energy_threshold

    def test_quiet_and_not_speech_does_not_trigger(self, vad_capture, monkeypatch):
        monkeypatch.setattr(vad_capture.vad, "is_speech", lambda chunk, rate: False)
        quiet = _pcm16_chunk(vad_capture.chunk_size, amplitude=10)
        triggers, rms = vad_capture._is_barge_in_trigger_chunk(quiet)
        assert triggers is False

    def test_vad_short_circuits_before_calling_is_speech_when_too_quiet(self, vad_capture, monkeypatch):
        # Cheap RMS check first - no reason to even run the classifier on
        # a chunk that's already too quiet to matter.
        calls = []
        monkeypatch.setattr(vad_capture.vad, "is_speech", lambda chunk, rate: calls.append(1) or True)
        quiet = _pcm16_chunk(vad_capture.chunk_size, amplitude=10)
        vad_capture._is_barge_in_trigger_chunk(quiet)
        assert calls == []

    def test_vad_exception_fails_safe_to_not_triggering(self, vad_capture, monkeypatch):
        def raise_err(chunk, rate):
            raise ValueError("malformed frame")
        monkeypatch.setattr(vad_capture.vad, "is_speech", raise_err)
        loud = _pcm16_chunk(vad_capture.chunk_size, amplitude=vad_capture._barge_in_energy_threshold + 500)
        triggers, rms = vad_capture._is_barge_in_trigger_chunk(loud)
        assert triggers is False  # a VAD glitch must never count toward a false barge-in


class TestBargeInNearMissReason:
    """
    _barge_in_near_miss_reason() - classifies why a non-triggering chunk
    during playback failed, for always_streaming's diagnostic log line.
    See that method's own docstring for why this exists (a remote report
    that short interjections don't interrupt playback but a longer/
    louder phrase does, with no way to see actual RMS/VAD numbers from a
    device nobody can run a terminal command on).
    """

    def test_an_actual_trigger_is_not_a_near_miss(self, vad_capture):
        reason = vad_capture._barge_in_near_miss_reason(
            is_trigger_chunk=True, rms=vad_capture._barge_in_energy_threshold + 500
        )
        assert reason is None

    def test_ordinary_silence_is_not_a_near_miss(self, vad_capture):
        reason = vad_capture._barge_in_near_miss_reason(
            is_trigger_chunk=False, rms=vad_capture._BARGE_IN_SILENCE_FLOOR_RMS - 1
        )
        assert reason is None

    def test_above_silence_floor_but_below_threshold_is_too_quiet(self, vad_capture):
        reason = vad_capture._barge_in_near_miss_reason(
            is_trigger_chunk=False,
            rms=(vad_capture._BARGE_IN_SILENCE_FLOOR_RMS + vad_capture._barge_in_energy_threshold) / 2,
        )
        assert reason == "too_quiet"

    def test_above_threshold_but_not_triggered_is_not_speech(self, vad_capture):
        # Loud enough to clear the energy bar - the only way this still
        # isn't a trigger is VAD having rejected it as not speech-shaped.
        reason = vad_capture._barge_in_near_miss_reason(
            is_trigger_chunk=False, rms=vad_capture._barge_in_energy_threshold + 500
        )
        assert reason == "not_speech"

    def test_exactly_at_threshold_counts_as_too_quiet(self, vad_capture):
        # Matches _is_barge_in_trigger_chunk's own boundary (rms <=
        # threshold fails to trigger) - consistent classification.
        reason = vad_capture._barge_in_near_miss_reason(
            is_trigger_chunk=False, rms=vad_capture._barge_in_energy_threshold
        )
        assert reason == "too_quiet"
