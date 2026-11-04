"""
Tests for audio.playback.AudioPlayer's sample-rate correction -
_resample_pcm16() and queue_audio()'s use of it.

Root cause this fixes (found live on a real Raspberry Pi, not
reproducible on Windows): TTS audio played back ~3-4x too fast ("like a
sped-up cartoon") for a second or two, then went silent - a classic
symptom of a stream opened at a rate the audio backend doesn't actually
honor, playing 16kHz-encoded PCM back at the device's real (higher)
native rate with no resampling. start_streaming() now detects this (on
non-Windows platforms only - see its own comment) and resamples outgoing
audio to the device's actual rate instead of trusting the OS to.

AudioPlayer() itself opens no real audio device - only start_streaming()/
play_audio() do - so it's safe to instantiate directly and drive
queue_audio()/_resample_pcm16() by hand without any real hardware.
"""

import numpy as np
import pytest

from audio.playback import AudioPlayer


@pytest.fixture
def player():
    p = AudioPlayer(sample_rate=16000, channels=1)
    p._is_playing = True  # queue_audio() requires this
    return p


def _tone_bytes(num_samples: int, value: int = 1000) -> bytes:
    """A constant-value "tone" as 16-bit PCM - easy to assert resampling
    preserves amplitude/shape, not just length."""
    return np.full(num_samples, value, dtype=np.int16).tobytes()


class TestResamplePcm16:
    def test_identity_when_rates_match(self):
        data = _tone_bytes(100)
        assert AudioPlayer._resample_pcm16(data, 16000, 16000) is data

    def test_empty_input_returns_empty(self):
        assert AudioPlayer._resample_pcm16(b"", 16000, 48000) == b""

    def test_upsampling_scales_length_by_the_rate_ratio(self):
        data = _tone_bytes(1000)
        resampled = AudioPlayer._resample_pcm16(data, 16000, 48000)
        out_samples = len(resampled) // 2  # 16-bit = 2 bytes/sample
        assert out_samples == pytest.approx(3000, rel=0.01)

    def test_downsampling_scales_length_by_the_rate_ratio(self):
        data = _tone_bytes(3000)
        resampled = AudioPlayer._resample_pcm16(data, 48000, 16000)
        out_samples = len(resampled) // 2
        assert out_samples == pytest.approx(1000, rel=0.01)

    def test_constant_signal_stays_constant_after_resampling(self):
        # A correctness check, not just a length check - resampling a
        # flat tone should produce (approximately) the same flat tone,
        # not noise or a scaled amplitude.
        data = _tone_bytes(1000, value=5000)
        resampled = AudioPlayer._resample_pcm16(data, 16000, 44100)
        samples = np.frombuffer(resampled, dtype=np.int16)
        assert samples.min() >= 4990
        assert samples.max() <= 5010

    def test_too_short_to_resample_returns_input_unchanged(self):
        data = _tone_bytes(1)  # < 2 samples
        assert AudioPlayer._resample_pcm16(data, 16000, 48000) == data


class TestQueueAudioAppliesResamplingWhenNeeded:
    def test_no_resampling_when_playback_rate_matches_sample_rate(self, player):
        data = _tone_bytes(100)
        player.queue_audio(data)
        assert player.audio_queue.get_nowait() == data

    def test_resamples_when_playback_rate_differs(self, player):
        player._playback_rate = 48000  # simulates start_streaming()'s device-rate override
        data = _tone_bytes(1000)
        player.queue_audio(data)
        queued = player.audio_queue.get_nowait()
        assert queued != data
        assert len(queued) // 2 == pytest.approx(3000, rel=0.01)

    def test_raises_if_not_playing(self):
        p = AudioPlayer(sample_rate=16000, channels=1)
        with pytest.raises(RuntimeError):
            p.queue_audio(_tone_bytes(10))
