"""
Tests for audio.playback.AudioPlayer's volume control - set_volume()'s
clamping, and the real _audio_callback() gain-scaling path (not a
reimplemented copy of the formula), since that's exactly where the bug
this session found and fixed lived: a stray local `import numpy as np`
elsewhere in the same function shadowed the module-level import for the
whole function scope, crashing every callback invocation whenever
volume != 1.0.

AudioPlayer() itself opens no real audio device - only start_streaming()/
play_audio() do - so it's safe to instantiate directly and drive
_audio_callback() by hand without any real hardware or PortAudio stream.
"""

import struct

import numpy as np
import pytest

from audio.playback import AudioPlayer


@pytest.fixture
def player():
    return AudioPlayer(sample_rate=16000, channels=1)


class TestSetVolume:
    def test_default_volume_is_normal(self, player):
        assert player.volume == 1.0

    @pytest.mark.parametrize(
        "requested,expected",
        [
            (1.5, 1.5),
            (0.0, 0.0),
            (2.0, 2.0),
            (-0.5, 0.0),   # clamped to floor
            (5.0, 2.0),    # clamped to ceiling
        ],
    )
    def test_clamps_to_valid_range(self, player, requested, expected):
        player.set_volume(requested)
        assert player.volume == expected


class TestAudioCallbackGain:
    """Drives the real _audio_callback() PortAudio callback directly."""

    def _make_frame_bytes(self, samples):
        return struct.pack(f"<{len(samples)}h", *samples)

    def _run_callback(self, player, pcm_bytes, frame_count):
        player._is_playing = True
        player._callback_buf = bytearray()
        player.audio_queue.put(pcm_bytes)
        output, flag = player._audio_callback(None, frame_count, None, None)
        return output

    def test_normal_volume_passes_samples_through_unchanged(self, player):
        samples = [1000, -1000, 16000, -16000]
        pcm = self._make_frame_bytes(samples)
        output = self._run_callback(player, pcm, frame_count=len(samples))
        assert np.frombuffer(output, dtype=np.int16).tolist() == samples

    def test_louder_volume_scales_up(self, player):
        player.set_volume(1.5)
        samples = [1000, -1000, 2000, -2000]
        pcm = self._make_frame_bytes(samples)
        output = self._run_callback(player, pcm, frame_count=len(samples))
        assert np.frombuffer(output, dtype=np.int16).tolist() == [1500, -1500, 3000, -3000]

    def test_quieter_volume_scales_down(self, player):
        player.set_volume(0.5)
        samples = [1000, -1000, 2000, -2000]
        pcm = self._make_frame_bytes(samples)
        output = self._run_callback(player, pcm, frame_count=len(samples))
        assert np.frombuffer(output, dtype=np.int16).tolist() == [500, -500, 1000, -1000]

    def test_high_volume_clips_cleanly_without_wraparound_distortion(self, player):
        # The whole point of clipping instead of raw multiplication: int16
        # overflow wraps around (30000 * 2 mod 65536 becomes a huge
        # negative-looking number) which sounds like harsh digital
        # distortion, not just "loud". Clipping caps it at the ceiling.
        player.set_volume(2.0)
        samples = [30000, -30000]
        pcm = self._make_frame_bytes(samples)
        output = self._run_callback(player, pcm, frame_count=len(samples))
        result = np.frombuffer(output, dtype=np.int16).tolist()
        assert result == [32767, -32768]

    def test_callback_does_not_crash_with_gain_applied(self, player):
        """
        Regression test for the exact bug found live: a redundant local
        `import numpy as np` inside the same function's lip-sync RMS block
        made Python treat `np` as local to the whole function, so the gain
        code (which runs earlier in the same function) raised
        UnboundLocalError on every chunk whenever volume != 1.0 - the
        callback would return garbage/crash instead of scaled audio.
        """
        player.set_volume(1.25)
        samples = [500, -500, 1000, -1000]
        pcm = self._make_frame_bytes(samples)
        # Must not raise
        output = self._run_callback(player, pcm, frame_count=len(samples))
        assert len(output) == len(pcm)
