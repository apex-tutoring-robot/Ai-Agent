"""
Tests for audio.device_detection - the PyAudio device-index resolution
behind main.py's input/output stream setup.

Three real bugs motivated this module (see its own docstring): a stale
Windows .env index that silently routed audio to the wrong device, a
Raspberry Pi where PulseAudio auto-detection always won even over an
already-correct .env index, and a remote deployment where a metadata-
valid .env index could never actually be opened (PulseAudio had already
claimed the underlying hardware exclusively for itself). These tests
exist to make sure the priority order (valid + openable env override >
PulseAudio > system default > last-resort 1) actually holds, since
there's no real audio hardware to test this against.
"""

import pytest

from audio.device_detection import resolve_device_index, describe_devices


class _FakeStream:
    def close(self):
        pass


class FakePyAudio:
    """A minimal PyAudio stand-in - just enough of the real API surface
    that device_detection.py actually calls. open() succeeds for every
    device by default (matching every pre-existing test's assumption,
    written before the real-open check existed) - pass unopenable_indices
    to simulate a device that's metadata-valid but can't actually be
    opened (e.g. claimed exclusively by PulseAudio)."""

    def __init__(self, devices, default_input=None, default_output=None, unopenable_indices=frozenset()):
        self._devices = {d["index"]: d for d in devices}
        self._default_input = default_input
        self._default_output = default_output
        self._unopenable_indices = unopenable_indices

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        if i not in self._devices:
            raise ValueError(f"PyAudio Error: No device with index {i}")
        return self._devices[i]

    def get_default_input_device_info(self):
        if self._default_input is None:
            raise OSError("No default input device available")
        return self._default_input

    def get_default_output_device_info(self):
        if self._default_output is None:
            raise OSError("No default output device available")
        return self._default_output

    def open(self, input_device_index=None, output_device_index=None, **kwargs):
        index = input_device_index if input_device_index is not None else output_device_index
        if index in self._unopenable_indices:
            raise OSError(f"[Errno -9985] Device unavailable (index {index})")
        return _FakeStream()


def _device(index, name, in_ch=0, out_ch=0):
    return {"index": index, "name": name, "maxInputChannels": in_ch, "maxOutputChannels": out_ch}


class TestEnvOverride:
    def test_valid_env_override_wins_over_everything(self, monkeypatch):
        pa = FakePyAudio([
            _device(0, "pulse", in_ch=2, out_ch=2),
            _device(3, "USB Headset", in_ch=0, out_ch=2),
        ])
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "3")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 3

    def test_unset_env_falls_through(self, monkeypatch):
        pa = FakePyAudio([_device(0, "pulse", in_ch=2, out_ch=2)])
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0

    def test_non_integer_env_is_ignored(self, monkeypatch):
        pa = FakePyAudio([_device(0, "pulse", in_ch=2, out_ch=2)])
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "not-a-number")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0

    def test_nonexistent_device_index_is_ignored(self, monkeypatch):
        pa = FakePyAudio([_device(0, "pulse", in_ch=2, out_ch=2)])
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "99")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0

    def test_wrong_direction_env_index_is_ignored(self, monkeypatch):
        # Index 3 exists but has zero output channels (an input-only mic) -
        # this is exactly the class of bug this module exists to catch.
        pa = FakePyAudio([
            _device(0, "pulse", in_ch=2, out_ch=2),
            _device(3, "USB Mic", in_ch=1, out_ch=0),
        ])
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "3")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0


class TestPulseAutoDetection:
    def test_finds_pulse_device_case_insensitively(self, monkeypatch):
        pa = FakePyAudio([
            _device(0, "HDA Intel", in_ch=2, out_ch=2),
            _device(1, "PulseAudio Sound Server", in_ch=2, out_ch=2),
        ])
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 1

    def test_skips_pulse_device_with_wrong_direction(self, monkeypatch):
        # A "pulse" monitor source that only has input channels shouldn't
        # be picked for output.
        pa = FakePyAudio([
            _device(0, "pulse monitor", in_ch=2, out_ch=0),
            _device(1, "pulse", in_ch=2, out_ch=2),
        ])
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 1


class TestSystemDefaultFallback:
    def test_falls_back_to_system_default_when_no_pulse(self, monkeypatch):
        pa = FakePyAudio(
            [_device(0, "HDA Intel", in_ch=2, out_ch=2)],
            default_output=_device(0, "HDA Intel", in_ch=2, out_ch=2),
        )
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0

    def test_last_resort_returns_1_when_nothing_found(self, monkeypatch):
        pa = FakePyAudio([])
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 1


class TestInputDirection:
    def test_input_direction_uses_input_channels_not_output(self, monkeypatch):
        pa = FakePyAudio([
            _device(0, "pulse", in_ch=0, out_ch=2),  # output-only - shouldn't match input
            _device(1, "pulse", in_ch=2, out_ch=0),
        ])
        monkeypatch.delenv("AUDIO_INPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "input", "AUDIO_INPUT_DEVICE_INDEX") == 1


class TestRealOpenCheck:
    """
    The actual bug this fixes (confirmed live via the on-screen
    ErrorBanner on a remote deployment): a device can be perfectly
    valid by metadata (exists, right channel count) yet still fail to
    actually open - most commonly because PulseAudio has already
    claimed exclusive access to the underlying ALSA hardware. Each tier
    must fall through to the next one instead of returning a candidate
    that merely *looks* usable.
    """

    def test_env_override_that_cannot_actually_open_falls_through_to_pulse(self, monkeypatch):
        pa = FakePyAudio(
            [
                _device(0, "pulse", in_ch=2, out_ch=2),
                _device(3, "USB Audio Device", in_ch=0, out_ch=2),
            ],
            unopenable_indices={3},
        )
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "3")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 0

    def test_pulse_device_that_cannot_actually_open_falls_through_to_system_default(self, monkeypatch):
        pa = FakePyAudio(
            [
                _device(0, "pulse", in_ch=2, out_ch=2),
                _device(1, "HDA Intel", in_ch=2, out_ch=2),
            ],
            default_output=_device(1, "HDA Intel", in_ch=2, out_ch=2),
            unopenable_indices={0},
        )
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 1

    def test_system_default_that_cannot_actually_open_falls_through_to_last_resort(self, monkeypatch):
        pa = FakePyAudio(
            [_device(0, "HDA Intel", in_ch=2, out_ch=2)],
            default_output=_device(0, "HDA Intel", in_ch=2, out_ch=2),
            unopenable_indices={0},
        )
        monkeypatch.delenv("AUDIO_OUTPUT_DEVICE_INDEX", raising=False)
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 1

    def test_openable_env_override_still_wins_as_before(self, monkeypatch):
        # Sanity check that the new check doesn't reject a genuinely
        # fine device - same scenario as TestEnvOverride's original test.
        pa = FakePyAudio([
            _device(0, "pulse", in_ch=2, out_ch=2),
            _device(3, "USB Headset", in_ch=0, out_ch=2),
        ])
        monkeypatch.setenv("AUDIO_OUTPUT_DEVICE_INDEX", "3")
        assert resolve_device_index(pa, "output", "AUDIO_OUTPUT_DEVICE_INDEX") == 3


class TestDescribeDevices:
    def test_lists_every_device(self):
        pa = FakePyAudio([
            _device(0, "HDA Intel", in_ch=2, out_ch=2),
            _device(1, "USB Mic", in_ch=1, out_ch=0),
        ])
        description = describe_devices(pa)
        assert "HDA Intel" in description
        assert "USB Mic" in description

    def test_handles_empty_device_list(self):
        pa = FakePyAudio([])
        assert "no audio devices" in describe_devices(pa).lower()

    def test_handles_enumeration_failure_gracefully(self):
        class BrokenPyAudio:
            def get_device_count(self):
                raise OSError("PortAudio not initialized")
        assert "could not enumerate" in describe_devices(BrokenPyAudio()).lower()
