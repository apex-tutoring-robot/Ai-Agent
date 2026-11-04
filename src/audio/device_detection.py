"""
Audio I/O device resolution - a single place that decides which PyAudio
input/output device index to actually use, replacing the ad-hoc "just
grep for a device named 'pulse', ignore whatever .env says" logic that
used to live directly in main.py.

Three real bugs motivated this:
1. On a Windows dev machine, a numeric device index in .env silently went
   stale after a driver/OS change (AUDIO_OUTPUT_DEVICE_INDEX pointed at a
   monitor's HDMI audio instead of the actual headset) - no error, just
   silence, only root-caused by manually cross-referencing every device's
   own get_device_info_by_index() against the .env value.
2. On a teammate's Raspberry Pi, PulseAudio auto-detection was *always*
   preferred over an explicit .env index, with no way to override it even
   after he'd confirmed his own index was correct - so a bad PulseAudio
   default sink would silently win with no indication why the "correctly
   configured" .env wasn't taking effect.
3. On a remote deployment (confirmed live via the on-screen ErrorBanner -
   see ui_signals.fatal_error), AUDIO_INPUT_DEVICE_INDEX pointed at a raw
   ALSA hardware device that *looked* perfectly valid (existed, had input
   channels) but could never actually be opened, because PulseAudio had
   already claimed exclusive access to that same hardware for itself -
   only its own 'pulse' bridging device could still reach it. Metadata
   alone can't see this: WakeWordDetector.start() only discovered it
   several steps later, as a generic "could not open audio input stream
   after all fallbacks failed" with the real reason nowhere in sight.

resolve_device_index() fixes all three: it validates whatever index is
being considered (does it exist? does it have the right direction's
channels? can a real stream actually be opened on it, not just claimed by
its metadata?) before trusting it, and makes an explicit, *valid* .env
override win over PulseAudio auto-detection instead of being silently
ignored. A candidate that fails the real-open check falls through to the
next tier instead of being returned anyway. describe_devices() logs every
available device once at startup, so a wrong choice is visible
immediately instead of discovered later as unexplained silence.
"""

import logging
import os
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)

# Matches every current caller (mic capture, VAD, and Azure TTS output
# all run at 16kHz) - see resolve_device_index's sample_rate param.
_DEFAULT_PROBE_SAMPLE_RATE = 16000


def _can_open_stream(pa, direction: str, index: int, sample_rate: int) -> bool:
    """
    Whether a real PyAudio stream can actually be opened on this device
    for `direction`, not just whether its metadata claims to support it
    - see this module's own docstring (bug #3) for the real gap this
    closes: a device can be metadata-valid (exists, right channel count)
    yet still fail to actually open, most commonly because PulseAudio
    has already claimed exclusive access to the underlying ALSA
    hardware and only its own 'pulse' bridging device can still reach
    it. Opens and immediately closes - this is a cheap probe, not a
    real stream, and runs once per candidate at startup, not per-call.
    """
    try:
        stream = pa.open(
            rate=sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=(direction == "input"),
            output=(direction == "output"),
            frames_per_buffer=1024,
            input_device_index=index if direction == "input" else None,
            output_device_index=index if direction == "output" else None,
        )
        stream.close()
        return True
    except Exception as e:
        logger.warning(f"Device index {index} looks valid but failed to actually open for {direction}: {e}")
        return False


def _has_direction(info: dict, direction: str) -> bool:
    key = "maxInputChannels" if direction == "input" else "maxOutputChannels"
    return info.get(key, 0) > 0


def describe_devices(pa) -> str:
    """Human-readable listing of every PyAudio device - log this once at
    startup so what's actually available is visible up front."""
    try:
        count = pa.get_device_count()
    except Exception as e:
        return f"(could not enumerate audio devices: {e})"

    lines = []
    for i in range(count):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue
        lines.append(
            f"  [{i}] {info.get('name', '?')} "
            f"(in={info.get('maxInputChannels', 0)}, out={info.get('maxOutputChannels', 0)})"
        )
    return "\n".join(lines) if lines else "  (no audio devices found)"


def _env_override(pa, direction: str, env_var: str, sample_rate: int) -> Optional[int]:
    """The env var's device index, IF it's set and valid for this
    direction - never trusted blindly. Returns None (not a fallback
    value) when unset, malformed, missing, the wrong direction, or a
    real stream can't actually be opened on it (see _can_open_stream -
    this is the case metadata alone can't catch), so the caller can
    fall through to auto-detection instead of silently using a device
    that can't actually do what's being asked of it."""
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return None

    try:
        index = int(raw)
    except ValueError:
        logger.warning(f"{env_var}={raw!r} isn't a valid integer - ignoring it.")
        return None

    try:
        info = pa.get_device_info_by_index(index)
    except Exception:
        logger.warning(f"{env_var}={index} doesn't exist on this machine - ignoring it.")
        return None

    if not _has_direction(info, direction):
        logger.warning(
            f"{env_var}={index} ('{info.get('name', '?')}') has no {direction} channels - ignoring it."
        )
        return None

    if not _can_open_stream(pa, direction, index, sample_rate):
        logger.warning(
            f"{env_var}={index} ('{info.get('name', '?')}') has {direction} channels but a real "
            f"stream won't open on it (likely claimed exclusively by PulseAudio) - ignoring it."
        )
        return None

    return index


def _find_pulse_device(pa, direction: str, sample_rate: int) -> Optional[int]:
    """PulseAudio (Linux/Pi) exposes a single bridging device that
    handles both directions - but only once it's confirmed to actually
    have channels in the requested direction (not just by name match)
    AND a real stream will actually open on it."""
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (info and "pulse" in info.get("name", "").lower() and _has_direction(info, direction)
                    and _can_open_stream(pa, direction, i, sample_rate)):
                return i
    except Exception as e:
        logger.warning(f"Error searching for PulseAudio device: {e}")
    return None


def _find_system_default(pa, direction: str, sample_rate: int) -> Optional[int]:
    """Last resort before giving up: whatever the OS/PyAudio itself
    considers the default device for this direction - still subject to
    the same real-open check as every other tier, since "default"
    offers no guarantee either."""
    try:
        info = (
            pa.get_default_input_device_info() if direction == "input"
            else pa.get_default_output_device_info()
        )
        if info and _has_direction(info, direction) and _can_open_stream(pa, direction, info["index"], sample_rate):
            return info["index"]
    except Exception:
        pass
    return None


def resolve_device_index(pa, direction: str, env_var: str, sample_rate: int = _DEFAULT_PROBE_SAMPLE_RATE) -> int:
    """
    Picks the PyAudio device index to use for `direction` ("input" or
    "output"), in priority order:
      1. An explicit env var override - IF it's valid for this direction
         AND a real stream actually opens on it.
      2. A PulseAudio bridging device (Linux/Pi) - IF one exists, has
         the right direction's channels, and actually opens.
      3. The system's own default device for this direction - same
         real-open requirement.
      4. A hardcoded last resort of 1 (kept only so callers always get an
         int back) - reaching this means nothing usable was found at
         all, so it's logged as an error with the full device list.

    `sample_rate` is only used for the real-open probe at each tier
    (see _can_open_stream) - it defaults to 16kHz, which every current
    caller (mic capture, VAD, and Azure TTS output) actually runs at.
    """
    override = _env_override(pa, direction, env_var, sample_rate)
    if override is not None:
        info = pa.get_device_info_by_index(override)
        logger.info(f"Using {env_var}={override} ('{info.get('name', '?')}') for {direction}")
        return override

    pulse = _find_pulse_device(pa, direction, sample_rate)
    if pulse is not None:
        info = pa.get_device_info_by_index(pulse)
        logger.info(f"Found PulseAudio device at index {pulse} ('{info.get('name', '?')}') for {direction}")
        return pulse

    default = _find_system_default(pa, direction, sample_rate)
    if default is not None:
        info = pa.get_device_info_by_index(default)
        logger.info(f"Using system default device {default} ('{info.get('name', '?')}') for {direction}")
        return default

    logger.error(
        f"No usable {direction} audio device found (no valid {env_var}, no PulseAudio "
        f"device, no system default) - falling back to index 1, which is very likely "
        f"wrong. Available devices:\n{describe_devices(pa)}"
    )
    return 1
