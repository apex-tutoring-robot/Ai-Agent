"""
Audio I/O device resolution - a single place that decides which PyAudio
input/output device index to actually use, replacing the ad-hoc "just
grep for a device named 'pulse', ignore whatever .env says" logic that
used to live directly in main.py.

Two real bugs motivated this:
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

resolve_device_index() fixes both: it validates whatever index is being
considered (does it exist? does it have the right direction's channels?)
before trusting it, and makes an explicit, *valid* .env override win over
PulseAudio auto-detection instead of being silently ignored. describe_
devices() logs every available device once at startup, so a wrong choice
is visible immediately instead of discovered later as unexplained silence.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


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


def _env_override(pa, direction: str, env_var: str) -> Optional[int]:
    """The env var's device index, IF it's set and valid for this
    direction - never trusted blindly. Returns None (not a fallback
    value) when unset, malformed, missing, or the wrong direction, so the
    caller can fall through to auto-detection instead of silently using
    a device that can't actually do what's being asked of it."""
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

    return index


def _find_pulse_device(pa, direction: str) -> Optional[int]:
    """PulseAudio (Linux/Pi) exposes a single bridging device that
    handles both directions - but only once it's confirmed to actually
    have channels in the requested direction, not just by name match."""
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info and "pulse" in info.get("name", "").lower() and _has_direction(info, direction):
                return i
    except Exception as e:
        logger.warning(f"Error searching for PulseAudio device: {e}")
    return None


def _find_system_default(pa, direction: str) -> Optional[int]:
    """Last resort before giving up: whatever the OS/PyAudio itself
    considers the default device for this direction."""
    try:
        info = (
            pa.get_default_input_device_info() if direction == "input"
            else pa.get_default_output_device_info()
        )
        if info and _has_direction(info, direction):
            return info["index"]
    except Exception:
        pass
    return None


def resolve_device_index(pa, direction: str, env_var: str) -> int:
    """
    Picks the PyAudio device index to use for `direction` ("input" or
    "output"), in priority order:
      1. An explicit env var override - IF it's valid for this direction.
      2. A PulseAudio bridging device (Linux/Pi) - IF one exists and has
         the right direction's channels.
      3. The system's own default device for this direction.
      4. A hardcoded last resort of 1 (kept only so callers always get an
         int back) - reaching this means nothing usable was found at
         all, so it's logged as an error with the full device list.
    """
    override = _env_override(pa, direction, env_var)
    if override is not None:
        info = pa.get_device_info_by_index(override)
        logger.info(f"Using {env_var}={override} ('{info.get('name', '?')}') for {direction}")
        return override

    pulse = _find_pulse_device(pa, direction)
    if pulse is not None:
        info = pa.get_device_info_by_index(pulse)
        logger.info(f"Found PulseAudio device at index {pulse} ('{info.get('name', '?')}') for {direction}")
        return pulse

    default = _find_system_default(pa, direction)
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
