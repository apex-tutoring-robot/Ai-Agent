"""
Robust PyAudio device resolution by name instead of index.

PyAudio/ALSA device indices are assigned by USB enumeration order, which
is not stable: it can (and does) change across reboots, USB re-plugs, or
even PipeWire restarts. Anything that pins a specific index in `.env`
works until the next boot enumerates devices differently, then silently
opens the wrong device (or a device that doesn't support the requested
direction, which is what a hardcoded index falling on a headphone-out
device does when asked to record). So indices are never stored anywhere
— devices are looked up by name, fresh, on every run.

Resolution order (first match wins):
  1. An explicit index passed by the caller (e.g. a `--device` CLI flag).
  2. Name hint(s) from an environment variable (comma-separated substrings,
     case-insensitive), checked against currently enumerated devices.
  3. Built-in name hints for hardware this project has been run on.
  4. DeviceNotFoundError, listing every currently available device for
     that direction so the operator can fix the name hint.
"""

import os

DEFAULT_INPUT_NAME_HINTS = ["usb pnp", "respeaker", "seeed", "usb audio", "usb"]
DEFAULT_OUTPUT_NAME_HINTS = ["respeaker", "seeed", "usb pnp", "usb audio", "usb"]


class DeviceNotFoundError(RuntimeError):
    pass


def _matches_direction(info, direction):
    return info["maxInputChannels"] > 0 if direction == "input" else info["maxOutputChannels"] > 0


def _channels(info, direction):
    return info["maxInputChannels"] if direction == "input" else info["maxOutputChannels"]


def list_devices(pa, direction=None):
    """Return [(index, info), ...] for all devices, optionally filtered by direction."""
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if direction is None or _matches_direction(info, direction):
            devices.append((i, info))
    return devices


def format_device_list(pa, direction):
    devices = list_devices(pa, direction)
    if not devices:
        return "  (none found)"
    return "\n".join(f"  [{i}] {info['name']} ({_channels(info, direction)} channels)" for i, info in devices)


def find_device(pa, direction, explicit_index=None, name_env_var=None, default_name_hints=None):
    """Resolve a PyAudio device index for `direction` ('input' or 'output')."""
    assert direction in ("input", "output")

    if explicit_index is not None:
        return explicit_index

    env_hints = os.getenv(name_env_var) if name_env_var else None
    hints = [h.strip().lower() for h in env_hints.split(",") if h.strip()] if env_hints else []
    hints += [h.lower() for h in (default_name_hints or [])]

    for hint in hints:
        for i, info in list_devices(pa, direction):
            if hint in info["name"].lower():
                return i

    raise DeviceNotFoundError(
        f"Could not find a {direction} device matching hints {hints or '(none)'}.\n"
        f"Available {direction} devices:\n{format_device_list(pa, direction)}\n"
        f"Set {name_env_var} in .env to a substring of the device name you want to use."
    )


def resolve_input_device(pa, explicit_index=None):
    return find_device(
        pa, "input",
        explicit_index=explicit_index,
        name_env_var="AUDIO_INPUT_DEVICE_NAME",
        default_name_hints=DEFAULT_INPUT_NAME_HINTS,
    )


def resolve_output_device(pa, explicit_index=None):
    return find_device(
        pa, "output",
        explicit_index=explicit_index,
        name_env_var="AUDIO_OUTPUT_DEVICE_NAME",
        default_name_hints=DEFAULT_OUTPUT_NAME_HINTS,
    )
