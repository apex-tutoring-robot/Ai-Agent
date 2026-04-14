"""
Suppress ALSA warnings and errors that appear during PyAudio/PortAudio init.
Import this module before initialising PyAudio to silence noisy ALSA messages
about missing OSS devices, a52, IEC958, etc.
"""

from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

_ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def _noop_error_handler(filename, line, function, err, fmt):
    pass

# Keep a module-level reference so the C callback is never garbage collected
# while it is registered with libasound.
_c_error_handler = _ERROR_HANDLER_FUNC(_noop_error_handler)

try:
    _asound = cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass
