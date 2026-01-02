"""
Suppress ALSA warnings and errors.
Import this module at the start of your script to hide ALSA configuration warnings.
"""

import os
import sys
from ctypes import *

# Suppress ALSA error messages
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    """Custom error handler that suppresses ALSA errors."""
    pass

try:
    asound = cdll.LoadLibrary('libasound.so.2')
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    # If we can't load the library, that's okay - just continue
    pass
