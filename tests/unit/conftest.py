"""
Ensures audio.wake_word (which imports onnxruntime) is loaded before any
test module gets a chance to import PyQt5 - see main.py's own import-order
comment: onnxruntime and PyQt5 bundle conflicting native DLLs on Windows,
and importing PyQt5 first segfaults the whole process the moment
onnxruntime is imported afterward.

pytest collects test files in alphabetical order by default, so a test
file that imports PyQt5 directly (e.g. test_face_animation_blending.py,
which needs a real QWidget) can end up running before any test that
imports main/audio.wake_word (e.g. test_teaching_repeat_request.py),
reproducing that exact segfault during collection. conftest.py is always
imported first, regardless of test file naming, so establishing the safe
order here fixes it for the whole suite rather than one file at a time.
"""

import audio.wake_word  # noqa: F401
