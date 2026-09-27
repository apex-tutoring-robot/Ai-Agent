"""
Thread-safe signal bus between JarvisBot's audio/LLM worker threads and the
Qt GUI (main thread). Qt widgets can only be touched from the thread that
owns the QApplication event loop - these signals let background threads
request UI changes safely via Qt's built-in cross-thread signal/slot queuing,
instead of calling into widgets directly.
"""

from PyQt5.QtCore import QObject, pyqtSignal


class UISignals(QObject):
    # Face state (replaces the old FaceAnimator.start_talking()/start_idle()/
    # start_thinking() direct calls - same states, now emitted as signals)
    listening = pyqtSignal()
    thinking = pyqtSignal()
    start_talking = pyqtSignal()
    stop_talking = pyqtSignal()
    mouth_level = pyqtSignal(float)

    # Emitted with an expression.controller.Expression's value ("friendly",
    # "encouraging", "calm", "excited", "neutral") right before speaking a
    # comprehension-check response - see JarvisBot._handle_teaching_answer.
    # Mapped down to FaceWidget's existing 3 emotion images (see
    # main_window.show_expression), not a 1:1 new asset per expression.
    set_expression = pyqtSignal(str)

    # Teaching canvas
    draw_actions = pyqtSignal(object)   # list[dict] - see teaching_canvas.handle_draw_actions
    clear_canvas = pyqtSignal()
    show_face_fullscreen = pyqtSignal()
    show_teaching_layout = pyqtSignal()

    # Emitted with the new AudioPlayer.volume (0.0-2.0) whenever the
    # set_volume tool actually changes it - see JarvisBot._tool_set_volume.
    volume_changed = pyqtSignal(float)

    # Display-sleep cycle after prolonged idle (see JarvisBot.run()'s main
    # loop) - Qt-side equivalent of the old cv2 FaceAnimator's
    # enter_sleep()/wake_up(), paired with an actual HDMI power toggle via
    # vcgencmd on a real Pi.
    enter_sleep = pyqtSignal()
    wake_up = pyqtSignal()
