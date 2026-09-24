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

    # Teaching canvas
    draw_actions = pyqtSignal(object)   # list[dict] - see teaching_canvas.handle_draw_actions
    clear_canvas = pyqtSignal()
    show_face_fullscreen = pyqtSignal()
    show_teaching_layout = pyqtSignal()

    # Emitted with the new AudioPlayer.volume (0.0-2.0) whenever the
    # set_volume tool actually changes it - see JarvisBot._tool_set_volume.
    volume_changed = pyqtSignal(float)
