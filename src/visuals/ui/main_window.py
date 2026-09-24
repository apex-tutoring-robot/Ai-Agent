"""
Top-level Qt window hosting the face + teaching canvas scene. Replaces
FaceAnimator's cv2 window as the display surface - QApplication.exec_() owns
the main thread instead of FaceAnimator.render_forever()'s blocking cv2 loop
(the two are mutually exclusive; a canvas can't be embedded in a cv2 window).
"""

import os
from PyQt5.QtWidgets import QMainWindow
from visuals.ui.face_widget import FaceWidget
from visuals.ui.teaching_canvas import TeachingCanvas
from visuals.ui.tutor_scene import TutorScene
from visuals.ui.tutor_view import TutorView


class MainWindow(QMainWindow):
    def __init__(self, signals, faces_dir: str, fullscreen: bool = True):
        super().__init__()
        self.signals = signals

        self.face_widget = FaceWidget(faces_dir)
        self.canvas = TeachingCanvas()
        self.scene = TutorScene(self.face_widget, self.canvas)
        self.view = TutorView(self.scene)

        self.setCentralWidget(self.view)
        self.setWindowTitle("Jarvis AI Tutor")

        if fullscreen:
            self.showFullScreen()
        else:
            # Dev/testing convenience - a real window alongside logs/terminal,
            # same reasoning as FaceAnimator's fullscreen param on the other
            # Windows-fixes branch.
            self.resize(1280, 720)

        self._connect_signals()
        self.show_idle_mode()

    def _connect_signals(self):
        if not self.signals:
            return
        self.signals.listening.connect(self.show_listening_mode)
        self.signals.thinking.connect(self.show_thinking_mode)
        self.signals.start_talking.connect(self.face_widget.start_talking)
        self.signals.stop_talking.connect(self.face_widget.stop_talking)
        self.signals.mouth_level.connect(self.face_widget.push_mouth_level)

        self.signals.draw_actions.connect(self.canvas.handle_draw_actions)
        self.signals.clear_canvas.connect(self.canvas.clear_canvas)
        self.signals.show_face_fullscreen.connect(self.scene.show_face_fullscreen)
        self.signals.show_teaching_layout.connect(self.scene.show_teaching_layout)

    def show_idle_mode(self):
        self.face_widget.stop_talking()
        self.face_widget.start_idle()

    def show_listening_mode(self):
        self.face_widget.stop_talking()
        self.face_widget.start_idle()

    def show_thinking_mode(self):
        self.face_widget.start_thinking()
