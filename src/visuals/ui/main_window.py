import os
import math
from PyQt5.QtWidgets import QMainWindow

from visuals.ui.face_widget import FaceWidget
from visuals.ui.teaching_canvas import TeachingCanvas
from visuals.ui.tutor_scene import TutorScene
from visuals.ui.tutor_view import TutorView


class MainWindow(QMainWindow):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

        base_dir = os.path.dirname(__file__)
        face_dir = os.path.abspath(os.path.join(base_dir, "..", "faces"))

        self.face_widget = FaceWidget(face_dir)
        self.canvas = TeachingCanvas()
        self.canvas.clear_canvas()

        self.scene = TutorScene(self.face_widget, self.canvas)
        self.view = TutorView(self.scene)

        self.signals.show_face_fullscreen.connect(self.scene.show_face_fullscreen)
        self.signals.show_teaching_layout.connect(self.scene.show_teaching_layout)

        self.setCentralWidget(self.view)
        self.setWindowTitle("CHIPPY AI Tutor")
        self.resize(1280, 720)

        self._connect_signals()
        # draw_actions is handled by canvas directly (single connection)
        self.signals.draw_actions.connect(self.canvas.handle_draw_actions)

        self.show_idle_mode()

    def _connect_signals(self):
        if not self.signals:
            return
        self.signals.listening.connect(self.show_listening_mode)
        self.signals.thinking.connect(self.show_thinking_mode)
        self.signals.clear_canvas.connect(self.clear_teaching)
        self.signals.start_talking.connect(self.face_widget.start_talking)
        self.signals.stop_talking.connect(self.face_widget.stop_talking)
        self.signals.mouth_level.connect(self.face_widget.push_mouth_level)

    def show_idle_mode(self):
        self.face_widget.stop_talking()

    def show_listening_mode(self):
        self.face_widget.stop_talking()

    def show_thinking_mode(self):
        self.face_widget.start_thinking()

    def clear_teaching(self):
        self.canvas.clear_canvas()

    def regular_polygon_points(self, cx, cy, radius, sides):
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((int(x), int(y)))
        return points
