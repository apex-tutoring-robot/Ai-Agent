"""
Tests for MainWindow.show_expression() - the expression-string -> FaceWidget
dispatch (see ui_signals.py's set_expression, driven by
ExpressionController's DeliveryStyle).

Regression test for a real bug found live: ExpressionController.refusal()
correctly decides CONFUSED, and TutorAction.RETEACH correctly decides SAD
(see expression/controller.py) - but show_expression() had no branch for
either string, so both silently fell through to idle/neutral despite
FaceWidget.start_confused()/start_sad() (and their art) already existing.

Constructs a real MainWindow (needs a real QApplication + real face
images - conftest.py in this directory already guarantees the
onnxruntime-before-PyQt5 import order project-wide) then swaps in a Mock
for face_widget, since show_expression's actual logic is just "which
FaceWidget method gets called", not anything about rendering itself.
"""
from unittest.mock import Mock

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.main_window import MainWindow


def _window_with_mock_face() -> MainWindow:
    window = MainWindow(signals=None, faces_dir=r"src\visuals\faces", fullscreen=False)
    window.face_widget = Mock()
    return window


class TestShowExpression:
    def test_friendly_shows_happy(self):
        window = _window_with_mock_face()
        window.show_expression("friendly")
        window.face_widget.start_happy.assert_called_once()

    def test_excited_shows_excited(self):
        window = _window_with_mock_face()
        window.show_expression("excited")
        window.face_widget.start_excited.assert_called_once()

    def test_encouraging_shows_thinking(self):
        window = _window_with_mock_face()
        window.show_expression("encouraging")
        window.face_widget.start_thinking.assert_called_once()

    def test_confused_shows_confused(self):
        window = _window_with_mock_face()
        window.show_expression("confused")
        window.face_widget.start_confused.assert_called_once()

    def test_sad_shows_sad(self):
        window = _window_with_mock_face()
        window.show_expression("sad")
        window.face_widget.start_sad.assert_called_once()

    def test_unhandled_expression_falls_back_to_idle(self):
        window = _window_with_mock_face()
        window.show_expression("calm")
        window.face_widget.start_idle.assert_called_once()
