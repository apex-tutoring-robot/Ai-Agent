"""
Hosts the TeachingCanvas and FaceWidget in one 1280x720 scene, toggling
between "face fills the whole screen" (normal chat) and "face shrinks to a
small circular corner bubble while the canvas shows a diagram" (math
teaching) layouts - the second one deliberately styled like a webcam
overlay in a screen recording (e.g. Loom), not just a smaller square face,
per live feedback that a same-size square face next to the whiteboard
read as two competing panels rather than a presenter overlay.
"""

from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsScene

# Bubble size/margin for the teaching-layout corner overlay - see
# show_teaching_layout(). Bottom-right, clear of both the equation zone
# (x: 60-420) and the diagram zone (x: 480-750) that generate_teaching_plan()
# targets, same reasoning the old top-right placement used.
_BUBBLE_SIZE = 190
_BUBBLE_MARGIN = 30


class TutorScene(QGraphicsScene):
    def __init__(self, face_widget, teaching_canvas, parent=None):
        super().__init__(parent)
        self.setSceneRect(QRectF(0, 0, 1280, 720))

        self.canvas_item = teaching_canvas
        self.addItem(self.canvas_item)

        self.face_proxy = self.addWidget(face_widget)
        self.face_proxy.setZValue(10)
        face_widget.setFixedSize(420, 360)
        self.show_face_fullscreen()

    def show_face_fullscreen(self):
        self.canvas_item.setVisible(False)
        self.face_proxy.widget().set_circular_mask(False)
        self.face_proxy.widget().setFixedSize(1280, 720)
        self.face_proxy.setPos(0, 0)

    def show_teaching_layout(self):
        self.canvas_item.setVisible(True)
        self.face_proxy.widget().setFixedSize(_BUBBLE_SIZE, _BUBBLE_SIZE)
        self.face_proxy.widget().set_circular_mask(True)
        self.face_proxy.setPos(
            1280 - _BUBBLE_SIZE - _BUBBLE_MARGIN,
            720 - _BUBBLE_SIZE - _BUBBLE_MARGIN,
        )
