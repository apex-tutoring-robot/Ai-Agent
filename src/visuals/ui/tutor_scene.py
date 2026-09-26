"""
Hosts the TeachingCanvas and FaceWidget in one 1280x720 scene, toggling
between "face fills the whole screen" (normal chat) and "face shrinks to a
small circular corner bubble while the canvas shows a diagram" (math
teaching) layouts - the second one deliberately styled like a webcam
overlay in a screen recording (e.g. Loom), not just a smaller square face,
per live feedback that a same-size square face next to the whiteboard
read as two competing panels rather than a presenter overlay.
"""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import QGraphicsScene

# Bubble size/margin for the teaching-layout corner overlay - see
# show_teaching_layout(). Bottom-right, clear of both the equation zone
# (x: 60-420) and the diagram zone (x: 480-750) that generate_teaching_plan()
# targets, same reasoning the old top-right placement used.
_BUBBLE_SIZE = 190
_BUBBLE_MARGIN = 30

# Same friendly blue as TeachingCanvas's whiteboard frame, for a
# consistent "framed" look across the two panels.
_FRAME_COLOR = QColor(60, 130, 200)
_FRAME_WIDTH = 6


class TutorScene(QGraphicsScene):
    def __init__(self, face_widget, teaching_canvas, parent=None):
        super().__init__(parent)
        self.setSceneRect(QRectF(0, 0, 1280, 720))

        self.canvas_item = teaching_canvas
        self.addItem(self.canvas_item)

        self.face_proxy = self.addWidget(face_widget)
        self.face_proxy.setZValue(10)
        face_widget.setFixedSize(420, 360)

        # A ring drawn as its own item, not part of FaceWidget itself -
        # setMask() (see FaceWidget.set_circular_mask) only clips the
        # widget, it can't also paint a border right at that clipped edge.
        # Sits above the face (higher z) so it reads as a picture frame
        # around the bubble, not a shape the face is drawn on top of.
        self.face_frame = self.addEllipse(
            0.0, 0.0, float(_BUBBLE_SIZE), float(_BUBBLE_SIZE),
            QPen(_FRAME_COLOR, _FRAME_WIDTH), QBrush(Qt.NoBrush)
        )
        self.face_frame.setZValue(11)
        self.face_frame.setVisible(False)

        self.show_face_fullscreen()

    def show_face_fullscreen(self):
        self.canvas_item.setVisible(False)
        self.face_proxy.widget().set_circular_mask(False)
        self.face_proxy.widget().setFixedSize(1280, 720)
        self.face_proxy.setPos(0, 0)
        self.face_frame.setVisible(False)

    def show_teaching_layout(self):
        self.canvas_item.setVisible(True)
        self.face_proxy.widget().setFixedSize(_BUBBLE_SIZE, _BUBBLE_SIZE)
        self.face_proxy.widget().set_circular_mask(True)
        bubble_x = 1280 - _BUBBLE_SIZE - _BUBBLE_MARGIN
        bubble_y = 720 - _BUBBLE_SIZE - _BUBBLE_MARGIN
        self.face_proxy.setPos(bubble_x, bubble_y)
        # Half the pen width sits outside the ellipse's own bounds, so
        # inset the frame's rect by that much to make the visible ring
        # land exactly on the bubble's edge instead of overhanging it.
        inset = _FRAME_WIDTH / 2
        self.face_frame.setRect(
            bubble_x + inset, bubble_y + inset,
            _BUBBLE_SIZE - _FRAME_WIDTH, _BUBBLE_SIZE - _FRAME_WIDTH,
        )
        self.face_frame.setVisible(True)
