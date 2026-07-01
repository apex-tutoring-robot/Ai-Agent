from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView


class TutorView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

        from PyQt5.QtGui import QPainter

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(0)
        self.setStyleSheet("background: black; border: none;")

    def resizeEvent(self, event):
        # Scene content is a fixed-size rect (e.g. 1024x600); scale it to fill
        # whatever the real screen/viewport size ends up being (it may not match
        # if the display negotiates a different mode than expected at boot).
        super().resizeEvent(event)
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
