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
