from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsView


class TutorView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(0)
        # White, not black - matches the face art's own white background
        # (see FaceWidget._load) and the whiteboard's own near-white
        # background (TeachingCanvas._BG_COLOR), so the thin letterbox
        # strip fitInView leaves on a screen whose aspect ratio isn't
        # exactly 1280:720 (e.g. a real Pi's 1024x600) blends in instead
        # of reading as a visible black seam.
        self.setStyleSheet("background: white; border: none;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The scene is a fixed 1280x720 "virtual" canvas (see TutorScene).
        # Without this, QGraphicsView renders it at native 1:1 pixel scale
        # regardless of the actual window size - fine only when the window
        # happens to BE exactly 1280x720 (e.g. this project's own dev/test
        # harness, which always uses fullscreen=False + resize(1280, 720)),
        # but wrong on any real screen that isn't that exact resolution -
        # every Pi display, any real fullscreen monitor. Confirmed live on
        # the actual Pi (photo from teammate): the face rendered as a small
        # circle in the middle of a much bigger white area instead of
        # filling the screen's height, because the fixed-size scene was
        # never being scaled up to the real (larger) screen at all.
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
