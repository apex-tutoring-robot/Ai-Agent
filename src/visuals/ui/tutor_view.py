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
        # background (TeachingCanvas._BG_COLOR), so a KeepAspectRatio
        # letterbox strip (teaching-layout mode, see below) blends in
        # instead of reading as a visible black seam.
        self.setStyleSheet("background: white; border: none;")

        # Whether to stretch the scene to fill the window exactly
        # (IgnoreAspectRatio) or preserve its aspect ratio with letterbox
        # margins (KeepAspectRatio) - see set_fill_mode(). Starts matching
        # TutorScene's own initial state (it calls show_face_fullscreen()
        # in its own __init__ before this view exists) - kept in sync by
        # MainWindow calling set_fill_mode() alongside every scene mode
        # switch, not inferred here.
        self._stretch_to_fill = True

    def set_fill_mode(self, stretch_to_fill: bool) -> None:
        """stretch_to_fill=True: the scene fills the window exactly, no
        margins, by stretching (distorts proportions) - used for the
        full-screen face, which has no content whose exact geometry
        matters. stretch_to_fill=False: the scene keeps its own aspect
        ratio, letterboxed if needed - used for the teaching layout, so
        the whiteboard's shapes/text aren't skewed on a screen whose
        aspect ratio isn't exactly the scene's fixed 1280:720 (e.g. a
        real Pi's 1024x600)."""
        self._stretch_to_fill = stretch_to_fill
        self._apply_fit()

    def _apply_fit(self) -> None:
        mode = Qt.IgnoreAspectRatio if self._stretch_to_fill else Qt.KeepAspectRatio
        self.fitInView(self.scene().sceneRect(), mode)

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
        self._apply_fit()
