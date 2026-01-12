import sys
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter


# ---------- Utilities ----------

def crop_transparent(pixmap: QPixmap) -> QPixmap:
    img = pixmap.toImage()
    rect = img.rect()

    left, top = rect.right(), rect.bottom()
    right, bottom = rect.left(), rect.top()

    for y in range(rect.top(), rect.bottom()):
        for x in range(rect.left(), rect.right()):
            if img.pixelColor(x, y).alpha() > 0:
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    if right < left or bottom < top:
        return pixmap

    return pixmap.copy(left, top, right - left + 1, bottom - top + 1)


def load_and_scale(path: str, target_w: int) -> QPixmap:
    pm = crop_transparent(QPixmap(path))
    if pm.isNull() or pm.width() == 0:
        raise RuntimeError(f"Invalid image: {path}")

    scale = target_w / pm.width()
    return pm.scaled(
        int(pm.width() * scale),
        int(pm.height() * scale),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )


# ---------- Main Window ----------

class FaceWindow(QGraphicsView):
    def __init__(self):
        super().__init__()

        # ---- Canvas ----
        self.CANVAS_W = 1600
        self.CANVAS_H = 900

        # ---- Face size ----
        self.FACE_W = 520
        self.FACE_H = 520

        # ✅ DEFINE ANCHORS FIRST (FIX)
        self.FACE_ANCHOR_X = self.CANVAS_W // 2
        self.FACE_ANCHOR_Y = self.CANVAS_H // 2 - 40

        self.setWindowTitle("AI Agent Face")
        self.setRenderHint(QPainter.Antialiasing)

        # White background (as requested)
        self.setStyleSheet("background-color: white;")
        self.viewport().setStyleSheet("background-color: white;")
        self.setFrameStyle(0)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, self.CANVAS_W, self.CANVAS_H)
        self.scene.setBackgroundBrush(Qt.white)
        self.setScene(self.scene)

        self.load_assets()
        self.setup_scene()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_face)
        self.timer.start(33)

    # ---------- Assets ----------

    def load_assets(self):
        base = "src/assets/"

        eye_w   = int(self.FACE_W * 0.22)
        iris_w  = int(eye_w * 0.55)
        pupil_w = int(iris_w * 0.45)
        brow_w  = int(self.FACE_W * 0.18)
        nose_w  = int(self.FACE_W * 0.08)
        mouth_w = int(self.FACE_W * 0.26)

        self.eye_white_left  = load_and_scale(base + "eye_white_left.png", eye_w)
        self.eye_white_right = load_and_scale(base + "eye_white_right.png", eye_w)

        self.iris_left  = load_and_scale(base + "iris_left.png", iris_w)
        self.iris_right = load_and_scale(base + "iris_right.png", iris_w)

        self.pupil_left  = load_and_scale(base + "pupil_left.png", pupil_w)
        self.pupil_right = load_and_scale(base + "pupil_right.png", pupil_w)

        self.brow_left  = load_and_scale(base + "brow_left.png", brow_w)
        self.brow_right = load_and_scale(base + "brow_right.png", brow_w)

        self.nose  = load_and_scale(base + "nose.png", nose_w)
        base = "src/assets/mouth/"
        self.mouth = load_and_scale(base + "mouth_3.png", mouth_w)

    # ---------- Scene ----------

    def setup_scene(self):
        self.scene.clear()

        # Face container
        self.face_base = QGraphicsPixmapItem(QPixmap(self.FACE_W, self.FACE_H))
        self.face_base.pixmap().fill(Qt.transparent)
        self.scene.addItem(self.face_base)

        self.face_base.setPos(
            self.FACE_ANCHOR_X - self.FACE_W / 2,
            self.FACE_ANCHOR_Y - self.FACE_H / 2
        )

        # Attach parts
        self.eye_white_left_i  = QGraphicsPixmapItem(self.eye_white_left, self.face_base)
        self.eye_white_right_i = QGraphicsPixmapItem(self.eye_white_right, self.face_base)

        self.iris_left_i  = QGraphicsPixmapItem(self.iris_left, self.face_base)
        self.iris_right_i = QGraphicsPixmapItem(self.iris_right, self.face_base)

        self.pupil_left_i  = QGraphicsPixmapItem(self.pupil_left, self.face_base)
        self.pupil_right_i = QGraphicsPixmapItem(self.pupil_right, self.face_base)

        self.brow_left_i  = QGraphicsPixmapItem(self.brow_left, self.face_base)
        self.brow_right_i = QGraphicsPixmapItem(self.brow_right, self.face_base)

        self.nose_i  = QGraphicsPixmapItem(self.nose, self.face_base)
        self.mouth_i = QGraphicsPixmapItem(self.mouth, self.face_base)

        self.layout_face()

    # ---------- Layout ----------

    def layout_face(self):
        fw, fh = self.FACE_W, self.FACE_H

        self.eye_white_left_i.setPos(0.23 * fw, 0.32 * fh)
        self.eye_white_right_i.setPos(0.55 * fw, 0.32 * fh)

        self.iris_left_i.setPos(0.28 * fw, 0.37 * fh)
        self.iris_right_i.setPos(0.60 * fw, 0.37 * fh)

        self.pupil_left_i.setPos(0.31 * fw, 0.40 * fh)
        self.pupil_right_i.setPos(0.63 * fw, 0.40 * fh)

        self.brow_left_i.setPos(0.22 * fw, 0.22 * fh)
        self.brow_right_i.setPos(0.58 * fw, 0.22 * fh)

        self.nose_i.setPos(0.46 * fw, 0.47 * fh)
        self.mouth_i.setPos(0.38 * fw, 0.60 * fh)

    def update_face(self):
        pass


# ---------- Run ----------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceWindow()
    window.show()
    sys.exit(app.exec())
