from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsScene


class TutorScene(QGraphicsScene):
    def __init__(self, face_widget, teaching_canvas, parent=None):
        super().__init__(parent)

        self.setSceneRect(QRectF(0, 0, 1280, 720))

        self.canvas_item = teaching_canvas
        self.addItem(self.canvas_item)

        self.face_proxy = self.addWidget(face_widget)
        self.face_proxy.setZValue(10)
        self.face_proxy.setPos(810, 30)
        face_widget.setFixedSize(420, 360)
        self.show_face_fullscreen()
        
    def show_face_fullscreen(self):
            self.canvas_item.setVisible(False)
            self.face_proxy.widget().setFixedSize(1280, 720)
            self.face_proxy.setPos(0, 0)
            
    def show_teaching_layout(self):
            self.canvas_item.setVisible(True)
            self.face_proxy.widget().setFixedSize(400, 400)
            self.face_proxy.setPos(800, 60)
