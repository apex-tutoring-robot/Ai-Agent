from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtGui import QPolygonF
from PyQt5.QtCore import QPointF


class TeachingCanvas(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.mode = "idle"
        self.message = "Waiting..."
        self.lines = []
        self.text_items = []
        self.rect_items = []
        self.circle_items = []
        self.polygon_items = []

    def boundingRect(self):
        return QRectF(0, 0, 1280, 720)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.boundingRect(), Qt.white)

        pen = QPen(Qt.black, 3)
        painter.setPen(pen)

        for (x1, y1, x2, y2) in self.lines:
            painter.drawLine(x1, y1, x2, y2)
            
        for (x, y, w, h) in self.rect_items:
            painter.drawRect(x, y, w, h)
            
        for (x, y, r) in self.circle_items:
            painter.drawEllipse(x, y, r * 2, r * 2)
            
        for points in self.polygon_items:
            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

        for text, x, y in self.text_items:
            if x < 500:
                font = QFont("Arial", 22)
                font.setBold(True)
            else:
                font = QFont("Arial", 16)
                font.setBold(True)
                
            painter.setFont(font)
            painter.drawText(x, y, text)

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def set_message(self, message):
        self.message = message
        self.update()

    def clear_canvas(self):
        self.lines = []
        self.text_items = []
        self.rect_items = []
        self.circle_items = []
        self.polygon_items = []
        self.update()

    def add_line(self, x1, y1, x2, y2):
        self.lines.append((x1, y1, x2, y2))
        self.update()

    def add_text(self, text, x, y):
        self.text_items.append((text, x, y))
        self.update()
        
    def add_rect(self, x, y, w, h):
            self.rect_items.append((x, y, w, h))
            self.update()
            
    def add_circle(self, x, y, r):
            self.circle_items.append((x, y, r))
            self.update()
            
    def add_polygon(self, points):
        self.polygon_items.append(points)
        self.update()
            
    def handle_draw_actions(self, actions):
            for action in actions:
                    action_type = action.get("action")
                    
                    if action_type == "clear":
                            self.clear_canvas()
                        
                    elif action_type == "draw_text":
                            self.add_text(
                                action.get("text", ""),
                                action.get("x", 100),
                                action.get("y", 100)
                            )
                            
                    elif action_type == "draw_line":
                            self.add_line(
                                action.get("x1", 0),
                                action.get("y1", 0),
                                action.get("x2", 100),
                                action.get("y2", 100)
                            )
                            
                    elif action_type == "draw_rect":
                            pass
                            
                    elif action_type == "draw_circle":
                            pass
