import os
from PyQt5.QtWidgets import QMainWindow
import math

from visuals.ui.face_widget import FaceWidget
from visuals.ui.teaching_canvas import TeachingCanvas
from visuals.ui.tutor_scene import TutorScene
from visuals.ui.tutor_view import TutorView


class MainWindow(QMainWindow):
    def __init__(self, signals):
        super().__init__()

        # -----------------------------
        # Store signals
        # -----------------------------
        self.signals = signals

        # -----------------------------
        # Load face assets
        # -----------------------------
        base_dir = os.path.dirname(__file__)
        face_dir = os.path.abspath(os.path.join(base_dir, "..", "faces"))

        # -----------------------------
        # Create UI components
        # -----------------------------
        self.face_widget = FaceWidget(face_dir)
        self.canvas = TeachingCanvas()
        self.canvas.clear_canvas()

        self.scene = TutorScene(self.face_widget, self.canvas)
        self.view = TutorView(self.scene)
        
        self.signals.show_face_fullscreen.connect(self.scene.show_face_fullscreen)
        self.signals.show_teaching_layout.connect(self.scene.show_teaching_layout)

        self.setCentralWidget(self.view)

        # -----------------------------
        # Window settings
        # -----------------------------
        self.setWindowTitle("CHIPPY AI Tutor")
        self.resize(1280, 720)

        # -----------------------------
        # Connect signals → UI methods
        # -----------------------------
        self._connect_signals()
        self.signals.draw_actions.connect(self.canvas.handle_draw_actions)

        # -----------------------------
        # Default state
        # -----------------------------
        self.show_idle_mode()

    # =========================================================
    # SIGNAL CONNECTIONS
    # =========================================================
    def _connect_signals(self):
        if not self.signals:
            return

        # High-level modes
        self.signals.listening.connect(self.show_listening_mode)
        self.signals.thinking.connect(self.show_thinking_mode)
        #self.signals.teaching_text.connect(self.show_teaching_mode)
        self.signals.clear_canvas.connect(self.clear_teaching)

        # Face controls
        self.signals.start_talking.connect(self.face_widget.start_talking)
        self.signals.stop_talking.connect(self.face_widget.stop_talking)
        self.signals.mouth_level.connect(self.face_widget.push_mouth_level)
        self.signals.draw_actions.connect(self.handle_draw_actions)
        self.signals.scanning.connect(self.face_widget.start_scanning)

    # =========================================================
    # UI MODES
    # =========================================================

    def show_idle_mode(self):
        """Default resting state"""
        self.face_widget.stop_talking()

    def show_listening_mode(self):
        """User is speaking"""
        self.face_widget.stop_talking()

    def show_thinking_mode(self):
        """LLM is processing"""
        self.face_widget.start_thinking()

    def show_teaching_mode(self, text):
        """Bot is explaining something"""
        self.canvas.clear_canvas()

        # Optional: demo drawing (remove later if needed)
        self.canvas.clear_canvas()
        self.canvas.add_text("Example:", 80, 150)
        self.canvas.add_text(text[:60], 80, 200)

    def clear_teaching(self):
        """Clear drawings"""
        self.canvas.clear_canvas()
        
    def handle_draw_actions(self, actions):
            for action in actions:
                    action_type = action.get("action")
                    
                    if action_type == "clear":
                            self.canvas.clear_canvas()
                            
                    elif action_type == "draw_text":
                            text = action.get("text", "")
                            x = action.get("x", 100)
                            y = action.get("y", 100)
                            self.canvas.add_text(text, x, y)
                            
                    elif action_type == "draw_line":
                            x1 = max(560, min(860, int(action.get("x1", 600))))
                            y1 = max(120, min(420, int(action.get("y1", 200))))
                            x2 = max(560, min(860, int(action.get("x2", 820))))
                            y2 = max(120, min(420, int(action.get("y2", 350))))
                            self.canvas.add_line(x1, y1, x2, y2) 
                            
                    elif action_type == "draw_rect":
                            x = max(520, min(700, int(action.get("x", 600))))
                            y = max(160, min(340, int(action.get("y", 200))))
                            w = max(220, min(320, int(action.get("w", 260))))
                            h = max(140, min(240, int(action.get("h", 180))))
                            self.canvas.add_rect(x, y, w, h)
                            
                    elif action_type == "draw_circle":
                            x = max(540, min(720, int(action.get("x", 620))))
                            y = max(160, min(340, int(action.get("y", 220))))
                            r = max(80, min(120, int(action.get("r", 100))))
                            self.canvas.add_circle(x, y, r)
                            
                    elif action_type == "draw_polygon":
                            points = action.get("points", [])
                            clean_points = []
                            
                            for p in points:
                                    x = max(520, min(860, int(p[0])))
                                    y = max(120, min(460, int(p[1])))
                                    clean_points.append((x, y))
                                    
                            if len(clean_points) >= 3:
                                    self.canvas.add_polygon(clean_points)
                                    
                    elif action_type == "draw_regular_polygon":
                            sides = int(action.get("sides", 5))
                            cx = int(action.get("cx", 700))
                            cy = int(action.get("cy", 260))
                            radius = int(action.get("radius", 125))
                            
                            sides = max(3, min(12, sides))                            
                            points = self.regular_polygon_points(cx, cy, radius, sides)
                            self.canvas.add_polygon(points)

    # =========================================================
    # OPTIONAL HELPERS (FOR FUTURE)
    # =========================================================

    def draw_line(self, x1, y1, x2, y2):
        self.canvas.add_line(x1, y1, x2, y2)

    def draw_text(self, text, x, y):
        self.canvas.add_text(text, x, y)

    def regular_polygon_points(self, cx, cy, radius, sides):
            points = []
            for i in range(sides):
                    angle = 2 * math.pi * i / sides - math.pi / 2
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    points.append((int(x), int(y)))
            return points
