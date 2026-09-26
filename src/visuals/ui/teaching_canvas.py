"""
Drawable whiteboard for math/geometry teaching. Stores drawn primitives as
typed dataclasses and paints them via QPainter - driven by draw_action
dicts that arrive from LLMClient.generate_teaching_plan() via JarvisBot's
teaching turn, routed through UISignals.draw_actions for thread-safety.

Built for a 3rd-5th grade audience: a colored, framed "whiteboard" look
(not a flat white rectangle), filled/colored shapes (not a black-on-white
wireframe), and every primitive animates in over a short duration - a
line grows from its start point, a circle sweeps around like it's being
traced, a rectangle/polygon draws edge by edge, text types on - with a
small marker-dot following the current drawing point, instead of
everything popping into existence instantly. Runs its own 60fps QTimer
(same idea as FaceWidget's) to drive that animation, which is why this is
a QGraphicsObject (a QObject-based QGraphicsItem) rather than a plain
QGraphicsItem - it needs somewhere to own that timer.

Important parts get a squiggly red underline drawn beneath them (like a
teacher underlining the answer with a marker), animated left to right -
not a circle around them, which read as visually noisy in practice.

A glowing red laser-pointer dot sits just left of whichever line is
currently being explained, and glides to the next line's position each
time a new speech step draws a new line of text - like a presenter
tracking their own explanation with a laser pointer, rather than a
wooden stick (tried first, then dropped) pointing at whole diagrams.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QFont, QPolygonF, QColor
from PyQt5.QtWidgets import QGraphicsObject

# Kid-friendly palette: warm paper background (not stark white), a
# friendly blue board frame, blue shape outlines with a warm translucent
# fill (so shapes read as solid, not wireframe), dark marker-black text
# for equations/labels, and a bright orange "marker tip" that traces
# each shape as it draws. Text is deliberately NOT red - red is reserved
# for the underline/laser emphasis below, so it actually stands out
# instead of blending in with every other letter on the board.
_BG_COLOR = QColor(250, 250, 245)
_GRID_COLOR = QColor(222, 233, 245)
_FRAME_COLOR = QColor(60, 130, 200)
_SHAPE_OUTLINE = QColor(40, 100, 170)
_SHAPE_FILL = QColor(255, 205, 90, 140)
_TEXT_COLOR = QColor(35, 40, 50)
_MARKER_COLOR = QColor(255, 140, 0)

_SHAPE_ANIM_SECONDS = 0.55
_GRID_SPACING = 40
_MARKER_RADIUS = 7

# Squiggly "underline the important part" annotation - like a teacher
# underlining the answer with a red marker, drawn left to right.
_UNDERLINE_COLOR = QColor(225, 30, 30)
_UNDERLINE_STROKE_WIDTH = 4
_UNDERLINE_AMPLITUDE = 5
_UNDERLINE_WAVELENGTH = 18
_UNDERLINE_STEP = 4  # px between sampled points along the wave
_UNDERLINE_ANIM_SECONDS = 0.4

# Laser-pointer dot that tracks the line currently being explained - see
# module docstring. Sits to the LEFT of the line's own x so it never
# overlaps the text itself.
_LASER_COLOR = QColor(255, 20, 20)
_LASER_RADIUS = 7
_LASER_GLOW_RADIUS = 20
_LASER_OFFSET_X = 18
_LASER_ANIM_SECONDS = 0.3
_LASER_PULSE_HZ = 6.0

# Decorative marker tray, bottom-left corner - static board chrome (like
# the frame/grid), not part of the animated lesson content, so it's
# painted every frame regardless of clear_canvas().
_TRAY_COLORS = [QColor(215, 35, 35), QColor(35, 95, 200), QColor(30, 30, 35)]  # red, blue, black
_TRAY_BARREL_COLOR = QColor(245, 245, 240)
_TRAY_BARREL_OUTLINE = QColor(120, 120, 120)
_TRAY_PEN_LENGTH = 78
_TRAY_PEN_WIDTH = 15
_TRAY_TIP_LENGTH = 16
_TRAY_PEN_SPACING = 26
_TRAY_X = 26
_TRAY_Y = 655


@dataclass
class _Line:
    x1: float; y1: float; x2: float; y2: float
    start_time: float


@dataclass
class _Text:
    text: str; x: float; y: float
    start_time: float


@dataclass
class _Rect:
    x: float; y: float; w: float; h: float
    start_time: float


@dataclass
class _Circle:
    x: float; y: float; rx: float; ry: float
    start_time: float


@dataclass
class _Polygon:
    points: List[Tuple[float, float]]
    start_time: float


@dataclass
class _Arc:
    x: float; y: float; w: float; h: float
    start_angle: int; span_angle: int  # 1/16th degree, Qt convention
    start_time: float


@dataclass
class _Underline:
    x: float; y: float; width: float  # left edge, baseline, span to the right
    start_time: float


class TeachingCanvas(QGraphicsObject):
    def __init__(self):
        super().__init__()
        self.lines: List[_Line] = []
        self.text_items: List[_Text] = []
        self.rect_items: List[_Rect] = []
        self.circle_items: List[_Circle] = []
        self.polygon_items: List[_Polygon] = []
        self.arc_items: List[_Arc] = []
        self.underline_items: List[_Underline] = []

        # Laser-pointer dot state - see module docstring and _retarget_laser.
        self._laser_visible = False
        self._laser_from: Tuple[float, float] = (0.0, 0.0)
        self._laser_target: Tuple[float, float] = (0.0, 0.0)
        self._laser_start_time = 0.0

        # Drives the draw-in animations - see module docstring.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // 60)

    def boundingRect(self):
        return QRectF(0, 0, 1280, 720)

    # --------------------------------------------------
    # Painting
    # --------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        now = time.monotonic()
        rect = self.boundingRect()

        painter.fillRect(rect, _BG_COLOR)
        self._paint_grid(painter, rect)
        self._paint_frame(painter, rect)
        self._paint_marker_tray(painter)

        for line in self.lines:
            self._paint_line(painter, line, now)
        for item in self.rect_items:
            self._paint_rect(painter, item, now)
        for item in self.circle_items:
            self._paint_circle(painter, item, now)
        for item in self.polygon_items:
            self._paint_polygon(painter, item, now)
        for item in self.arc_items:
            self._paint_arc(painter, item, now)
        for item in self.text_items:
            self._paint_text(painter, item, now)
        for item in self.underline_items:
            self._paint_underline(painter, item, now)

        # Drawn last so the pointer dot reads as sitting on top of
        # everything else on the board, like a real laser dot would.
        self._paint_laser(painter, now)

    def _paint_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(_GRID_COLOR, 1))
        x = rect.left() + _GRID_SPACING
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += _GRID_SPACING
        y = rect.top() + _GRID_SPACING
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += _GRID_SPACING

    def _paint_frame(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(_FRAME_COLOR, 6))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3), 18, 18)

    def _paint_marker_tray(self, painter: QPainter) -> None:
        """Three resting markers (red/blue/black) in the bottom-left corner -
        decorative chrome, same idea as the grid/frame, always drawn."""
        for i, color in enumerate(_TRAY_COLORS):
            y = _TRAY_Y + i * _TRAY_PEN_SPACING
            barrel = QRectF(_TRAY_X, y, _TRAY_PEN_LENGTH, _TRAY_PEN_WIDTH)
            painter.setPen(QPen(_TRAY_BARREL_OUTLINE, 1.5))
            painter.setBrush(QBrush(_TRAY_BARREL_COLOR))
            painter.drawRoundedRect(barrel, _TRAY_PEN_WIDTH / 2, _TRAY_PEN_WIDTH / 2)

            tip = QRectF(
                _TRAY_X + _TRAY_PEN_LENGTH - _TRAY_TIP_LENGTH, y,
                _TRAY_TIP_LENGTH, _TRAY_PEN_WIDTH
            )
            painter.setPen(QPen(color.darker(130), 1.5))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(tip, _TRAY_PEN_WIDTH / 2, _TRAY_PEN_WIDTH / 2)

    @staticmethod
    def _progress(start_time: float, now: float, duration: float = _SHAPE_ANIM_SECONDS) -> float:
        if duration <= 0:
            return 1.0
        return max(0.0, min(1.0, (now - start_time) / duration))

    def _draw_marker(self, painter: QPainter, x: float, y: float) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_MARKER_COLOR))
        painter.drawEllipse(QPointF(x, y), _MARKER_RADIUS, _MARKER_RADIUS)

    def _paint_line(self, painter: QPainter, item: _Line, now: float) -> None:
        p = self._progress(item.start_time, now)
        ex = item.x1 + (item.x2 - item.x1) * p
        ey = item.y1 + (item.y2 - item.y1) * p
        painter.setPen(QPen(_SHAPE_OUTLINE, 4))
        painter.drawLine(QPointF(item.x1, item.y1), QPointF(ex, ey))
        if 0 < p < 1:
            self._draw_marker(painter, ex, ey)

    def _paint_edge_traced_shape(self, painter: QPainter, corners: List[Tuple[float, float]], p: float,
                                  closed_shape_drawer) -> None:
        """
        Shared "draw edge by edge, like a hand tracing the perimeter"
        animation for anything made of straight segments (rectangles,
        polygons) - once fully drawn, hands off to `closed_shape_drawer`
        to paint the final filled shape in one call.
        """
        if p >= 1.0:
            painter.setBrush(QBrush(_SHAPE_FILL))
            closed_shape_drawer()
            return
        painter.setBrush(Qt.NoBrush)
        edge_count = len(corners) - 1
        edge_progress = p * edge_count
        for i in range(edge_count):
            seg_p = max(0.0, min(1.0, edge_progress - i))
            if seg_p <= 0:
                break
            (sx, sy), (ex, ey) = corners[i], corners[i + 1]
            tx = sx + (ex - sx) * seg_p
            ty = sy + (ey - sy) * seg_p
            painter.drawLine(QPointF(sx, sy), QPointF(tx, ty))
            if seg_p < 1.0:
                self._draw_marker(painter, tx, ty)

    def _paint_rect(self, painter: QPainter, item: _Rect, now: float) -> None:
        p = self._progress(item.start_time, now)
        x, y, w, h = item.x, item.y, item.w, item.h
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        painter.setPen(QPen(_SHAPE_OUTLINE, 4))
        self._paint_edge_traced_shape(painter, corners, p, lambda: painter.drawRect(QRectF(x, y, w, h)))

    def _paint_polygon(self, painter: QPainter, item: _Polygon, now: float) -> None:
        p = self._progress(item.start_time, now)
        pts = item.points
        corners = list(pts) + [pts[0]]
        painter.setPen(QPen(_SHAPE_OUTLINE, 4))
        self._paint_edge_traced_shape(
            painter, corners, p,
            lambda: painter.drawPolygon(QPolygonF([QPointF(px, py) for px, py in pts]))
        )

    def _paint_circle(self, painter: QPainter, item: _Circle, now: float) -> None:
        p = self._progress(item.start_time, now)
        x, y, rx, ry = item.x, item.y, item.rx, item.ry
        rect = QRectF(x - rx, y - ry, rx * 2, ry * 2)
        painter.setPen(QPen(_SHAPE_OUTLINE, 4))
        if p >= 1.0:
            painter.setBrush(QBrush(_SHAPE_FILL))
            painter.drawEllipse(rect)
            return
        painter.setBrush(Qt.NoBrush)
        # Sweeps clockwise starting at the top (Qt: 0 deg = 3 o'clock,
        # positive = counter-clockwise, angles in 1/16th degree).
        span = int(360 * 16 * p)
        painter.drawArc(rect, 90 * 16, -span)
        angle_rad = math.radians(90 - 360 * p)
        mx = x + rx * math.cos(angle_rad)
        my = y - ry * math.sin(angle_rad)
        self._draw_marker(painter, mx, my)

    def _paint_arc(self, painter: QPainter, item: _Arc, now: float) -> None:
        p = self._progress(item.start_time, now)
        rect = QRectF(item.x, item.y, item.w, item.h)
        painter.setPen(QPen(_SHAPE_OUTLINE, 4))
        painter.setBrush(Qt.NoBrush)
        span = int(item.span_angle * p)
        painter.drawArc(rect, item.start_angle, span)
        if 0 < p < 1:
            cx, cy = rect.center().x(), rect.center().y()
            rx, ry = rect.width() / 2, rect.height() / 2
            angle_rad = math.radians((item.start_angle + span) / 16.0)
            mx = cx + rx * math.cos(angle_rad)
            my = cy - ry * math.sin(angle_rad)
            self._draw_marker(painter, mx, my)

    def _paint_text(self, painter: QPainter, item: _Text, now: float) -> None:
        # Typewriter reveal - duration scales with length so a short
        # label and a long equation both feel like they're being written
        # at a similar, natural pace rather than a fixed time each.
        duration = max(0.35, len(item.text) * 0.035)
        p = self._progress(item.start_time, now, duration=duration)
        visible = item.text[: int(len(item.text) * p)]
        font = QFont("Comic Sans MS", 22 if item.x < 500 else 17)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_COLOR))
        painter.drawText(QPointF(item.x, item.y), visible)

    def _paint_underline(self, painter: QPainter, item: _Underline, now: float) -> None:
        p = self._progress(item.start_time, now, duration=_UNDERLINE_ANIM_SECONDS)
        visible_width = item.width * p
        if visible_width <= 0:
            return
        pen = QPen(_UNDERLINE_COLOR, _UNDERLINE_STROKE_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        prev_point = None
        x = 0.0
        while x <= visible_width:
            wave_y = item.y + _UNDERLINE_AMPLITUDE * math.sin(2 * math.pi * x / _UNDERLINE_WAVELENGTH)
            point = QPointF(item.x + x, wave_y)
            if prev_point is not None:
                painter.drawLine(prev_point, point)
            prev_point = point
            x += _UNDERLINE_STEP

    def _current_laser_pos(self, now: float) -> Tuple[float, float]:
        p = self._progress(self._laser_start_time, now, duration=_LASER_ANIM_SECONDS)
        fx, fy = self._laser_from
        tx, ty = self._laser_target
        return (fx + (tx - fx) * p, fy + (ty - fy) * p)

    def _retarget_laser(self, target: Tuple[float, float], now: float) -> None:
        self._laser_from = self._current_laser_pos(now)
        self._laser_target = target
        self._laser_start_time = now
        self._laser_visible = True

    def _hide_laser(self) -> None:
        self._laser_visible = False

    def _paint_laser(self, painter: QPainter, now: float) -> None:
        if not self._laser_visible:
            return
        x, y = self._current_laser_pos(now)
        # Slight pulsing so the dot reads as "glowing", not a flat sticker.
        pulse = 1.0 + 0.15 * math.sin(now * _LASER_PULSE_HZ)
        painter.setPen(Qt.NoPen)

        outer = QColor(_LASER_COLOR)
        outer.setAlpha(55)
        painter.setBrush(QBrush(outer))
        painter.drawEllipse(QPointF(x, y), _LASER_GLOW_RADIUS * pulse, _LASER_GLOW_RADIUS * pulse)

        mid = QColor(_LASER_COLOR)
        mid.setAlpha(130)
        painter.setBrush(QBrush(mid))
        painter.drawEllipse(QPointF(x, y), _LASER_GLOW_RADIUS * 0.55 * pulse, _LASER_GLOW_RADIUS * 0.55 * pulse)

        painter.setBrush(QBrush(_LASER_COLOR))
        painter.drawEllipse(QPointF(x, y), _LASER_RADIUS, _LASER_RADIUS)

    # --------------------------------------------------
    # Mutators (see handle_draw_actions for the dict-based entry point)
    # --------------------------------------------------

    def clear_canvas(self):
        self.lines = []
        self.text_items = []
        self.rect_items = []
        self.circle_items = []
        self.polygon_items = []
        self.arc_items = []
        self.underline_items = []
        self._hide_laser()
        self.update()

    def add_line(self, x1, y1, x2, y2):
        self.lines.append(_Line(x1, y1, x2, y2, time.monotonic()))
        self.update()

    def add_text(self, text, x, y):
        self.text_items.append(_Text(text, x, y, time.monotonic()))
        self.update()

    def add_rect(self, x, y, w, h):
        self.rect_items.append(_Rect(x, y, w, h, time.monotonic()))
        self.update()

    def add_circle(self, x, y, rx, ry=None):
        """x, y = center. rx = x-radius, ry = y-radius (defaults to rx for circles)."""
        if ry is None:
            ry = rx
        self.circle_items.append(_Circle(x, y, rx, ry, time.monotonic()))
        self.update()

    def add_polygon(self, points):
        self.polygon_items.append(_Polygon(list(points), time.monotonic()))
        self.update()

    def add_arc(self, x, y, w, h, start_angle=0, span_angle=5760):
        """Draw an arc/ellipse. Angles in 1/16th of a degree (Qt convention)."""
        self.arc_items.append(_Arc(x, y, w, h, start_angle, span_angle, time.monotonic()))
        self.update()

    def add_underline(self, x, y, width):
        """x, y = left edge / baseline to draw the squiggle under. width =
        how far right it spans (roughly the width of the text/value above it)."""
        self.underline_items.append(_Underline(x, y, width, time.monotonic()))
        self.update()

    def handle_draw_actions(self, actions):
        """Entry point for UISignals.draw_actions - a list of action dicts."""
        # Tracks each draw_text line drawn in this batch so the laser
        # pointer can retarget to the last (i.e. current) one - each call
        # here is one speech step's worth of visuals, so the last text
        # line added is the one Jarvis is now explaining.
        text_positions: List[Tuple[float, float]] = []

        for action in actions:
            action_type = action.get("action")

            if action_type == "clear":
                self.clear_canvas()

            elif action_type == "draw_text":
                x, y = action.get("x", 100), action.get("y", 100)
                self.add_text(action.get("text", ""), x, y)
                text_positions.append((x, y))

            elif action_type == "draw_line":
                self.add_line(
                    action.get("x1", 0), action.get("y1", 0),
                    action.get("x2", 100), action.get("y2", 100)
                )

            elif action_type == "draw_rect":
                self.add_rect(
                    action.get("x", 100), action.get("y", 100),
                    action.get("w", 200), action.get("h", 150)
                )

            elif action_type == "draw_circle":
                # Support both r (circle) and rx/ry (ellipse)
                rx = action.get("rx", action.get("r", 80))
                ry = action.get("ry", rx)
                self.add_circle(action.get("x", 700), action.get("y", 270), rx, ry)

            elif action_type == "draw_polygon":
                points = action.get("points", [])
                if len(points) >= 3:
                    self.add_polygon([(p[0], p[1]) for p in points])

            elif action_type == "draw_regular_polygon":
                sides = max(3, min(12, int(action.get("sides", 5))))
                cx = action.get("cx", 700)
                cy = action.get("cy", 270)
                radius = action.get("radius", 120)
                points = []
                for i in range(sides):
                    angle = 2 * math.pi * i / sides - math.pi / 2
                    points.append((int(cx + radius * math.cos(angle)), int(cy + radius * math.sin(angle))))
                self.add_polygon(points)

            elif action_type == "draw_arc":
                self.add_arc(
                    action.get("x", 600), action.get("y", 160),
                    action.get("w", 200), action.get("h", 200),
                    int(action.get("start_angle", 0) * 16),
                    int(action.get("span_angle", 360) * 16)
                )

            elif action_type == "squiggly_underline":
                self.add_underline(
                    action.get("x", 100), action.get("y", 100), action.get("width", 100)
                )

        if text_positions:
            x, y = text_positions[-1]
            self._retarget_laser((x - _LASER_OFFSET_X, y), time.monotonic())
