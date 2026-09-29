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

A title naming what's being taught sits centered at the top of the
board (e.g. "Area of a Rectangle") - set via a "set_title" draw action
the LLM emits alongside "clear" at the start of each new question, so
it flows through the same visuals pipeline as everything else on the
board rather than needing its own wiring.

When a problem describes a real-world object (a car, a person, a ball,
a plant), a simple pencil-style sketch of that object - thin dark
outline, no fill, unlike the colored geometry shapes above - is drawn
and animated (driving/walking/kicking across the board, or growing) to
show the scenario itself, not just abstract shapes. Which icon and
animation to use is decided separately from the LLM by
visuals.scene_planner.ScenePlanner, and arrives here as the same kind
of draw_icon/animate_icon actions as everything else.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QFont, QFontMetrics, QPolygonF, QColor, QPainterPath
from PyQt5.QtWidgets import QGraphicsObject

from visuals.fraction_bar import build_fraction_bar_layout, FractionBarLayout

logger = logging.getLogger(__name__)

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

# Title naming the current topic, centered at the top of the board.
_TITLE_COLOR = QColor(60, 130, 200)
_TITLE_FONT_SIZE = 28
_TITLE_TOP_MARGIN = 15
_TITLE_BAND_HEIGHT = 50

# Squiggly "underline the important part" annotation - like a teacher
# underlining the answer with a thin red marker, drawn left to right at
# roughly the pace of speech, not a quick flash - it should read as
# being drawn WHILE Jarvis explains that part, not before/after it.
_UNDERLINE_COLOR = QColor(225, 30, 30)
_UNDERLINE_STROKE_WIDTH = 2
_UNDERLINE_AMPLITUDE = 4
_UNDERLINE_WAVELENGTH = 18
_UNDERLINE_STEP = 4  # px between sampled points along the wave
_UNDERLINE_ANIM_MIN_SECONDS = 1.4
_UNDERLINE_SECONDS_PER_PX = 0.02

# Laser-pointer dot that tracks the line currently being explained - see
# module docstring. Sits to the LEFT of the line's own x so it never
# overlaps the text itself.
_LASER_COLOR = QColor(255, 20, 20)
_LASER_RADIUS = 7
_LASER_GLOW_RADIUS = 20
_LASER_OFFSET_X = 18
_LASER_ANIM_SECONDS = 0.3
_LASER_PULSE_HZ = 6.0

# Pencil-style sketches of real-world objects (car, person, ball, plant) -
# see visuals.scene_planner.ScenePlanner and module docstring. Thin dark
# outline, no fill, deliberately different from the bold colored/filled
# geometry shapes above so it reads as a quick sketch, not a diagram shape.
_ICON_COLOR = QColor(70, 70, 70)
_ICON_STROKE_WIDTH = 2.5
_ICON_DRAW_SECONDS = 1.1     # time to sketch the icon in, stroke by stroke
_ICON_ANIM_SECONDS = 2.2     # default move/grow duration, once the sketch is done

# Fraction bar - see visuals.fraction_bar and module docstring. Reuses the
# same blue outline / warm fill used for the other filled shapes, so a
# shaded fraction segment reads as "the same kind of shape" as a
# rectangle/circle elsewhere on the board, not a new visual language.
_FRACTION_BAR_GAP = 70          # vertical gap between stacked bars
_FRACTION_BAR_LABEL_FONT_SIZE = 20
_FRACTION_BAR_SECONDS_PER_PART = 0.35


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


@dataclass
class _Icon:
    icon_id: str          # referenced by a later animate_icon action
    name: str              # "car", "person", "ball", "plant" - dispatches to a _draw_*_icon method
    x: float; y: float     # anchor point - bottom-center of the sketch
    scale: float
    start_time: float
    # Animation state - see TeachingCanvas.animate_icon. anim_kind "none"
    # means the icon just sits at (x, scale) once drawn in.
    anim_kind: str = "none"          # "move" (animates x) or "grow" (animates scale)
    anim_from_x: float = 0.0
    anim_to_x: float = 0.0
    anim_from_scale: float = 1.0
    anim_to_scale: float = 1.0
    anim_start_time: float = 0.0
    anim_duration: float = 0.0


@dataclass
class _FractionBar:
    layout: FractionBarLayout  # fully computed by visuals.fraction_bar.build_fraction_bar_layout - no math left to do
    x: float; y: float          # top-left anchor
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
        self.title_text: str = ""
        self.icons: List[_Icon] = []
        self.fraction_bar_items: List[_FractionBar] = []

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
        self._paint_title(painter, rect)

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
        for item in self.icons:
            self._paint_icon(painter, item, now)
        for item in self.fraction_bar_items:
            self._paint_fraction_bar(painter, item, now)
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

    def _paint_title(self, painter: QPainter, rect: QRectF) -> None:
        if not self.title_text:
            return
        font = QFont("Comic Sans MS", _TITLE_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(_TITLE_COLOR))
        title_rect = QRectF(rect.left(), rect.top() + _TITLE_TOP_MARGIN, rect.width(), _TITLE_BAND_HEIGHT)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop, self.title_text)

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

    def _current_icon_state(self, icon: _Icon, now: float) -> Tuple[float, float]:
        """Returns (x, scale) - animated if animate_icon() was called, static otherwise."""
        if icon.anim_kind == "none" or icon.anim_duration <= 0:
            return icon.x, icon.scale
        p = self._progress(icon.anim_start_time, now, duration=icon.anim_duration)
        x = icon.anim_from_x + (icon.anim_to_x - icon.anim_from_x) * p
        scale = icon.anim_from_scale + (icon.anim_to_scale - icon.anim_from_scale) * p
        return x, scale

    def _paint_icon(self, painter: QPainter, icon: _Icon, now: float) -> None:
        # Sketch-in reveal - each stroke draws in turn, like a pencil
        # actually tracing the icon, rather than the whole thing popping
        # in or scaling up from a point. Any move/grow animation (set
        # separately via animate_icon) is timed to start only once this
        # finishes - see animate_icon.
        draw_progress = self._progress(icon.start_time, now, duration=_ICON_DRAW_SECONDS)
        if draw_progress <= 0:
            return
        x, scale = self._current_icon_state(icon, now)
        if scale <= 0:
            return

        painter.setBrush(Qt.NoBrush)

        if icon.name == "car":
            strokes = self._car_icon_strokes(x, icon.y, scale)
        elif icon.name == "person":
            strokes = self._person_icon_strokes(x, icon.y, scale)
        elif icon.name == "ball":
            strokes = self._ball_icon_strokes(x, icon.y, scale)
        elif icon.name == "plant":
            strokes = self._plant_icon_strokes(x, icon.y, scale)
        else:
            return
        self._reveal_strokes(painter, strokes, draw_progress)

    @staticmethod
    def _partial_path(path: QPainterPath, t: float) -> QPainterPath:
        """A polyline approximation of `path` traced up through fraction
        `t` of its length - used to animate any curve (straight or
        bezier) being drawn stroke by stroke."""
        if t >= 1.0:
            return path
        result = QPainterPath()
        steps = 24
        result.moveTo(path.pointAtPercent(0.0))
        for i in range(1, steps + 1):
            result.lineTo(path.pointAtPercent(min(1.0, t * i / steps)))
        return result

    def _reveal_strokes(self, painter: QPainter, strokes: list, progress: float) -> None:
        """Draws a list of ("path"|"line"|"ellipse"|"rounded_rect"|"arc", ...)
        stroke descriptors in order, each getting an equal slice of the
        overall progress - one stroke visibly finishes before the next
        starts, the same "being sketched" feel as a real pencil drawing,
        instead of every part of the icon fading/growing in at once.

        Round caps/joins (not Qt's default sharp miter joins) so lines
        look like they came from a soft pencil, not a CAD drawing - and
        the very first stroke (always the main silhouette) is drawn
        bolder than the interior detail strokes that follow it, the way
        an illustrator weights an outline heavier than interior linework."""
        n = len(strokes)
        if n == 0:
            return
        bold_pen = QPen(_ICON_COLOR, _ICON_STROKE_WIDTH + 1.3)
        bold_pen.setCapStyle(Qt.RoundCap)
        bold_pen.setJoinStyle(Qt.RoundJoin)
        fine_pen = QPen(_ICON_COLOR, _ICON_STROKE_WIDTH - 0.6)
        fine_pen.setCapStyle(Qt.RoundCap)
        fine_pen.setJoinStyle(Qt.RoundJoin)

        for i, stroke in enumerate(strokes):
            local_t = max(0.0, min(1.0, progress * n - i))
            if local_t <= 0:
                break
            painter.setPen(bold_pen if i == 0 else fine_pen)
            kind = stroke[0]
            if kind == "path":
                painter.drawPath(self._partial_path(stroke[1], local_t))
            elif kind == "line":
                _, p1, p2 = stroke
                mid = QPointF(p1.x() + (p2.x() - p1.x()) * local_t, p1.y() + (p2.y() - p1.y()) * local_t)
                painter.drawLine(p1, mid)
            elif kind == "ellipse":
                _, center, rx, ry = stroke
                if local_t >= 1.0:
                    painter.drawEllipse(center, rx, ry)
                else:
                    rect = QRectF(center.x() - rx, center.y() - ry, rx * 2, ry * 2)
                    painter.drawArc(rect, 90 * 16, -int(360 * 16 * local_t))
            elif kind == "rounded_rect":
                _, rect, rx, ry = stroke
                painter.setOpacity(local_t)
                painter.drawRoundedRect(rect, rx, ry)
                painter.setOpacity(1.0)
            elif kind == "arc":
                _, rect, start_angle, span_angle = stroke
                painter.drawArc(rect, start_angle, int(span_angle * local_t))

    def _car_icon_strokes(self, cx: float, ground_y: float, scale: float) -> list:
        """Pencil-sketch car in side profile - a smooth rounded-hatchback
        silhouette (three quadTo arcs sharing endpoints, so the roofline
        forms one continuous dome instead of two curves meeting at a
        sharp kink) plus windows, a door seam, a mirror, a headlight,
        wheel hubs and a ground line - built as an ordered stroke list so
        _reveal_strokes can sketch it in one piece at a time, body first."""
        s = scale
        rear_bottom = QPointF(cx - 65 * s, ground_y - 24 * s)
        rear_top = QPointF(cx - 40 * s, ground_y - 60 * s)
        rear_ctrl = QPointF(cx - 65 * s, ground_y - 55 * s)
        dome_ctrl = QPointF(cx - 8 * s, ground_y - 82 * s)
        front_top = QPointF(cx + 25 * s, ground_y - 60 * s)
        front_ctrl = QPointF(cx + 60 * s, ground_y - 50 * s)
        front_bottom = QPointF(cx + 68 * s, ground_y - 24 * s)

        body = QPainterPath()
        body.moveTo(rear_bottom)
        body.quadTo(rear_ctrl, rear_top)
        body.quadTo(dome_ctrl, front_top)
        body.quadTo(front_ctrl, front_bottom)
        body.lineTo(rear_bottom)

        door_top = QPointF(cx + 2 * s, ground_y - 64 * s)
        door_bottom = QPointF(cx + 4 * s, ground_y - 24 * s)
        door = QPainterPath()
        door.moveTo(door_top)
        door.quadTo(QPointF(cx, ground_y - 45 * s), door_bottom)

        rear_win = QRectF(cx - 36 * s, ground_y - 64 * s, 34 * s, 22 * s)
        front_win = QRectF(cx + 6 * s, ground_y - 64 * s, 20 * s, 22 * s)
        wheel_r, hub_r = 12 * s, 5 * s
        wheel_y = ground_y - wheel_r
        rear_wheel_x, front_wheel_x = cx - 34 * s, cx + 34 * s

        return [
            ("path", body),
            ("path", door),
            ("rounded_rect", rear_win, 4 * s, 4 * s),
            ("rounded_rect", front_win, 4 * s, 4 * s),
            ("line", QPointF(cx + 20 * s, ground_y - 58 * s), QPointF(cx + 30 * s, ground_y - 55 * s)),
            ("ellipse", QPointF(cx + 58 * s, ground_y - 34 * s), 3.5 * s, 3.5 * s),
            ("ellipse", QPointF(rear_wheel_x, wheel_y), wheel_r, wheel_r),
            ("ellipse", QPointF(rear_wheel_x, wheel_y), hub_r, hub_r),
            ("ellipse", QPointF(front_wheel_x, wheel_y), wheel_r, wheel_r),
            ("ellipse", QPointF(front_wheel_x, wheel_y), hub_r, hub_r),
            ("line", QPointF(rear_bottom.x() - 20 * s, ground_y), QPointF(front_bottom.x() + 20 * s, ground_y)),
        ]

    def _person_icon_strokes(self, cx: float, ground_y: float, scale: float) -> list:
        """Pencil-sketch figure mid-stride, anchored at its feet - a curved
        torso and bent limbs (quadTo) instead of a rigid stick figure, with
        a small hair tuft and a ground shadow for grounding."""
        s = scale
        head_r = 9 * s
        head_cy = ground_y - 70 * s
        neck_y = head_cy + head_r
        hip_y = ground_y - 26 * s
        shoulder_y = neck_y + 6 * s

        torso = QPainterPath()
        torso.moveTo(cx, neck_y)
        torso.quadTo(cx - 3 * s, (neck_y + hip_y) / 2, cx, hip_y)

        return [
            ("ellipse", QPointF(cx, head_cy), head_r, head_r),
            ("arc", QRectF(cx - head_r, head_cy - head_r * 1.3, head_r * 2, head_r * 1.4), 20 * 16, 140 * 16),
            ("path", torso),
            ("line", QPointF(cx, shoulder_y), QPointF(cx - 20 * s, shoulder_y + 14 * s)),
            ("line", QPointF(cx - 20 * s, shoulder_y + 14 * s), QPointF(cx - 14 * s, shoulder_y + 26 * s)),
            ("line", QPointF(cx, shoulder_y), QPointF(cx + 18 * s, shoulder_y + 10 * s)),
            # Mid-stride, feet spread wide apart (~58s) at the ground - at
            # this icon's small on-screen size, anything narrower visually
            # merges into a solid wedge instead of reading as two legs.
            ("line", QPointF(cx, hip_y), QPointF(cx - 18 * s, hip_y + 14 * s)),
            ("line", QPointF(cx - 18 * s, hip_y + 14 * s), QPointF(cx - 30 * s, ground_y)),
            ("line", QPointF(cx, hip_y), QPointF(cx + 14 * s, hip_y + 18 * s)),
            ("line", QPointF(cx + 14 * s, hip_y + 18 * s), QPointF(cx + 28 * s, ground_y)),
            ("ellipse", QPointF(cx - 30 * s, ground_y + 2 * s), 6 * s, 2 * s),
            ("ellipse", QPointF(cx + 28 * s, ground_y + 2 * s), 6 * s, 2 * s),
        ]

    def _ball_icon_strokes(self, cx: float, ground_y: float, scale: float) -> list:
        """Pencil-sketch ball with two crossing lens-shaped seam curves
        (the classic way to suggest a sphere's curved surface - tracing
        the ball's own outer edge would be invisible against its own
        outline) and a ground shadow, resting on the ground line."""
        s = scale
        r = 20 * s
        cy = ground_y - r
        center = QPointF(cx, cy)
        return [
            ("ellipse", center, r, r),
            ("ellipse", center, r * 0.32, r),
            ("ellipse", center, r, r * 0.32),
            ("ellipse", QPointF(cx, ground_y + 2 * s), r * 0.8, 3 * s),
        ]

    def _plant_icon_strokes(self, cx: float, ground_y: float, scale: float) -> list:
        """Pencil-sketch sprout with a curved stem, leaves, a ring of petals,
        and a soil mound - scale drives the "grow" animation, so this needs
        to look right small too, not just at full size."""
        s = scale
        stem_h = 60 * s
        top_y = ground_y - stem_h
        stem = QPainterPath()
        stem.moveTo(cx, ground_y - 4 * s)
        stem.quadTo(cx - 6 * s, ground_y - stem_h * 0.5, cx, top_y)

        leaf_y = ground_y - stem_h * 0.55
        petal_r = 6 * s
        strokes = [
            ("path", stem),
            ("ellipse", QPointF(cx - 14 * s, leaf_y), 14 * s, 7 * s),
            ("ellipse", QPointF(cx + 14 * s, leaf_y - 8 * s), 14 * s, 7 * s),
        ]
        for angle_deg in (0, 60, 120, 180, 240, 300):
            rad = math.radians(angle_deg)
            strokes.append((
                "ellipse",
                QPointF(cx + petal_r * 1.3 * math.cos(rad), top_y + petal_r * 1.3 * math.sin(rad)),
                petal_r * 0.6, petal_r * 0.6
            ))
        strokes.append(("ellipse", QPointF(cx, top_y), petal_r * 0.7, petal_r * 0.7))
        strokes.append(("arc", QRectF(cx - 18 * s, ground_y - 4 * s, 36 * s, 10 * s), 0, -180 * 16))
        return strokes

    def _paint_fraction_bar(self, painter: QPainter, item: _FractionBar, now: float) -> None:
        """Reveals each bar in turn (if there's more than one, for a
        comparison) - first its outline/dividers/label (like a rectangle
        being drawn and split into parts), then its shaded segments one
        at a time left to right (like a marker coloring them in). Every
        segment boundary and shaded/unshaded flag came from
        visuals.fraction_bar.build_fraction_bar_layout(), never computed
        here or by the LLM."""
        layout = item.layout
        n_parts = sum(1 + bar.numerator for bar in layout.bars)
        duration = max(1.2, n_parts * _FRACTION_BAR_SECONDS_PER_PART)
        p = self._progress(item.start_time, now, duration=duration)
        if p <= 0:
            return

        label_font = QFont("Comic Sans MS", _FRACTION_BAR_LABEL_FONT_SIZE)
        label_font.setBold(True)

        def part_alpha(i):
            return max(0.0, min(1.0, p * n_parts - i))

        idx = 0
        row_y = item.y
        for bar in layout.bars:
            bar_top = row_y
            bar_bottom = row_y + layout.bar_height

            struct_alpha = part_alpha(idx); idx += 1
            if struct_alpha > 0:
                painter.setOpacity(struct_alpha)
                painter.setFont(label_font)
                painter.setPen(QPen(_TEXT_COLOR))
                painter.drawText(QPointF(item.x, bar_top - 10), bar.label)

                painter.setPen(QPen(_SHAPE_OUTLINE, 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(item.x, bar_top, layout.width, layout.bar_height))
                for seg in bar.segments[:-1]:
                    sx = item.x + seg.x1
                    painter.drawLine(QPointF(sx, bar_top), QPointF(sx, bar_bottom))

            for seg in bar.segments:
                if not seg.shaded:
                    continue
                seg_alpha = part_alpha(idx); idx += 1
                if seg_alpha <= 0:
                    continue
                painter.setOpacity(seg_alpha)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(_SHAPE_FILL))
                painter.drawRect(QRectF(
                    item.x + seg.x0 + 1, bar_top + 1,
                    seg.x1 - seg.x0 - 2, layout.bar_height - 2
                ))

            row_y += layout.bar_height + _FRACTION_BAR_GAP

        painter.setOpacity(1.0)

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
        # Scales with span so a longer underline takes proportionally
        # longer, same idea as _paint_text's typewriter pacing - this is
        # what makes it read as being drawn WHILE the line is explained
        # rather than flashing in before the explanation even starts.
        duration = max(_UNDERLINE_ANIM_MIN_SECONDS, item.width * _UNDERLINE_SECONDS_PER_PX)
        p = self._progress(item.start_time, now, duration=duration)
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
        self.icons = []
        self.fraction_bar_items = []
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

    def set_title(self, text):
        self.title_text = text
        self.update()

    def add_icon(self, icon_id, name, x, y, scale=1.0):
        self.icons.append(_Icon(icon_id=icon_id, name=name, x=x, y=y, scale=scale, start_time=time.monotonic()))
        self.update()

    def add_fraction_bar(self, fractions, width, bar_height, x, y):
        """Validates + computes via visuals.fraction_bar.
        build_fraction_bar_layout(), then stores the result for rendering
        - no math or coordinate decisions happen here or in the LLM's own
        output. Silently skips an invalid spec (same pattern as
        add_vertical_arithmetic), logging a warning so a bad LLM output
        is visible without crashing the turn."""
        try:
            layout = build_fraction_bar_layout(fractions, width, bar_height)
        except ValueError as e:
            logger.warning(f"Skipping invalid fraction_bar action: {e}")
            return
        self.fraction_bar_items.append(_FractionBar(layout, x, y, time.monotonic()))
        self.update()

    def animate_icon(self, icon_id, animation, duration, to_x=None, to_scale=None):
        """Moves (animation="move", varies x) or grows (animation="grow",
        varies scale) a previously add_icon'd icon, found by icon_id.
        Starts only once that icon's own sketch-in reveal has finished -
        see _ICON_DRAW_SECONDS - so it doesn't look like it's flying
        across the board while still being drawn."""
        icon = next((i for i in self.icons if i.icon_id == icon_id), None)
        if icon is None:
            return
        icon.anim_kind = animation
        icon.anim_from_x = icon.x
        icon.anim_to_x = to_x if to_x is not None else icon.x
        icon.anim_from_scale = icon.scale
        icon.anim_to_scale = to_scale if to_scale is not None else icon.scale
        icon.anim_start_time = max(time.monotonic(), icon.start_time + _ICON_DRAW_SECONDS)
        icon.anim_duration = max(0.1, duration)
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

            elif action_type == "set_title":
                self.set_title(action.get("text", ""))

            elif action_type == "draw_icon":
                self.add_icon(
                    action.get("id", ""), action.get("icon", "car"),
                    action.get("x", 620), action.get("y", 340),
                    action.get("scale", 1.0)
                )

            elif action_type == "animate_icon":
                self.animate_icon(
                    action.get("id", ""), action.get("animation", "move"),
                    action.get("duration", _ICON_ANIM_SECONDS),
                    to_x=action.get("to_x"), to_scale=action.get("to_scale")
                )

            elif action_type == "fraction_bar":
                self.add_fraction_bar(
                    action.get("fractions", []), action.get("width", 600), action.get("bar_height", 50),
                    action.get("x", 100), action.get("y", 200)
                )

        if text_positions:
            x, y = text_positions[-1]
            self._retarget_laser((x - _LASER_OFFSET_X, y), time.monotonic())
