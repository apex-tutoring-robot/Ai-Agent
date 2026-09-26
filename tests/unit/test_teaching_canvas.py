"""
Tests for visuals.ui.teaching_canvas.TeachingCanvas - the animation
progress math and draw_action dispatching. Not covered here: actual
pixel-level QPainter output (paint() itself), since that needs a real
render target and isn't meaningfully assertable in a unit test - the
progress/geometry math and the action-dict routing are what's actually
testable and where a real bug would show up (e.g. wrong duration, wrong
field parsed from an action dict).

Needs a real QApplication since TeachingCanvas is a QGraphicsObject
(QObject-based) - conftest.py in this directory ensures audio.wake_word
(onnxruntime) is imported before this file's PyQt5 import, avoiding the
DLL-conflict segfault documented there.
"""

import time

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from visuals.ui.teaching_canvas import TeachingCanvas, _SHAPE_ANIM_SECONDS


class TestProgress:
    def test_before_start_is_zero(self):
        now = time.monotonic()
        assert TeachingCanvas._progress(start_time=now + 10, now=now) == 0.0

    def test_at_start_is_zero(self):
        now = time.monotonic()
        assert TeachingCanvas._progress(start_time=now, now=now) == 0.0

    def test_halfway_through_duration_is_one_half(self):
        now = time.monotonic()
        start = now - (_SHAPE_ANIM_SECONDS / 2)
        assert abs(TeachingCanvas._progress(start_time=start, now=now) - 0.5) < 1e-6

    def test_at_full_duration_is_one(self):
        now = time.monotonic()
        start = now - _SHAPE_ANIM_SECONDS
        assert TeachingCanvas._progress(start_time=start, now=now) == 1.0

    def test_past_duration_clamps_to_one(self):
        now = time.monotonic()
        start = now - (_SHAPE_ANIM_SECONDS * 10)
        assert TeachingCanvas._progress(start_time=start, now=now) == 1.0

    def test_zero_duration_is_always_complete(self):
        now = time.monotonic()
        assert TeachingCanvas._progress(start_time=now, now=now, duration=0) == 1.0


class TestMutators:
    def test_add_line_stores_coordinates_and_a_start_time(self):
        canvas = TeachingCanvas()
        before = time.monotonic()
        canvas.add_line(1, 2, 3, 4)
        after = time.monotonic()
        assert len(canvas.lines) == 1
        line = canvas.lines[0]
        assert (line.x1, line.y1, line.x2, line.y2) == (1, 2, 3, 4)
        assert before <= line.start_time <= after

    def test_add_circle_defaults_ry_to_rx(self):
        canvas = TeachingCanvas()
        canvas.add_circle(100, 100, 50)
        assert canvas.circle_items[0].rx == 50
        assert canvas.circle_items[0].ry == 50

    def test_add_circle_accepts_distinct_ry_for_ellipses(self):
        canvas = TeachingCanvas()
        canvas.add_circle(100, 100, 50, 30)
        assert canvas.circle_items[0].rx == 50
        assert canvas.circle_items[0].ry == 30

    def test_clear_canvas_empties_everything(self):
        canvas = TeachingCanvas()
        canvas.add_line(0, 0, 1, 1)
        canvas.add_text("hi", 0, 0)
        canvas.add_rect(0, 0, 10, 10)
        canvas.add_circle(0, 0, 10)
        canvas.add_polygon([(0, 0), (1, 0), (1, 1)])
        canvas.add_arc(0, 0, 10, 10)

        canvas.clear_canvas()

        assert canvas.lines == []
        assert canvas.text_items == []
        assert canvas.rect_items == []
        assert canvas.circle_items == []
        assert canvas.polygon_items == []
        assert canvas.arc_items == []


class TestHandleDrawActions:
    def test_clear_action_empties_the_canvas(self):
        canvas = TeachingCanvas()
        canvas.add_line(0, 0, 1, 1)
        canvas.handle_draw_actions([{"action": "clear"}])
        assert canvas.lines == []

    def test_draw_text_action_adds_a_text_item(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_text", "text": "A=12", "x": 50, "y": 60}])
        assert len(canvas.text_items) == 1
        assert canvas.text_items[0].text == "A=12"

    def test_draw_rect_action_adds_a_rect_item(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_rect", "x": 10, "y": 20, "w": 100, "h": 50}])
        item = canvas.rect_items[0]
        assert (item.x, item.y, item.w, item.h) == (10, 20, 100, 50)

    def test_draw_circle_action_supports_r_shorthand(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_circle", "x": 100, "y": 100, "r": 40}])
        item = canvas.circle_items[0]
        assert item.rx == 40 and item.ry == 40

    def test_draw_polygon_action_requires_at_least_three_points(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_polygon", "points": [[0, 0], [10, 0]]}])
        assert canvas.polygon_items == []

    def test_draw_regular_polygon_generates_the_right_number_of_points(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_regular_polygon", "sides": 6, "cx": 700, "cy": 270, "radius": 100}])
        assert len(canvas.polygon_items[0].points) == 6

    def test_draw_regular_polygon_clamps_sides_to_a_sane_range(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_regular_polygon", "sides": 50}])
        assert len(canvas.polygon_items[0].points) == 12

    def test_draw_arc_action_converts_degrees_to_qt_sixteenths(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_arc", "start_angle": 10, "span_angle": 90}])
        item = canvas.arc_items[0]
        assert item.start_angle == 10 * 16
        assert item.span_angle == 90 * 16

    def test_unknown_action_is_silently_ignored(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "levitate"}])
        assert canvas.lines == canvas.text_items == canvas.rect_items == []

    def test_squiggly_underline_action_adds_an_underline_item(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "squiggly_underline", "x": 400, "y": 300, "width": 90}])
        item = canvas.underline_items[0]
        assert (item.x, item.y, item.width) == (400, 300, 90)

    def test_clear_action_also_clears_underlines(self):
        canvas = TeachingCanvas()
        canvas.add_underline(100, 100, 50)
        canvas.handle_draw_actions([{"action": "clear"}])
        assert canvas.underline_items == []


class TestLaserPointerRetargeting:
    def test_laser_is_hidden_before_any_visuals_arrive(self):
        canvas = TeachingCanvas()
        assert canvas._laser_visible is False

    def test_a_text_line_makes_the_laser_visible_and_targets_its_left_side(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_text", "text": "Area = 24", "x": 100, "y": 200}])
        assert canvas._laser_visible is True
        assert canvas._laser_target == (100 - 18, 200)

    def test_a_later_line_in_the_same_batch_wins_the_target(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([
            {"action": "draw_text", "text": "first line", "x": 60, "y": 150},
            {"action": "draw_text", "text": "second line", "x": 60, "y": 210},
        ])
        assert canvas._laser_target == (60 - 18, 210)

    def test_a_shape_with_no_text_does_not_move_the_laser(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_text", "text": "hi", "x": 10, "y": 10}])
        first_target = canvas._laser_target
        canvas.handle_draw_actions([{"action": "draw_rect", "x": 100, "y": 100, "w": 200, "h": 100}])
        assert canvas._laser_target == first_target

    def test_clear_hides_the_laser(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_text", "text": "hi", "x": 10, "y": 10}])
        canvas.handle_draw_actions([{"action": "clear"}])
        assert canvas._laser_visible is False

    def test_a_batch_with_no_visual_content_does_not_move_the_laser(self):
        canvas = TeachingCanvas()
        canvas.handle_draw_actions([{"action": "draw_text", "text": "hi", "x": 10, "y": 10}])
        first_target = canvas._laser_target
        canvas.handle_draw_actions([{"action": "levitate"}])
        assert canvas._laser_target == first_target

    def test_laser_position_starts_at_its_from_point(self):
        canvas = TeachingCanvas()
        assert canvas._current_laser_pos(now=time.monotonic()) == canvas._laser_from
