"""
Tests for visuals.scene_planner.ScenePlanner - the keyword-based decision
of which real-world-object icon (and animation) a problem's text calls
for, independent of the LLM's own teaching plan. Pure Python, no PyQt5 -
doesn't need conftest.py's import-order guard.
"""

from visuals.scene_planner import ScenePlanner


class TestScenePlanner:
    def test_car_problem_returns_a_car_move_directive(self):
        directive = ScenePlanner.decide("A car travels 150 meters in 10 seconds. What is its average speed?")
        assert directive.icon == "car"
        assert directive.animation == "move"

    def test_truck_and_bus_also_match_car(self):
        assert ScenePlanner.decide("A truck drives 60 miles in 2 hours.").icon == "car"
        assert ScenePlanner.decide("A bus travels 40 miles in 1 hour.").icon == "car"

    def test_person_problem_returns_a_person_move_directive(self):
        directive = ScenePlanner.decide("A runner runs 5 kilometers in 30 minutes. What is her speed?")
        assert directive.icon == "person"
        assert directive.animation == "move"

    def test_ball_problem_returns_a_ball_move_directive(self):
        directive = ScenePlanner.decide("A ball is thrown 12 meters. How far did it travel?")
        assert directive.icon == "ball"
        assert directive.animation == "move"

    def test_plant_problem_returns_a_plant_grow_directive(self):
        directive = ScenePlanner.decide("A plant grows 3 centimeters every week.")
        assert directive.icon == "plant"
        assert directive.animation == "grow"

    def test_tree_also_matches_plant(self):
        directive = ScenePlanner.decide("A tree grows 2 feet per year.")
        assert directive.icon == "plant"
        assert directive.animation == "grow"

    def test_abstract_math_problem_returns_no_directive(self):
        assert ScenePlanner.decide("What is the area of a rectangle with width 6 and height 4?") is None
        assert ScenePlanner.decide("Solve for x: 2x + 5 = 17") is None

    def test_pure_algebra_problem_returns_no_directive(self):
        assert ScenePlanner.decide("Solve for x in 3x - 7 = 20.") is None

    def test_a_genetics_problem_about_a_pea_plant_matches_plant(self):
        # The pea plant IS a real described object, same as a car or ball -
        # this is a feature, not a false positive.
        directive = ScenePlanner.decide(
            "If a pea plant with genotype Bb is crossed with another Bb plant, "
            "what are the possible offspring genotypes?"
        )
        assert directive.icon == "plant"

    def test_is_case_insensitive(self):
        directive = ScenePlanner.decide("A CAR TRAVELS 100 METERS.")
        assert directive.icon == "car"

    def test_matches_whole_words_only(self):
        # "scar" contains "car" as a substring but isn't the word "car" -
        # a naive substring match would misfire here.
        assert ScenePlanner.decide("She has a scar on her arm.") is None
