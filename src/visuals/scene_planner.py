"""
Decides whether a teaching turn's problem describes a real-world object
(a car, a person, a ball, a plant) that deserves an actual sketched icon
on the whiteboard - not just an abstract box/circle - and, when so, what
simple animation best shows the scenario being described: a car driving
across the board for a distance/speed problem, a plant growing for a
life-cycle question, and so on.

Deliberately a separate, local, deterministic engine rather than another
LLM call or an extension of the whiteboard-content prompt: choosing "does
this problem mention a car" is simple enough that a keyword lookup is
both faster and more reliable than asking an LLM to remember to do it
consistently on top of everything else generate_teaching_plan already
juggles (title, equations, diagram, underline...). Same reasoning as
FastContentFilter's synchronous pre-check - fast, free, and still works
if Azure is unreachable.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SceneDirective:
    icon: str        # matches a TeachingCanvas _draw_<icon>_icon method
    animation: str    # "move" (drive/walk/kick across the board) or "grow"


# Order matters - first match wins, so more specific words (checked via
# word boundaries) come first where overlap is possible.
_ICON_RULES: List[Tuple[str, str, str]] = [
    ("car", r"\b(car|truck|bus|taxi|vehicle)\b", "move"),
    ("person", r"\b(runner|walker|hiker|swimmer|student|walks|walking|runs|running)\b", "move"),
    ("ball", r"\b(ball|baseball|basketball|soccer|kicked|kicks|thrown|throws|throwing)\b", "move"),
    ("plant", r"\b(plant|seed|sprout|sapling|flower|tree)\b", "grow"),
]


class ScenePlanner:
    """Stateless - decide() is the only entry point."""

    @staticmethod
    def decide(problem_text: str) -> Optional[SceneDirective]:
        text = problem_text.lower()
        for icon, pattern, animation in _ICON_RULES:
            if re.search(pattern, text):
                return SceneDirective(icon=icon, animation=animation)
        return None
