"""
The curated library of real-world-example photos shown on the whiteboard
during a curiosity-question teaching turn (see JarvisBot._run_teaching_turn
and LLMClient.generate_teaching_plan's "image_tag" field).

Deliberately a small, hand-picked local set rather than AI-generated
images (real cost/latency per turn, would delay "speech starts
immediately") or a live web image search (real latency, and licensing/
content-safety risk for an image nobody here actually chose) - see the
decision that shaped this. Every photo here is CC0 or Public Domain Mark
licensed (sourced via Openverse, which verifies this), downscaled to a
sensible on-screen size, and reviewed for being a clean, single-subject,
kid-appropriate photo (not a branded product, not showing identifiable
people, not visually confusing).

generate_teaching_plan's prompt is given this manifest's tags (and ONLY
these tags) and picks one per speech step, or none - this file is the
single source of truth for what's available, so the prompt and the
renderer never drift out of sync with each other.

A deliberately small starter set - not every topic has a matching photo,
and that's fine: a step with no good match just gets the existing
sketch-diagram treatment instead, same as before this existed.
"""

import os

# Co-located with this module (NOT under /data/, which is gitignored for
# real student data - see .gitignore - and would silently ship with no
# images at all on a fresh checkout) rather than a repo-root assets
# folder, since this project has no existing convention for one.
_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_images")

# tag -> {filename, label (shown nowhere yet, just for humans reading this
# file), attribution (kept for the record - CC0/PDM don't require display,
# but tracking provenance is good practice for a product used by kids)}
EXAMPLE_IMAGES = {
    "pizza": {
        "filename": "pizza.jpg",
        "label": "Two slices of pepperoni pizza on a plate",
        "attribution": "Photo by pixellaphoto, marked CC0 1.0 (via Openverse/Flickr).",
    },
    "cookies": {
        "filename": "cookies.jpg",
        "label": "Chocolate chip cookies on small plates",
        "attribution": "\"Individual servings of chocolate chip cookies\", CC0 (via Openverse/WordPress Photo Directory).",
    },
    "apple": {
        "filename": "apple.jpg",
        "label": "A single red apple on a white background",
        "attribution": "CC0 licensed photo (via Openverse/Rawpixel).",
    },
    "leaf": {
        "filename": "leaf.jpg",
        "label": "A single green leaf on a white background",
        "attribution": "CC0 licensed photo (via Openverse/Rawpixel).",
    },
}


def available_tags() -> list:
    """The tag list handed to generate_teaching_plan's prompt - see module docstring."""
    return sorted(EXAMPLE_IMAGES.keys())


def image_path(tag: str):
    """Absolute path to the tag's image file, or None if the tag isn't in
    the library (an unrecognized tag from a malformed LLM response, same
    defensive pattern as every other LLM-supplied field in this file's
    draw-action pipeline - never trust it blindly)."""
    entry = EXAMPLE_IMAGES.get(tag)
    if entry is None:
        return None
    path = os.path.join(_IMAGE_DIR, entry["filename"])
    return path if os.path.isfile(path) else None
