"""
Tests for visuals.example_image_library - the curated local photo
library shown during a curiosity-question teaching turn (see
JarvisBot._run_teaching_turn and LLMClient.generate_teaching_plan's
"image_tag" field). Confirms the manifest and the actual files on disk
agree with each other - a tag with no matching file, or a file that
isn't where the manifest says, would silently break the feature.
"""

import os

from visuals.example_image_library import EXAMPLE_IMAGES, available_tags, image_path


class TestAvailableTags:
    def test_returns_every_manifest_tag(self):
        assert set(available_tags()) == set(EXAMPLE_IMAGES.keys())

    def test_is_sorted(self):
        tags = available_tags()
        assert tags == sorted(tags)


class TestImagePath:
    def test_every_manifest_tag_resolves_to_a_real_file(self):
        # Regression guard: the manifest and the actual downloaded/
        # checked-in images must never drift apart.
        for tag in available_tags():
            path = image_path(tag)
            assert path is not None, f"{tag!r} has no resolvable file"
            assert os.path.isfile(path), f"{tag!r} points to a missing file: {path}"

    def test_unknown_tag_returns_none(self):
        assert image_path("spaceship") is None

    def test_empty_string_returns_none(self):
        assert image_path("") is None
