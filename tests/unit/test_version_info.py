"""
Tests for utils.version_info.get_current_commit - what the
CommitIndicator HUD (visuals/ui/commit_indicator.py) shows, so someone
at the physical robot can tell whether it actually picked up a pushed
fix instead of guessing from an ambiguous "it's not working" report.
"""

import subprocess

import utils.version_info as version_info


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


class TestGetCurrentCommit:
    def test_returns_the_stripped_commit_hash(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _FakeCompletedProcess("a5c2570\n")
        )
        assert version_info.get_current_commit() == "a5c2570"

    def test_passes_short_flag_by_default(self, monkeypatch):
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return _FakeCompletedProcess("a5c2570\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        version_info.get_current_commit()
        assert captured["args"] == ["git", "rev-parse", "--short", "HEAD"]

    def test_omits_short_flag_when_asked_for_the_full_hash(self, monkeypatch):
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return _FakeCompletedProcess("a5c25709876543210fedcba9876543210fedcba\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        version_info.get_current_commit(short=False)
        assert captured["args"] == ["git", "rev-parse", "HEAD"]

    def test_falls_back_to_unknown_when_git_raises(self, monkeypatch):
        def fake_run(*a, **k):
            raise subprocess.CalledProcessError(128, ["git"])

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert version_info.get_current_commit() == "unknown"

    def test_falls_back_to_unknown_when_git_is_missing(self, monkeypatch):
        def fake_run(*a, **k):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert version_info.get_current_commit() == "unknown"

    def test_falls_back_to_unknown_on_empty_output(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess("\n"))
        assert version_info.get_current_commit() == "unknown"
