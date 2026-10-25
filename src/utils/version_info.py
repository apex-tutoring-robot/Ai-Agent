"""
Reads the currently checked-out git commit so the running app can show
what code it's actually running - see visuals/ui/commit_indicator.py.
Exists because of a real deployment confusion: after pushing a fix to
`dev`, there was no way to glance at the physical robot and tell whether
it had actually picked up the new code (pulled + restarted) or was still
running something stale - "it's not working" reports were ambiguous
between "the new code has a bug" and "the device never got the new code".
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

# Resolved relative to this file (src/utils/version_info.py), not the
# process's working directory - a deployed app may be launched by a
# service manager / boot script from somewhere other than the repo
# root, where a plain `git rev-parse` would fail to find the .git dir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_current_commit(short: bool = True) -> str:
    """
    Returns the current commit hash (short by default, e.g. "a5c2570"),
    or "unknown" if this isn't a git checkout or git isn't available -
    never raises, since this is a display nicety, not something that
    should be able to break startup.
    """
    try:
        args = ["git", "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=True, cwd=_REPO_ROOT,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception as e:
        logger.warning(f"Couldn't resolve current git commit: {e}")
        return "unknown"
