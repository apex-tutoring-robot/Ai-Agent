"""
Persistent user identity and name for this device.

Stores a UUID4 and the student's name in config/user_profile.json.
"""

import json
import os
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROFILE_PATH = "config/user_profile.json"


def _load_profile() -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(_PROFILE_PATH)), exist_ok=True)
    if os.path.exists(_PROFILE_PATH):
        try:
            with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("user_id", str(uuid.uuid4()))
            data.setdefault("user_name", None)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("user_profile: could not read %s: %s — regenerating", _PROFILE_PATH, e)
    return {"user_id": str(uuid.uuid4()), "user_name": None}


def _save_profile(data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(_PROFILE_PATH)), exist_ok=True)
    tmp = _PROFILE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _PROFILE_PATH)
    except OSError as e:
        logger.error("user_profile: could not write %s: %s", _PROFILE_PATH, e)
        raise


def get_user_id() -> str:
    """Return the stable user ID for this device, generating one on first call."""
    return _load_profile()["user_id"]


def get_user_name() -> Optional[str]:
    """Return the stored user name, or None if not yet set."""
    return _load_profile().get("user_name")


def set_user_name(name: str) -> None:
    """Persist the user's name (title-cased)."""
    data = _load_profile()
    data["user_name"] = name.strip().title()
    _save_profile(data)
    logger.info("user_profile: saved name '%s' (id=%s)", data["user_name"], data["user_id"])
