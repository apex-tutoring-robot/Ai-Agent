"""
Persistent user identity for this device.

Generates a UUID4 on first run and saves it to config/user_profile.json.
Any module that needs to tag user interactions imports get_user_id() from here.
"""

import json
import os
import uuid
import logging

logger = logging.getLogger(__name__)

_PROFILE_PATH = "config/user_profile.json"


def get_user_id() -> str:
    """
    Return the stable user ID for this device, generating one on first call.
    """
    os.makedirs(os.path.dirname(os.path.abspath(_PROFILE_PATH)), exist_ok=True)

    if os.path.exists(_PROFILE_PATH):
        try:
            with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_id = data.get("user_id")
            if user_id:
                return user_id
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("user_profile: could not read %s: %s — regenerating", _PROFILE_PATH, e)

    user_id = str(uuid.uuid4())
    try:
        with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"user_id": user_id}, f, indent=2)
        logger.info("user_profile: generated new user_id")
    except OSError as e:
        logger.error("user_profile: could not write %s: %s", _PROFILE_PATH, e)

    return user_id
