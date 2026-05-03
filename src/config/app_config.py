"""
Loads configuration from Azure App Configuration on startup.

Bootstrap credential: AZURE_APP_CONFIG_CONNECTION_STRING (read-only access key,
stored in /etc/ai-agent/credentials on the device). All other secrets live in
Azure and are never written to the device filesystem.

Falls back to environment variables when the connection string is absent,
preserving local development behaviour.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keys fetched from Azure App Configuration (matches .env.template names)
_REMOTE_KEYS = [
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
]


def load() -> dict:
    """
    Fetch credentials from Azure App Configuration and return them as a dict.

    If AZURE_APP_CONFIG_CONNECTION_STRING is not set, returns an empty dict and
    the callers fall back to os.getenv() as before.
    """
    connection_string = os.getenv("AZURE_APP_CONFIG_CONNECTION_STRING")
    if not connection_string:
        logger.info("AZURE_APP_CONFIG_CONNECTION_STRING not set — using local environment variables")
        return {}

    try:
        from azure.appconfiguration import AzureAppConfigurationClient
    except ImportError as exc:
        raise RuntimeError(
            "azure-appconfiguration is not installed. Run: pip install azure-appconfiguration>=1.6.0"
        ) from exc

    label = os.getenv("AZURE_APP_CONFIG_LABEL") or None  # None means no label filter

    client = AzureAppConfigurationClient.from_connection_string(connection_string)
    config: dict = {}

    for key in _REMOTE_KEYS:
        try:
            setting = client.get_configuration_setting(key=key, label=label)
            if setting and setting.value:
                config[key] = setting.value
        except Exception as exc:
            logger.warning("Could not fetch '%s' from App Configuration: %s", key, exc)

    fetched = [k for k in _REMOTE_KEYS if k in config]
    missing = [k for k in _REMOTE_KEYS if k not in config]

    logger.info("App Configuration: fetched %s", fetched)
    if missing:
        logger.warning("App Configuration: missing %s — will fall back to env vars", missing)

    return config


def get(config: dict, key: str, default: Optional[str] = None) -> Optional[str]:
    """Return value from the fetched config dict, falling back to os.getenv."""
    return config.get(key) or os.getenv(key, default)
