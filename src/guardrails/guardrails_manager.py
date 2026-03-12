"""
NeMo Guardrails manager for Jarvis K-8 tutoring robot.

Provides input and output safety rails:
  - Input: topic restriction (math/science only) + safety (no harmful/jailbreak content)
  - Output: safety check before TTS (blocks any harmful LLM output before it is spoken)

Both checks fail-open: if NeMo encounters an error, the bot continues normally
rather than silently breaking the conversation.
"""

import os
import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Kid-friendly refusal messages keyed by block reason
REFUSAL_MESSAGES = {
    "off_topic": (
        "That is an interesting question, but I can only help with math and science! "
        "Would you like to explore a math or science topic today?"
    ),
    "harmful_input": (
        "I cannot help with that. Let us talk about something fun, "
        "like why the sky is blue or how to add fractions!"
    ),
    "jailbreak": (
        "I am Jarvis, your math and science tutor! "
        "What would you like to learn today?"
    ),
    "harmful_output": (
        "I am not sure how to answer that safely. "
        "Let us get back to learning! Do you have a math or science question?"
    ),
}

# Substrings of our defined bot refusal messages — used to detect NeMo blocks
_REFUSAL_SIGNATURES = [msg[:40].lower() for msg in REFUSAL_MESSAGES.values()]

# NeMo's own default block sentinel (returned when no custom flow matches)
_NEMO_DEFAULT_BLOCK = "i'm sorry, i can't respond to that"

# Keywords for fast keyword-based refusal classification (no extra LLM call)
_JAILBREAK_KEYWORDS = {
    "ignore", "pretend", "forget", "instructions", "unrestricted",
    "developer mode", "dan", "jailbreak", "no rules", "act as",
    "override", "bypass", "disable", "evil ai",
}
_HARM_KEYWORDS = {
    "hurt", "kill", "weapon", "bomb", "hate", "fight", "punch",
    "bad word", "curse", "swear", "naked", "sex", "inappropriate",
}


class GuardrailsManager:
    """
    Wraps NeMo Guardrails LLMRails to provide input and output safety rails.

    Usage:
        guardrails = GuardrailsManager()
        allowed, refusal = guardrails.check_input(user_text)
        if not allowed:
            speak(refusal)
            return
        llm_response = generate_response(...)
        safe, final_text = guardrails.check_output(llm_response)
        speak(final_text)

    Both methods are synchronous (blocking) and safe to call from a non-async
    thread. They use asyncio.run() internally, which creates a fresh event loop
    for each call and tears it down after — correct behaviour for Python threads
    that have no running event loop.
    """

    def __init__(self, config_path: str = "config/guardrails"):
        """
        Initialise NeMo Guardrails from a config directory.

        Args:
            config_path: Path to the directory containing config.yml and *.co files.
        """
        self._config_path = config_path
        self._rails = None
        self._enabled = False

        # Allow disabling guardrails via env var for development / testing
        if os.getenv("GUARDRAILS_ENABLED", "true").lower() == "false":
            logger.warning("GuardrailsManager DISABLED by GUARDRAILS_ENABLED=false")
            return

        self._initialize()

    def _initialize(self) -> None:
        """Load NeMo Guardrails config and build the LLMRails instance."""
        try:
            from nemoguardrails import LLMRails, RailsConfig

            config = RailsConfig.from_path(self._config_path)
            self._rails = LLMRails(config)
            self._enabled = True
            logger.info("GuardrailsManager initialized from %s", self._config_path)
        except ImportError:
            logger.error(
                "nemoguardrails package not installed. "
                "Run: pip install nemoguardrails>=0.9.0 — guardrails DISABLED."
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize GuardrailsManager: %s. "
                "Guardrails DISABLED — running without safety checks.",
                exc,
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def check_input(self, user_text: str) -> Tuple[bool, Optional[str]]:
        """
        Run NeMo input rails against user_text BEFORE calling the main LLM.

        This is the gate that prevents off-topic or harmful queries from ever
        reaching Azure OpenAI, saving both API cost and response latency.

        Args:
            user_text: The transcribed (and PII-anonymised) user utterance.

        Returns:
            (allowed, refusal_message)
              allowed=True  → input passed all rails; proceed with LLM call.
              allowed=False → input was blocked; speak refusal_message instead.
        """
        if not self._enabled:
            return True, None

        try:
            response = asyncio.run(
                self._rails.generate_async(
                    messages=[{"role": "user", "content": user_text}]
                )
            )

            bot_text = response.strip() if isinstance(response, str) else ""

            if self._is_blocked_response(bot_text):
                refusal = self._classify_refusal(user_text)
                logger.info("Guardrails BLOCKED input: '%.60s'", user_text)
                return False, refusal

            return True, None

        except Exception as exc:
            logger.error(
                "Guardrails input check error: %s — allowing through (fail-open).", exc
            )
            return True, None

    def check_output(self, bot_response: str) -> Tuple[bool, str]:
        """
        Run NeMo output rails against the FULL buffered LLM response,
        BEFORE it is sent to TTS.

        Args:
            bot_response: The complete LLM-generated text for this turn.

        Returns:
            (safe, final_text)
              safe=True  → final_text is the original bot_response (pass through).
              safe=False → final_text is a kid-friendly refusal; speak this instead.
        """
        if not self._enabled:
            return True, bot_response

        try:
            # Feed as a two-message context so NeMo runs output rails on the response
            response = asyncio.run(
                self._rails.generate_async(
                    messages=[
                        {"role": "user", "content": "[output check]"},
                        {"role": "assistant", "content": bot_response},
                    ]
                )
            )

            result_text = response.strip() if isinstance(response, str) else bot_response

            if self._is_blocked_response(result_text):
                logger.warning(
                    "Guardrails BLOCKED output: '%.80s'", bot_response
                )
                return False, REFUSAL_MESSAGES["harmful_output"]

            # Always return the ORIGINAL response — do not use NeMo's re-generation
            return True, bot_response

        except Exception as exc:
            logger.error(
                "Guardrails output check error: %s — passing through (fail-open).", exc
            )
            return True, bot_response

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _is_blocked_response(self, text: str) -> bool:
        """
        Detect whether NeMo returned a block/refusal response.

        NeMo 0.9 outputs the bot's defined refusal message verbatim when a rail
        fires, or the generic sentinel when no custom flow matched.
        """
        if not text:
            return False

        lowered = text.lower()

        if _NEMO_DEFAULT_BLOCK in lowered:
            return True

        for sig in _REFUSAL_SIGNATURES:
            if sig in lowered:
                return True

        return False

    def _classify_refusal(self, user_text: str) -> str:
        """
        Return the most contextually appropriate kid-friendly refusal message
        based on simple keyword analysis — avoids an extra LLM classification call.
        """
        lowered = user_text.lower()

        if any(kw in lowered for kw in _JAILBREAK_KEYWORDS):
            return REFUSAL_MESSAGES["jailbreak"]

        if any(kw in lowered for kw in _HARM_KEYWORDS):
            return REFUSAL_MESSAGES["harmful_input"]

        return REFUSAL_MESSAGES["off_topic"]
