"""
NeMo Guardrails manager for Jarvis K-8 tutoring robot.

Provides input and output safety rails:
  - Input: school-subject allowlist + safety (no harmful/jailbreak content)
  - Output: safety check before TTS (blocks any harmful LLM output before it is spoken)

Both checks fail-open: if NeMo encounters an error, the bot continues normally
rather than silently breaking the conversation.
"""

import os
import re
import asyncio
import logging
from typing import Optional, Tuple

_ENV_PATTERN = re.compile(r'\$\{env:([^}]+)\}')

logger = logging.getLogger(__name__)

# Kid-friendly refusal messages keyed by block reason
REFUSAL_MESSAGES = {
    "off_topic": (
        "That is not something I can help with. I am here to help with school subjects — "
        "English, math, science, history, civics, computers, arts, sports, and more. "
        "What would you like to learn today?"
    ),
    "harmful_input": (
        "I cannot help with that. Let us focus on learning! "
        "What school subject would you like to explore today?"
    ),
    "jailbreak": (
        "I am Jarvis, your school tutor! "
        "What would you like to learn today?"
    ),
    "harmful_output": (
        "I am not sure how to answer that. "
        "Let us get back to learning! What school subject can I help you with?"
    ),
}

# Substrings of our defined bot refusal messages — used to detect NeMo blocks
_REFUSAL_SIGNATURES = [msg[:40].lower() for msg in REFUSAL_MESSAGES.values()]

# NeMo's own default block sentinel (returned when no custom flow matches)
_NEMO_DEFAULT_BLOCK = "i'm sorry, i can't respond to that"

# Marker injected by the vision pipeline when image content is appended to the user message
_IMAGE_CONTENT_MARKER = "[Scanned homework content:"

# Allowlist-style prompt for evaluating scanned image content.
# Intentionally more restrictive than the general input prompt: only clearly
# math/science homework/diagrams pass — portraits, objects, scenes do not.
_IMAGE_EVAL_PROMPT = """\
You are a content filter for Jarvis, an AI tutor for K-8 students covering all school subjects.

A student has shared an image. Below is what the image shows.

Image content: "{image_content}"

The image is APPROPRIATE (answer No) if it clearly shows educational content from any school subject:
- Mathematics: equations, numbers, word problems, graphs, geometry, fractions, worksheets
- Science: biology diagrams, chemistry equations, physics problems, experiments, nature diagrams
- English/language arts: text passages, grammar exercises, writing prompts, literature excerpts
- History/civics/geography: maps, historical documents, timelines, diagrams
- Economics: charts, supply/demand diagrams, financial concepts
- Computer science: code, flowcharts, circuit diagrams
- Arts/music: labeled artwork, music notation, art history images
- Sports/PE: technique diagrams, rules illustrations, fitness charts
- Any other school homework worksheet or educational diagram

The image is NOT APPROPRIATE (answer Yes) if it shows:
- People, portraits, selfies, or faces unrelated to education
- Random food, drinks, or cooking
- Celebrity or entertainment content
- Anything clearly not educational or inappropriate for children

Should this image be blocked?
Answer [Yes/No]:"""

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

            # NeMo Guardrails does not expand ${env:VAR} when variables are
            # injected via load_dotenv() rather than the OS environment at
            # process start. Resolve them here before LLMRails consumes them.
            for model in config.models:
                if model.parameters:
                    for key, value in list(model.parameters.items()):
                        if isinstance(value, str) and _ENV_PATTERN.search(value):
                            model.parameters[key] = _ENV_PATTERN.sub(
                                lambda m: os.getenv(m.group(1), ''), value
                            )

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
        Gate user_text BEFORE calling the main LLM.

        Camera-trigger path: if the message contains scanned image content
        (appended by the vision pipeline), evaluate only the IMAGE content for
        topic relevance. The camera command itself ("can you take a picture?")
        is always allowed — we only gate what the image actually shows.

        Fast path: keyword scan for obvious jailbreak/harmful content (no LLM).
        LLM path: single self_check_input call to enforce the math/science-only
        allowlist.

        Returns:
            (allowed, refusal_message)
        """
        if not self._enabled:
            return True, None

        # Vision pipeline appends "[Scanned homework content: ...]" to the user
        # message. Evaluate only the image content — the camera command itself
        # is always a valid meta-request.
        marker_idx = user_text.find(_IMAGE_CONTENT_MARKER)
        if marker_idx != -1:
            image_content = (
                user_text[marker_idx + len(_IMAGE_CONTENT_MARKER):]
                .rstrip(']')
                .strip()
            )
            if not image_content:
                return True, None
            try:
                allowed = asyncio.run(self._check_image_content_async(image_content))
                if not allowed:
                    logger.info("Guardrails BLOCKED image content (not math/science)")
                    return False, REFUSAL_MESSAGES["off_topic"]
                return True, None
            except Exception as exc:
                logger.error(
                    "Guardrails image content check error: %s — allowing through (fail-open).", exc
                )
                return True, None

        lowered = user_text.lower()

        if any(kw in lowered for kw in _JAILBREAK_KEYWORDS):
            logger.info("Guardrails BLOCKED input (jailbreak keyword): '%.60s'", user_text)
            return False, REFUSAL_MESSAGES["jailbreak"]

        if any(kw in lowered for kw in _HARM_KEYWORDS):
            logger.info("Guardrails BLOCKED input (harm keyword): '%.60s'", user_text)
            return False, REFUSAL_MESSAGES["harmful_input"]

        try:
            allowed = asyncio.run(self._self_check_input_async(user_text))
            if not allowed:
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
        Run only self_check_output against the LLM response, BEFORE TTS.

        Returns:
            (safe, final_text)
              safe=True  → pass through original bot_response.
              safe=False → speak kid-friendly refusal instead.
        """
        if not self._enabled:
            return True, bot_response

        try:
            allowed = asyncio.run(self._self_check_output_async(bot_response))
            if not allowed:
                logger.warning("Guardrails BLOCKED output: '%.80s'", bot_response)
                return False, REFUSAL_MESSAGES["harmful_output"]
            return True, bot_response

        except Exception as exc:
            logger.error(
                "Guardrails output check error: %s — passing through (fail-open).", exc
            )
            return True, bot_response

    async def _check_image_content_async(self, image_content: str) -> bool:
        """
        Evaluate scanned image content against a strict math/science allowlist.

        Uses a hardcoded allowlist prompt rather than the general self_check_input
        prompt, because the general prompt is a blocklist (food/sports/etc.) that
        leaves ambiguous content like 'person in an office' uncaught.  This prompt
        only passes clearly educational math/science material.
        """
        from langchain_core.messages import HumanMessage
        filled = _IMAGE_EVAL_PROMPT.format(image_content=image_content)
        result = await self._rails.llm.ainvoke(
            [HumanMessage(content=filled)],
            max_completion_tokens=3,
        )
        answer = (result.content if hasattr(result, 'content') else str(result)).strip().lower()
        return not answer.startswith('yes')

    async def _self_check_input_async(self, user_text: str) -> bool:
        """Call self_check_input directly via LLM, bypassing NeMo's full pipeline."""
        prompt_template = None
        for prompt in self._rails.config.prompts:
            if prompt.task == 'self_check_input':
                prompt_template = prompt.content
                break

        if not prompt_template:
            return True

        filled = re.sub(r'\{\{\s*user_input\s*\}\}', user_text, prompt_template)

        from langchain_core.messages import HumanMessage
        result = await self._rails.llm.ainvoke(
            [HumanMessage(content=filled)],
            max_completion_tokens=3,
        )
        answer = (result.content if hasattr(result, 'content') else str(result)).strip().lower()
        return not answer.startswith('yes')

    async def _self_check_output_async(self, bot_response: str) -> bool:
        """Call self_check_output directly, bypassing NeMo's full pipeline."""
        # Pull the prompt template from the loaded config
        prompt_template = None
        for prompt in self._rails.config.prompts:
            if prompt.task == 'self_check_output':
                prompt_template = prompt.content
                break

        if not prompt_template:
            return True

        filled = re.sub(r'\{\{\s*bot_response\s*\}\}', bot_response, prompt_template)

        from langchain_core.messages import HumanMessage
        result = await self._rails.llm.ainvoke(
            [HumanMessage(content=filled)],
            max_completion_tokens=3,
        )
        answer = (result.content if hasattr(result, 'content') else str(result)).strip().lower()
        return not answer.startswith('yes')

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
