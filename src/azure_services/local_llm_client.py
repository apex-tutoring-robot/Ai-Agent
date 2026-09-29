"""
Local LLM client using llama.cpp's OpenAI-compatible API.
Used as a fallback when Azure OpenAI is unavailable.
Auto-starts the llama-server process if it isn't already running.
"""

import atexit
import os
import logging
import json
import re
import subprocess
import time
import urllib.request
import urllib.error
from typing import Callable, Iterator, List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

_TEACHING_PLAN_INSTRUCTIONS = """
You are an AI tutor for any K-8 subject - math, physics, biology, geography,
history, anything a student asks about, not only math.

Return ONLY valid JSON.
Do NOT include markdown.
Do NOT include explanation outside JSON.

Schema:
{
  "concept": "short_snake_case_identifier",
  "speech": [
    {"id": 1, "text": "string"}
  ],
  "check_question": "string or null",
  "visuals": [
    {"speech_id": 1, "action": "clear"},
    {"speech_id": 1, "action": "set_title", "text": "Area of a Rectangle"},
    {"speech_id": 1, "action": "draw_text", "text": "string", "x": 100, "y": 120}
  ]
}

Rules:
- "concept": a short snake_case identifier for the specific skill being taught
  (e.g. "equivalent_fractions", "area_rectangle") - always include it.
- "check_question": ONE short comprehension-check question to ask after
  explaining, or null for a simple already-answered question. Not part of speech.
  MUST use DIFFERENT numbers/values than the worked example - test whether the
  student can apply the idea to a new case, never repeat the same numbers.
- Allowed actions: clear, set_title, draw_text, draw_line, draw_rect, draw_circle, draw_polygon, draw_regular_polygon, draw_arc, squiggly_underline, vertical_arithmetic, long_multiplication
- set_title shows a short Title Case title at the top of the board naming the topic
  (e.g. "Area of a Rectangle") - always include exactly one, speech_id 1, right after
  the initial "clear".
- squiggly_underline draws a red squiggly line under something already on the board to
  call it out (x,y = left edge/baseline of what's being underlined, width = span) -
  always use it to underline the final answer once calculated, like a teacher
  underlining it with a marker. Skip it when the answer was shown via
  vertical_arithmetic or long_multiplication instead (see below) - those already
  make the answer clear.
- vertical_arithmetic draws the standard stacked column algorithm (numbers lined up,
  a rule line, the answer below) for addition, subtraction, or single-digit-multiplier
  multiplication - operation: "add"|"subtract"|"multiply", operands: [a, b] (exactly 2
  non-negative integers), x/y: top-left anchor. Every digit, carry, and borrow mark is
  computed and drawn for you - you supply ONLY operation and operands, never the
  result or individual digit coordinates. subtract needs operands[0] >= operands[1];
  multiply ALWAYS needs the single-digit number in operands[1] and the larger
  number in operands[0], regardless of the word problem's phrasing order (e.g.
  "6 boxes of 234 crayons each" is operands: [234, 6], not [6, 234]). Use this
  INSTEAD of writing draw_text lines for the computation whenever the problem is
  standard column arithmetic - don't do both (that shows the answer twice).
- long_multiplication draws multi-digit x multi-digit multiplication (one partial-
  product row per digit of the multiplier, then their sum) - operands: [a, b], BOTH
  with 2+ digits. operands[0] = number being multiplied (top row), operands[1] =
  multiplier (bottom row, generates one row per digit) - put the number with FEWER
  digits in operands[1] when you have a choice. Same rule: you supply ONLY the two
  operands, every partial product/carry/sum is computed and drawn for you.
- CHOOSING vertical_arithmetic("multiply") vs long_multiplication: check BOTH
  numbers' digit counts. If EITHER is a single digit (0-9), use vertical_arithmetic
  - NEVER long_multiplication - even if the other number is huge. Only use
  long_multiplication when BOTH numbers have 2+ digits. E.g. "6 boxes of 234
  crayons" -> 6 is single-digit -> vertical_arithmetic, operands [234, 6]. "23
  boxes of 14 pencils" -> both 2-digit -> long_multiplication, operands [23, 14].
  Getting this wrong makes the drawing silently disappear from the whiteboard.
- Write the actual solving steps as SEPARATE draw_text lines (formula, then substituted
  values, then simplified result), not one vague summary line - a real board shows the
  work, not a caption describing it. (Except for column arithmetic - use
  vertical_arithmetic or long_multiplication for that instead, see above.)
- Use 2–5 speech steps
- Keep explanations short and teacher-like
- Every visual must map to a valid speech_id
- Use integers for coordinates
- Always include at least one visual action for each speech step after the first
- Always fully solve the problem when enough information is given
- Do not stop at just writing the formula
- Substitute the given values
- Show the final numeric answer when possible
- For problems with a numeric answer (math, physics, etc.), include:
    1. formula
    2. substituted values
    3. simplified result
    4. final answer
DIAGRAM IS MANDATORY FOR EVERY QUESTION, NOT JUST GEOMETRY - a text-only board is
never acceptable:
- Always include a diagram on the canvas, regardless of subject
- place the diagram beside the equations, not on top of them
- Always label dimensions or key values on the diagram using draw_text
- Never place labels on top of the shape boundary
- If the problem involves geometry, shapes, area, perimeter, radius, diameter,
  rectangle, square, triangle, or circle, draw the REAL shape:
    - rectangles/squares: draw_rect, width label above, height label to the left/right
    - circles: draw_circle, radius label outside the circle
    - triangles: 3x draw_line for edges, side labels near but not on the edges
    - regular polygons (pentagon, hexagon, etc.): MUST use draw_regular_polygon
      (never draw_circle/partial arcs), sides/cx/cy/radius, cx=700 cy=260 radius=120
      unless there's a reason to change it, side length labeled below/beside it
- For any OTHER subject (physics, biology/genetics, geography, history, etc.), draw
  the simplest diagram representing the idea using the same primitives:
    - physics motion/force: draw_line between two points, labeled with the
      distance/force/speed value
    - biology/genetics: a Punnett square as a 2x2 grid of draw_rect cells, each
      labeled with draw_text, or a labeled draw_circle for a cell/organism part
    - a process/cycle/sequence: 3-4 draw_rect boxes in a row connected by draw_line,
      each with its own draw_text label
    - anything else: at minimum one labeled draw_circle or draw_rect for the central
      object/idea - never leave the diagram area empty
- Always include both:
    1. the visual diagram
    2. the calculation steps

Canvas layout:
- equations on the left: x between 60 and 420
- diagrams in the middle-right: x between 560 and 860
- keep the far-right area x > 900 empty for the face
- use y values between 130 and 420
- space equation rows at least 55 pixels apart
- never place text labels on top of other text
- never state the same fact/value in two places (e.g. don't add a diagram label
  like "Distance = 150 meters" if the equations already show that same number) -
  each given value appears in the equations OR as a short diagram label, never both
"""

_SERVER_STARTUP_TIMEOUT = 60  # seconds to wait for llama-server to become ready


class LocalLLMClient:
    """llama.cpp / Liquid AI fallback client using OpenAI-compatible API."""

    _server_process: Optional[subprocess.Popen] = None  # shared across instances

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv('LOCAL_LLM_BASE_URL', 'http://localhost:8080/v1')
        self.model = model or os.getenv('LOCAL_LLM_MODEL', 'local-model')
        self.system_prompt = system_prompt or "You are Jarvis, a helpful AI tutoring assistant."

        # Derive the health-check URL from base_url (strip /v1 suffix)
        base = self.base_url.rstrip('/')
        server_root = base[:-3] if base.endswith('/v1') else base
        self._health_url = f"{server_root}/health"

        self._ensure_server_running()

        self.client = OpenAI(base_url=self.base_url, api_key='not-needed')
        logger.info(f"Local LLM (llama.cpp) client ready at {self.base_url}")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _is_server_up(self) -> bool:
        try:
            with urllib.request.urlopen(self._health_url, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _ensure_server_running(self) -> None:
        if self._is_server_up():
            logger.info("llama-server already running — skipping auto-start")
            return

        model_path = os.getenv('LOCAL_LLM_MODEL_PATH', '')
        if not model_path:
            raise RuntimeError(
                "LOCAL_LLM_MODEL_PATH is not set. "
                "Point it to your .gguf file so llama-server can be started automatically."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        server_bin = os.getenv('LOCAL_LLM_SERVER_BIN', 'llama-server')
        port = self._parse_port()

        cmd = [server_bin, '-m', model_path, '--port', str(port), '--host', '127.0.0.1']
        logger.info(f"Starting llama-server: {' '.join(cmd)}")

        LocalLLMClient._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self._shutdown_server)

        self._wait_for_server()

    def _wait_for_server(self) -> None:
        logger.info(f"Waiting for llama-server to be ready (timeout={_SERVER_STARTUP_TIMEOUT}s)...")
        deadline = time.time() + _SERVER_STARTUP_TIMEOUT
        while time.time() < deadline:
            if LocalLLMClient._server_process.poll() is not None:
                raise RuntimeError("llama-server exited unexpectedly during startup")
            if self._is_server_up():
                logger.info("llama-server is ready")
                return
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become ready within {_SERVER_STARTUP_TIMEOUT}s")

    def _shutdown_server(self) -> None:
        proc = LocalLLMClient._server_process
        if proc and proc.poll() is None:
            logger.info("Shutting down llama-server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _parse_port(self) -> int:
        try:
            return int(self.base_url.split(':')[2].split('/')[0])
        except (IndexError, ValueError):
            return 8080

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    def generate_response_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_response(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 15,
    ) -> str:
        return ''.join(self.generate_response_stream(messages, temperature, max_tokens))

    def generate_response_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_executor: Callable[[str, dict], str],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Iterator[str]:
        """
        Degraded-mode fallback: this exists so LocalLLMClient satisfies the
        same LLMProvider surface main.py's live turn-handling path actually
        calls (generate_response_with_tools, not the simpler
        generate_response_stream - see interfaces.py). Real tool-calling
        support depends entirely on whichever .gguf model/chat-template is
        loaded, which isn't something to assume works without testing
        against the specific model in use. Rather than risk a broken
        integration on the one path used when Azure is already down, this
        just answers directly without tools/RAG - a real but weaker
        response beats a fallback that itself throws.
        """
        logger.warning(
            "Local LLM fallback does not support tool-calling - "
            "answering without search_curriculum/set_volume/etc."
        )
        yield from self.generate_response_stream(messages, temperature, max_tokens)

    def generate_teaching_plan(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 800,
        scene_hint: Optional[str] = None,
    ) -> dict:
        # Deliberately NOT prepending self.system_prompt here - same
        # conversational-persona-vs-JSON-only conflict found and fixed in
        # llm_client.py's generate_teaching_plan: combining "helpful AI
        # tutoring assistant" chat framing with "return ONLY JSON" below
        # risks the model following the former and ignoring the latter.
        system_content = _TEACHING_PLAN_INSTRUCTIONS
        if scene_hint:
            system_content = f"{system_content}\n\nSCENE NOTE: {scene_hint}"
        full_messages = [{"role": "system", "content": system_content}] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"Local LLM teaching plan raw: {content[:300]}")

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in local LLM response")

        plan = json.loads(match.group(0))
        if "speech" not in plan or "visuals" not in plan:
            raise ValueError("Invalid teaching plan format from local LLM")

        speech_ids = {s["id"] for s in plan["speech"]}
        for v in plan["visuals"]:
            if v.get("speech_id") not in speech_ids:
                raise ValueError(f"Invalid speech_id in local LLM visuals: {v}")

        return plan

    def extract_image_content(self, image_url: str, max_tokens: int = 1000) -> str:
        raise NotImplementedError("Local model does not support vision/image analysis")
