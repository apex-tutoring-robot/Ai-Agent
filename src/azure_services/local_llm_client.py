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
from typing import Iterator, List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

_TEACHING_PLAN_INSTRUCTIONS = """

You are an AI math tutor.

Return ONLY valid JSON.
Do NOT include markdown.
Do NOT include explanation outside JSON.

Schema:
{
  "speech": [
    {"id": 1, "text": "string"}
  ],
  "visuals": [
    {"speech_id": 1, "action": "clear"},
    {"speech_id": 1, "action": "draw_text", "text": "string", "x": 100, "y": 120}
  ]
}

Rules:
- Allowed actions: clear, draw_text, draw_line, draw_rect, draw_circle, draw_polygon, draw_regular_polygon
- Use 2–5 speech steps
- Keep explanations short and teacher-like
- Every visual must map to a valid speech_id
- Use integers for coordinates
- Always include at least one visual action for each speech step after the first
- Always fully solve the problem when enough information is given
- Do not stop at just writing the formula
- Substitute the given values
- Show the final numeric answer when possible
- For geometry problems, include:
    1. formula
    2. substituted values
    3. simplified result
    4. final answer
If the problem involves geometry, shapes, area, perimeter, radius, diameter, rectangle, square, triangle, or circle:
- Always include a diagram on the canvas
- place the diagram beside the equations, not on top of them
- Always label dimensions or key values on the diagram using draw_text
- Never place labels on top of the shape boundary
- For rectangles and squares, put width labels above the shape and height label to the left or right
- For circles, always draw the circle and place the radius label outside the circle
- For triangles, use draw_line for all three edges and place side labels near, but not on, the edges
- For regular polygons:
    - MUST use draw_regular_polygon
    - Do not use draw_circle or partial arcs
    - Do not skip the shape
    - The first diagram action after clear must be draw_regular_polygon
    - provide sides, cx, cy, radius
    - use cx=700, cy=260, radius=120 unless there is a reason to change it
    - label the side length below or beside the polygon using draw_text
- use draw_line for triangle edges and markings
- use draw_rect for rectangles and squares
- use draw_circle for circles
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

    def generate_teaching_plan(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> dict:
        full_messages = [
            {"role": "system", "content": self.system_prompt + _TEACHING_PLAN_INSTRUCTIONS}
        ] + messages

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
