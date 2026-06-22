"""
Azure OpenAI client with streaming support.
Generates tutoring responses using Azure OpenAI with streaming output.
"""

import os
import logging
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional, Union
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import re


@dataclass
class ToolCall:
    """Returned by generate_response_stream_with_tools when the model requests a function call."""
    name: str
    args: dict
    call_id: str

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class LLMClient:
    """Azure OpenAI client with streaming chat completions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        system_prompt_path: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment: Deployment name
            api_version: API version
            system_prompt_path: Path to system prompt file
        """
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        self.system_prompt_path = system_prompt_path or os.getenv('SYSTEM_PROMPT_PATH', './config/system_prompt.txt')
        self.user_id = user_id
        
        if not all([self.api_key, self.endpoint, self.deployment]):
            raise ValueError("Azure OpenAI credentials not fully provided")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        logger.info("Azure OpenAI client initialized")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            if os.path.exists(self.system_prompt_path):
                with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    logger.info(f"Loaded system prompt from {self.system_prompt_path}")
                    return prompt
            else:
                logger.warning(f"System prompt file not found at {self.system_prompt_path}, using default")
                return "You are Jarvis, a helpful AI tutoring assistant."
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return "You are Jarvis, a helpful AI tutoring assistant."
    
    def generate_response_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_completion_tokens: int = 500,
    ) -> Iterator[str]:
        """
        Generate streaming response from Azure OpenAI.
        
        Args:
            messages: Conversation history (list of message dicts)
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens to generate
        
        Yields:
            Text chunks as they arrive
        """
        try:
            # Prepend system prompt
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            logger.info(f"Generating response for {len(messages)} messages")
            
            # Create streaming completion
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                stream=True,
            )
            
            # Stream chunks
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_response_stream_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.7,
        max_completion_tokens: int = 500,
    ) -> Iterator[Union[str, ToolCall]]:
        """
        Streaming completion with tool support.

        Yields text chunks as usual. If the model decides to call a tool instead
        of producing text, yields a single ToolCall object and stops — the caller
        must execute the tool and make a follow-up call for the final response.
        """
        import json as _json

        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=full_messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            stream=True,
        )

        tool_calls_acc: Dict[int, Dict] = {}

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] += tc.id
                    if tc.function and tc.function.name:
                        tool_calls_acc[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["args"] += tc.function.arguments

            if delta.content:
                yield delta.content

            if choice.finish_reason == "tool_calls":
                for tc_data in tool_calls_acc.values():
                    try:
                        args = _json.loads(tc_data["args"]) if tc_data["args"] else {}
                    except _json.JSONDecodeError:
                        args = {}
                    yield ToolCall(
                        name=tc_data["name"],
                        args=args,
                        call_id=tc_data["id"],
                    )

    def extract_image_content(self, image_url: str, max_completion_tokens: int = 1000) -> str:
        """
        One-shot GPT-4V call to extract all image content as plain text.
        Called once on the vision turn; result stored in history so the image
        is never re-sent — eliminates per-turn image re-encoding cost/latency.
        """
        extraction_prompt = (
            "You are an image content extractor. Describe every question, equation, "
            "diagram, number, and piece of text visible in this image with complete "
            "accuracy. Write all math in plain spoken form (e.g. '2x plus 5 equals 15'). "
            "Be exhaustive — do not skip any content."
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            stream=False,
        )
        extracted = response.choices[0].message.content.strip()
        logger.info(f"📷 Image extracted ({len(extracted)} chars)")
        return extracted

    def generate_teaching_plan(
        self,
        messages,
        temperature: float = 0.2,
        max_completion_tokens: int = 800,
    ):
        try:
            full_messages = [
                {
                    "role": "system",
                    "content": self.system_prompt + """

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
    - Always include at least  one visual action for each speech step after the first
    - ALways fully solve the problem when enough information is given
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
                }
            ] + messages

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )

            content = response.choices[0].message.content.strip()
            logger.info(f"Raw teaching plan: {content[:300]}")

            # ✅ STEP 1: extract JSON safely
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")

            json_str = match.group(0)

            # ✅ STEP 2: parse JSON
            plan = json.loads(json_str)

            # ✅ STEP 3: validate structure
            if "speech" not in plan or "visuals" not in plan:
                raise ValueError("Invalid teaching plan format")

            # ✅ STEP 4: ensure IDs exist
            speech_ids = {s["id"] for s in plan["speech"]}

            for v in plan["visuals"]:
                if v.get("speech_id") not in speech_ids:
                    raise ValueError(f"Invalid speech_id in visuals: {v}")

            return plan

        except Exception as e:
            logger.error(f"Error generating teaching plan: {e}")
            raise

    def generate_structured_response(
        self,
        prompt: str,
        schema: dict,
        schema_name: str,
        max_completion_tokens: int = 1000,
    ) -> str:
        """
        One-shot call that enforces a JSON schema via OpenAI structured outputs.

        Uses response_format=json_schema with strict=True so the model is
        constrained to the exact shape — no prompt engineering required for
        structure. Falls back to json_object mode if the deployment does not
        support structured outputs.

        Args:
            prompt:      Full task prompt.
            schema:      JSON schema dict (must be strict-mode compatible:
                         additionalProperties: false on all object nodes).
            schema_name: Short identifier for the schema (letters/digits/dashes only).
            max_completion_tokens:  Maximum tokens to generate.

        Returns:
            Raw response string (valid JSON matching the schema).
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=max_completion_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
                stream=False,
            )
            result = response.choices[0].message.content.strip()
            logger.info("generate_structured_response(%s): received %d chars", schema_name, len(result))
            return result
        except Exception as e:
            if any(kw in str(e).lower() for kw in ("json_schema", "structured", "unsupported", "response_format")):
                logger.warning(
                    "generate_structured_response: structured outputs not supported, "
                    "falling back to json_object mode: %s", e
                )
                return self.generate_json_response(prompt, max_completion_tokens=max_completion_tokens)
            logger.error("generate_structured_response(%s) error: %s", schema_name, e)
            raise

    def generate_json_response(
        self,
        prompt: str,
        max_completion_tokens: int = 1000
    ) -> str:
        """
        One-shot non-streaming call that requests a JSON object response.

        Used for study plan generation and session summarization. Low temperature
        (0.2) for deterministic structured output. Attempts response_format JSON
        mode and falls back to a plain call if the deployment does not support it.

        Args:
            prompt: Full task prompt (system + instruction combined).
            max_completion_tokens: Maximum tokens to generate.

        Returns:
            Raw response string (should be valid JSON).

        Raises:
            Exception: Re-raises any non-format-related API error after logging.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
                stream=False,
            )
            result = response.choices[0].message.content.strip()
            logger.info("generate_json_response: received %d chars", len(result))
            return result
        except Exception as e:
            if "response_format" in str(e).lower() or "unsupported" in str(e).lower():
                logger.warning(
                    "generate_json_response: response_format unsupported, retrying without it: %s", e
                )
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=max_completion_tokens,
                    stream=False,
                )
                return response.choices[0].message.content.strip()
            logger.error("generate_json_response error: %s", e)
            raise

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_completion_tokens: int = 15
    ) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens to generate
        
        Returns:
            Complete response text
        """
        try:
            # Collect all chunks
            chunks = list(self.generate_response_stream(messages, temperature, max_completion_tokens))
            response = ''.join(chunks)
            logger.info(f"Generated response: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


if __name__ == "__main__":
    # Test LLM client
    print("Testing Azure OpenAI client...\n")
    
    try:
        client = LLMClient()
        
        # Test messages
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("Streaming response:")
        print("-" * 60)
        
        for chunk in client.generate_response_stream(messages):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("\nTest complete!")
    
    except Exception as e:
        print(f"Error: {e}")
