"""
Azure OpenAI client with streaming support.
Generates tutoring responses using Azure OpenAI with streaming output.
"""

import os
import logging
from typing import Iterator, List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import re

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
        max_tokens: int = 500
    ) -> Iterator[str]:
        """
        Generate streaming response from Azure OpenAI.
        
        Args:
            messages: Conversation history (list of message dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
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
                max_tokens=max_tokens,
                stream=True,
                user=self.user_id,
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
    
    def extract_image_content(self, image_url: str, max_tokens: int = 1000) -> str:
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
            max_tokens=max_tokens,
            stream=False,
            user=self.user_id,
        )
        extracted = response.choices[0].message.content.strip()
        logger.info(f"📷 Image extracted ({len(extracted)} chars)")
        return extracted

    def generate_teaching_plan(
        self,
        messages,
        temperature: float = 0.2,
        max_tokens: int = 800
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
                max_tokens=max_tokens,
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
        
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 15
    ) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Complete response text
        """
        try:
            # Collect all chunks
            chunks = list(self.generate_response_stream(messages, temperature, max_tokens))
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
