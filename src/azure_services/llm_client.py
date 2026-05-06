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
        system_prompt_path: Optional[str] = None
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
        self.api_key = api_key or os.getenv('AZURE_OPENAI_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        self.system_prompt_path = system_prompt_path or os.getenv('SYSTEM_PROMPT_PATH', './config/system_prompt.txt')
        
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
        messages: List[Dict[str, str]],
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
                stream=True
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
        {"speech_id": 1, "action": "draw_text", "text": "string", "x": 100, "y": 120},
        {"speech_id": 1, "action": "draw_line", "x1": 0, "y1": 0, "x2": 100, "y2": 100},
        {"speech_id": 1, "action": "draw_rect", "x": 580, "y": 160, "w": 220, "h": 160},
        {"speech_id": 1, "action": "draw_circle", "x": 620, "y": 270, "r": 100},
        {"speech_id": 1, "action": "draw_circle", "x": 620, "y": 270, "rx": 110, "ry": 75},
        {"speech_id": 1, "action": "draw_polygon", "points": [[580,160],[780,160],[780,360],[580,360]]},
        {"speech_id": 1, "action": "draw_regular_polygon", "sides": 6, "cx": 620, "cy": 270, "radius": 110},
        {"speech_id": 1, "action": "draw_arc", "x": 510, "y": 160, "w": 200, "h": 200, "start_angle": 0, "span_angle": 360}
      ]
    }

    Action field reference:
    - draw_circle: x,y = CENTER of circle. r = radius (for circles). rx,ry = separate radii (for ellipses).
    - draw_rect: x,y = top-left corner. w,h = width and height.
    - draw_regular_polygon: sides=number of sides, cx/cy=center, radius=circumscribed radius.
    - draw_polygon: points = list of [x,y] pairs (minimum 3 points).
    - draw_arc: x,y = top-left of bounding box, w/h = bounding box size, start_angle/span_angle in degrees.

    Rules:
    - Allowed actions: clear, draw_text, draw_line, draw_rect, draw_circle, draw_polygon, draw_regular_polygon, draw_arc
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
    GEOMETRY DIAGRAM RULES — MANDATORY, NO EXCEPTIONS:
    - You MUST draw a shape diagram for EVERY geometry problem. Never omit it.
    - Equations go LEFT side (x: 60–420). Diagrams go MIDDLE zone (x: 480–750).
    - x > 780 is reserved for the face widget — NEVER place any shape or label there.
    - The face widget occupies x: 800–1280. Keep all drawing strictly left of x=780.
    - y range for both: 130 to 420.

    LABEL PLACEMENT — CRITICAL RULE:
    - Labels must be OUTSIDE the shape, never on or crossing an edge.
    - Place labels for the BOTTOM of a shape at y = shape_bottom + 30 (e.g. y=410 if shape ends at y=380).
    - Place labels for the TOP of a shape at y = shape_top - 15 (e.g. y=145 if shape starts at y=160).
    - Place labels for the LEFT side at x = shape_left - 70 (e.g. x=510 if shape starts at x=580).
    - Place labels for the RIGHT side at x = shape_right + 15.
    - Stack multiple bottom labels 30px apart: y=410, y=440, etc.
    - Stack multiple top labels 25px apart: y=145, y=120, etc.

    SHAPE-BY-SHAPE RULES (with exact coordinate examples):

    TRIANGLE (right, scalene, isosceles, equilateral — any):
      Use 3x draw_line for edges. Example right triangle:
        draw_line x1=500 y1=380 x2=700 y2=380  (base, horizontal)
        draw_line x1=700 y1=380 x2=500 y2=160  (hypotenuse)
        draw_line x1=500 y1=160 x2=500 y2=380  (height, vertical)
      Labels OUTSIDE:
        Base label:   draw_text "Base=5"   x=575 y=410  (30px BELOW the base line)
        Height label: draw_text "Height=10" x=420 y=270  (70px LEFT of the vertical edge)
        Hyp label:    draw_text "Hyp=11"   x=615 y=265  (beside the slant, not on it)

    RECTANGLE / SQUARE:
      draw_rect x=500 y=160 w=220 h=160
      Labels:
        Width:  draw_text "w=10" x=580 y=145   (15px ABOVE top edge, y=160-15=145)
        Height: draw_text "h=8"  x=735 y=245   (15px RIGHT of right edge, x=500+220+15=735)

    CIRCLE:
      draw_circle x=620 y=270 r=100   (x,y = CENTER)
      Labels:
        draw_line x1=620 y1=270 x2=720 y2=270  (radius line from center to edge)
        draw_text "r=7" x=655 y=260            (above the radius line)

    REGULAR POLYGON (pentagon, hexagon, octagon, etc.):
      ALWAYS use draw_regular_polygon — NEVER draw_circle for these shapes.
      draw_regular_polygon sides=5 cx=620 cy=270 radius=110
      Label side length BELOW the shape:
        draw_text "s=10" x=565 y=405   (cy + radius + 25 = 270+110+25 = 405)
      CRITICAL: cx=620 keeps the whole polygon left of x=750. NEVER use cx > 650.

    TRAPEZOID:
      draw_polygon points=[[520,170],[680,170],[720,360],[480,360]]
      Labels:
        Top side:    draw_text "a=6"  x=570 y=150  (20px above top edge y=170)
        Bottom side: draw_text "b=10" x=550 y=390  (30px below bottom edge y=360)
        Height:      draw_text "h=8"  x=420 y=265  (left of shape)

    PARALLELOGRAM / RHOMBUS:
      draw_polygon points=[[540,170],[740,170],[680,360],[480,360]]
      Labels:
        Base:   draw_text "b=10" x=575 y=390
        Height: draw_text "h=8"  x=415 y=265

    ELLIPSE:
      draw_circle x=620 y=270 rx=110 ry=75   (use rx and ry, not r)
      Labels: draw_text "a=11" x=635 y=260 and draw_text "b=7.5" x=620 y=185

    SECTOR (pie slice):
      draw_line x1=620 y1=270 x2=740 y2=270   (radius 1)
      draw_line x1=620 y1=270 x2=684 y2=166   (radius 2, at the sector angle)
      draw_arc  x=500 y=150 w=240 h=240 start_angle=0 span_angle=60
      Labels:
        draw_text "r=6"     x=665 y=260
        draw_text "60°"     x=640 y=245

    Always include BOTH:
      1. The visual diagram with labeled dimensions
      2. The full calculation steps as draw_text on the left (x: 60–420)
        
    Canvas layout:
    - equations on the left: x between 60 and 420
    - diagrams in the middle zone: x between 480 and 750
    - HARD LIMIT: nothing at x > 780 — face widget lives there
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
