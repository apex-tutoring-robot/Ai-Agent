"""
Azure OpenAI client with streaming support.
Generates tutoring responses using Azure OpenAI with streaming output.
"""

import os
import json
import re
import logging
from typing import Callable, Iterator, List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure_services.local_llm_client import LocalLLMClient

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
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
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

        # Local (llama.cpp) fallback for when Azure is unreachable - fails
        # open to None (not a hard dependency) since LOCAL_LLM_MODEL_PATH
        # won't be set on most dev machines, and this client shouldn't stop
        # Jarvis from starting at all just because the fallback isn't
        # configured. See LocalLLMClient/generate_response_with_tools -
        # NOT live-tested (no .gguf model/llama-server available in this
        # dev environment), only verified at the code/structural level.
        try:
            self._local_client = LocalLLMClient(system_prompt=self.system_prompt)
        except Exception as e:
            logger.warning(f"Local LLM fallback unavailable: {e}")
            self._local_client = None

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
            logger.warning(f"Azure LLM stream failed ({type(e).__name__}: {e}), trying local model")
            if self._local_client:
                self._local_client.system_prompt = self.system_prompt
                yield from self._local_client.generate_response_stream(messages, temperature, max_tokens)
            else:
                logger.error(f"Error generating response: {e}")
                raise

    def generate_response_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_executor: Callable[[str, dict], str],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Iterator[str]:
        """
        Real agentic tool-use loop (Decision -> Tool Call -> Observe ->
        Respond), not a keyword-routed heuristic like is_math_query(). The
        model decides for itself whether it needs to call a tool before
        answering.

        Step 1 is necessarily non-streaming - we need to see whether a
        tool_call came back before committing to stream a response. If the
        model calls tool(s), `tool_executor(name, arguments)` runs them and
        results are fed back for a second, streamed call. If no tool call,
        the first call's content is the (already complete) answer.

        This means every turn pays for one non-streaming round-trip before
        anything can be spoken, even turns that don't end up needing a tool
        - a real latency trade-off inherent to tool-calling in general, not
        specific to this implementation. `tool_executor` is a plain
        callable rather than this class knowing about any specific tool, so
        this stays a generic Azure OpenAI wrapper - Jarvis-specific tool
        definitions and execution live in main.py.
        """
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            logger.warning(f"Azure tool-decision call failed ({type(e).__name__}: {e}), trying local model")
            if self._local_client:
                self._local_client.system_prompt = self.system_prompt
                yield from self._local_client.generate_response_with_tools(
                    messages, tools, tool_executor, temperature, max_tokens
                )
                return
            logger.error(f"Error in tool-decision call: {e}")
            raise

        message = response.choices[0].message

        if not message.tool_calls:
            if message.content:
                yield message.content
            return

        full_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = tool_executor(tool_call.function.name, args)
            except Exception as e:
                result = f"Error running tool: {e}"
            logger.info(f"🔧 Tool call: {tool_call.function.name}({args}) -> {str(result)[:150]}")
            full_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

        try:
            final_stream = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in final_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        except Exception as e:
            # Rarer than the tool-decision call failing (Azure was up a
            # moment ago), and the local model wasn't involved in choosing
            # this tool, so this just answers plainly rather than trying
            # to replay tool context into a model that may not support it.
            logger.warning(f"Azure post-tool-call response failed ({type(e).__name__}: {e}), trying local model")
            if self._local_client:
                self._local_client.system_prompt = self.system_prompt
                yield from self._local_client.generate_response_stream(messages, temperature, max_tokens)
                return
            logger.error(f"Error in post-tool-call response: {e}")
            raise

    def generate_teaching_plan(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800
    ) -> Dict:
        """
        Generate a structured teaching plan (speech steps + synchronized
        whiteboard draw actions) for a math/geometry question, instead of a
        plain text response. See Teaching Canvas docs for the action schema.

        Returns a dict: {"speech": [{"id": int, "text": str}, ...],
                          "visuals": [{"speech_id": int, "action": str, ...}, ...]}
        """
        try:
            # Deliberately NOT prepending self.system_prompt here - it's the
            # conversational K-8 tutoring persona ("keep responses to 2-4
            # sentences", "ask comprehension questions", "no formatting"),
            # which directly conflicts with "return ONLY JSON" below. Tested
            # live: combining them makes the model follow the conversational
            # persona and ignore the JSON requirement entirely, returning
            # plain chat text instead of a parseable plan.
            full_messages = [
                {
                    "role": "system",
                    "content": """
    You are an AI math tutor.

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
    - "concept": a short snake_case identifier for the specific skill being taught
      (e.g. "equivalent_fractions", "area_rectangle", "pythagorean_theorem") - used
      to track this student's progress on this concept across sessions. Always include it.
    - "check_question": ONE short, natural comprehension-check question to ask the
      student after explaining, so they demonstrate understanding rather than just
      listening (e.g. "So if a pizza has 8 slices and you eat 4, what fraction is
      that?"). MUST use DIFFERENT specific numbers/values than the worked example
      above - the point is to test whether the student can apply the idea to a new
      case, not recall the exact answer you just gave them (e.g. if you just solved
      a rectangle with width 5 and height 6, check with different numbers like
      width 3 and height 4 - never repeat the same numbers). Use null only for a
      simple, already fully-answered question where a follow-up check would feel
      repetitive. Do NOT put the check question in speech - it is spoken separately,
      after the explanation.
    - Allowed actions: clear, draw_text, draw_line, draw_rect, draw_circle, draw_polygon, draw_regular_polygon, draw_arc
    - Use 2-5 speech steps
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
    GEOMETRY DIAGRAM RULES - MANDATORY, NO EXCEPTIONS:
    - You MUST draw a shape diagram for EVERY geometry problem. Never omit it.
    - Equations go LEFT side (x: 60-420). Diagrams go MIDDLE zone (x: 480-750).
    - x > 780 is reserved for the face widget - NEVER place any shape or label there.
    - The face widget occupies x: 800-1280. Keep all drawing strictly left of x=780.
    - y range for both: 130 to 420.

    LABEL PLACEMENT - CRITICAL RULE:
    - Labels must be OUTSIDE the shape, never on or crossing an edge.
    - Place labels for the BOTTOM of a shape at y = shape_bottom + 30 (e.g. y=410 if shape ends at y=380).
    - Place labels for the TOP of a shape at y = shape_top - 15 (e.g. y=145 if shape starts at y=160).
    - Place labels for the LEFT side at x = shape_left - 70 (e.g. x=510 if shape starts at x=580).
    - Place labels for the RIGHT side at x = shape_right + 15.
    - Stack multiple bottom labels 30px apart: y=410, y=440, etc.
    - Stack multiple top labels 25px apart: y=145, y=120, etc.

    SHAPE-BY-SHAPE RULES (with exact coordinate examples):

    TRIANGLE (right, scalene, isosceles, equilateral - any):
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
      ALWAYS use draw_regular_polygon - NEVER draw_circle for these shapes.
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
      2. The full calculation steps as draw_text on the left (x: 60-420)

    Canvas layout:
    - equations on the left: x between 60 and 420
    - diagrams in the middle zone: x between 480 and 750
    - HARD LIMIT: nothing at x > 780 - face widget lives there
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

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")

            plan = json.loads(match.group(0))

            if "speech" not in plan or "visuals" not in plan:
                raise ValueError("Invalid teaching plan format")

            speech_ids = {s["id"] for s in plan["speech"]}
            for v in plan["visuals"]:
                if v.get("speech_id") not in speech_ids:
                    raise ValueError(f"Invalid speech_id in visuals: {v}")

            return plan

        except Exception as e:
            logger.warning(f"Azure teaching plan failed ({type(e).__name__}: {e}), trying local model")
            if self._local_client:
                self._local_client.system_prompt = self.system_prompt
                return self._local_client.generate_teaching_plan(messages, temperature, max_tokens)
            logger.error(f"Error generating teaching plan: {e}")
            raise

    def evaluate_answer(
        self,
        question: str,
        concept: str,
        student_answer: str,
        attempts_so_far: int,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ) -> Dict:
        """
        Judge a student's spoken answer to a comprehension-check question
        from a teaching turn (see JarvisBot._handle_teaching_answer).
        Never reduces to exact-string matching - a K-8 student's spoken
        answer ("um, I think it's a half?") needs real judgment.

        Returns {correctness, misconception, recommended_action} - see the
        schema in the prompt below. Unlike generate_teaching_plan(), this
        does NOT raise or fall back to the local LLM on failure - it has a
        universally-safe default ("unclear" -> "hint") that keeps the
        tutoring turn moving rather than needing the conversation to stop.

        LLM-as-judge caveat, confirmed live: even at temperature=0 this can
        occasionally misjudge an objectively simple case (e.g. flagged "20"
        as incorrect for a 4x5 rectangle's area on one call out of several
        identical retries, correct on all others) - the same class of
        non-determinism found earlier in guardrails' self_check_input. The
        consequence here is mild (one redundant hint, not a safety issue),
        so this is an accepted limitation, not something fixed here.
        """
        full_messages = [
            {
                "role": "system",
                "content": f"""
You are judging a K-8 student's spoken answer to a math comprehension
question, as part of a live tutoring conversation.

Question asked: "{question}"
Concept being checked: {concept}
This is the student's attempt number {attempts_so_far + 1} at this question.

Return ONLY valid JSON, no markdown, no explanation outside the JSON.

Schema:
{{
  "correct_answer": "work out the actual correct answer to the question yourself, step by step, BEFORE judging the student - this catches cases where a reflexive judgment would be wrong",
  "correctness": "correct" | "partial" | "incorrect",
  "misconception": "short description of the specific misunderstanding, or null if none",
  "recommended_action": "continue" | "hint" | "reteach" | "challenge",
  "response": "the exact words to say to the student next, 1-2 short sentences, warm and age-appropriate"
}}

Fill in "correct_answer" FIRST, by actually solving the question yourself -
only then compare the student's answer against it for "correctness". Do not
judge correctness from a first impression before you've worked it out.

Rules for recommended_action:
- "continue": the answer was correct - move on, maybe offer a harder challenge
- "hint": the answer was incorrect or partial and this is the student's first attempt - give one small hint, don't re-explain everything
- "reteach": the answer was incorrect AND this is the student's second or later attempt at this question - stop asking, explain the concept a different way instead
- "challenge": the answer was correct and shows strong understanding - offer something harder

Rules for "response":
- If recommended_action is "hint": give ONE small, specific hint related to their
  specific mistake - do not just repeat the question or give away the answer.
- If recommended_action is "reteach": briefly explain the concept a different way
  (a new example or analogy), do not just repeat the original explanation.
- If recommended_action is "continue" or "challenge": brief genuine praise. Do not
  ask a new question here - a new one will be asked separately.
- Keep it to 1-2 sentences, spoken out loud to a K-8 student - no formatting.
"""
            },
            {"role": "user", "content": student_answer},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"Answer evaluation raw: {content[:200]}")

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")

            result = json.loads(match.group(0))
            if "correctness" not in result or "recommended_action" not in result or "response" not in result:
                raise ValueError("Invalid answer-evaluation format")
            return result

        except Exception as e:
            logger.error(f"Error evaluating answer ({type(e).__name__}: {e}) - defaulting to hint")
            return {
                "correctness": "unclear",
                "misconception": None,
                "recommended_action": "hint",
                "response": "Let's think about that one a bit more - want to try again?",
            }

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
