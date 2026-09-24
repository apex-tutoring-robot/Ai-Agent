"""
Session: the unit of "one conversation" (from wake word to idle timeout),
analogous to a call_id in a phone-based voice agent. Created fresh each time
_handle_wake_word fires.

Deliberately separate from ProfileManager's per-profile conversation history
(that's long-term memory spanning many sessions across days); this is
short-lived, in-memory bookkeeping for the CURRENT conversation only - which
tools were called, what task is active, the most recent turn's latency
breakdown. Doesn't replace JarvisBot's existing threading/audio state
(_bot_is_speaking, the various threading.Event flags) - those are low-level
control primitives, not conversational/business state.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    profile_id: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    turn_count: int = 0

    # What kind of turn is currently/most-recently active - "teaching_math",
    # "onboarding", or None for normal chat. Useful for logging/debugging
    # ("why did it go into teaching mode") without grepping through flags.
    active_task: Optional[str] = None

    # Audit trail of tool invocations this session (see llm_client.py's
    # function-calling support) - what was called, with what arguments, and
    # what came back. Mirrors the doc's "tool_results" session field.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Most recent turn's latency breakdown (see _speaker_loop's timing),
    # e.g. {"llm_ttft_ms": 241, "tts_first_byte_ms": 137, "total_ms": 758}
    last_latency_breakdown: Dict[str, float] = field(default_factory=dict)

    def record_tool_call(self, tool_name: str, arguments: dict, result: Any) -> None:
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "at": time.time(),
        })

    def new_turn(self) -> None:
        self.turn_count += 1
