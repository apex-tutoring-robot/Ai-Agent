"""
Pydantic schemas for study plan generation and persistence.

SessionPlan — represents a session through its full lifecycle.
              LLM fills at plan creation: focus, topics, key_concepts, practice.
              System fills: session_id (UUID), status.
              LLM fills at session end: summary, struggles, next_focus,
                                        topics_covered, performance_notes.

StudyPlan   — the full plan: a list of SessionPlans plus top-level metadata.
              assessment_summary is filled at plan creation from diagnostic Q&A.

Usage:
  plan = StudyPlan.model_validate_json(raw_llm_output)
  # Validate end-of-session debrief by merging into the existing session:
  updated = SessionPlan.model_validate({**existing_session, **llm_debrief})
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Fields the LLM must not populate during plan generation.
_SYSTEM_FIELDS = {
    "session_id", "status",
    "summary", "struggles", "next_focus",
    "topics_covered", "performance_notes",
}


class SessionPlan(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="System-assigned unique identifier for this session. Do not populate.",
    )
    focus: str = Field(
        description="One sentence describing the main learning goal for this session, e.g. 'Introduction to equivalent fractions'.",
    )
    topics: List[str] = Field(
        description="Ordered list of specific topics to cover during this session, e.g. ['What is a fraction?', 'Numerator and denominator', 'Comparing fractions with the same denominator'].",
    )
    key_concepts: List[str] = Field(
        description="Core vocabulary and concepts the student must understand by the end of the session, e.g. ['numerator', 'denominator', 'equivalent fraction'].",
    )
    practice: str = Field(
        description="A short description of the hands-on practice activity or exercise set for this session, e.g. 'Draw and shade fraction bars to compare 1/2, 1/3, and 1/4'.",
    )
    status: Literal["not_started", "in_progress", "completed"] = Field(
        default="not_started",
        description="System-managed session lifecycle status. Do not populate.",
    )
    # Populated by LLM after the session ends
    summary: Optional[str] = Field(
        default=None,
        description="2–3 sentence summary of what was actually covered and how the session went. Filled after the session ends.",
    )
    struggles: List[str] = Field(
        default_factory=list,
        description="List of specific concepts or skills the student found difficult during the session, e.g. ['converting improper fractions', 'finding common denominators']. Filled after the session ends.",
    )
    next_focus: str = Field(
        default="",
        description="The single most important concept or skill to prioritise at the start of the next session, based on where the student struggled. Filled after the session ends.",
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="List of topics that were actually covered during the session (may differ from the planned topics list). Filled after the session ends.",
    )
    performance_notes: str = Field(
        default="",
        description="Brief qualitative assessment of the student's engagement, confidence, and understanding during the session, e.g. 'Student understood basic fractions quickly but struggled with equivalent fractions'. Filled after the session ends.",
    )


class StudyPlan(BaseModel):
    topic: str = Field(
        default="",
        description="The specific subject or topic the student chose to focus on, e.g. 'Fractions' or 'Introduction to Algebra'.",
    )
    source_summary: str = Field(
        description="One sentence describing what the uploaded syllabus covers overall, e.g. 'Grade 4 math syllabus covering fractions, decimals, and basic geometry across three terms'.",
    )
    source_text: str = Field(
        default="",
        description="Full extracted syllabus text. Set by the system after extraction — do not populate.",
    )
    assessment_summary: str = Field(
        default="",
        description="One sentence summarising the student's prior knowledge level based on their diagnostic answers, e.g. 'Student can identify basic fractions but has not encountered equivalent fractions yet'.",
    )
    sessions: List[SessionPlan] = Field(
        description="Ordered list of study sessions. Each session builds on the previous one.",
    )


def session_plan_llm_schema() -> dict:
    """Return a JSON-schema dict describing only the fields the LLM should fill."""
    full = SessionPlan.model_json_schema()
    props = {k: v for k, v in full.get("properties", {}).items() if k not in _SYSTEM_FIELDS}
    required = [k for k in full.get("required", []) if k not in _SYSTEM_FIELDS]
    return {"type": "object", "properties": props, "required": required}
