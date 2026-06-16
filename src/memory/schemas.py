"""
Pydantic schemas for study plan generation and persistence.

SessionPlan — represents a session through its full lifecycle.
              LLM fills at plan creation: focus, topics, key_concepts, practice.
              System fills: session_id (UUID), status.
              LLM fills at session end: summary, struggles, next_focus,
                                        topics_covered, performance_notes.

StudyPlan   — the full plan: a list of SessionPlans plus top-level metadata.
              assessment_summary is filled at plan creation from diagnostic Q&A.

TopicList, DiagnosticQuestions, SessionDebrief, SessionProgress — lightweight
              models used for individual LLM structured-output calls. They use
              extra="forbid" so their JSON schemas are strict-mode compatible.
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence summary of what was actually covered and how the session went. Filled after the session ends.",
    )
    struggles: List[str] = Field(
        default_factory=list,
        description="Concepts or skills the student found difficult. Filled after the session ends.",
    )
    next_focus: str = Field(
        default="",
        description="The single most important concept to prioritise at the start of the next session. Filled after the session ends.",
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="Topics actually covered during the session (may differ from the planned list). Filled after the session ends.",
    )
    performance_notes: str = Field(
        default="",
        description="Brief qualitative assessment of the student's engagement and understanding. Filled after the session ends.",
    )


class StudyPlan(BaseModel):
    topic: str = Field(
        default="",
        description="The specific subject or topic the student chose to focus on.",
    )
    source_summary: str = Field(
        description="One sentence describing what the uploaded syllabus covers overall.",
    )
    source_text: str = Field(
        default="",
        description="Full extracted syllabus text. Set by the system — do not populate.",
    )
    assessment_summary: str = Field(
        default="",
        description="One sentence summarising the student's prior knowledge level based on their diagnostic answers.",
    )
    sessions: List[SessionPlan] = Field(
        description="Ordered list of study sessions. Each session builds on the previous one.",
    )


# ── Lightweight structured-output models ─────────────────────────────────────
# extra="forbid" makes Pydantic emit additionalProperties: false in the JSON
# schema, which is required for OpenAI structured outputs strict mode.

class TopicList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: List[str] = Field(description="Distinct topic or subject-area names found in the syllabus.")


class DiagnosticQuestions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: List[str] = Field(description="Exactly 2 open-ended prior-knowledge questions for the student.")


class SessionDebrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="2-3 sentence summary of what was covered in the session.")
    struggles: List[str] = Field(description="Concepts or skills the student found difficult.")
    next_focus: str = Field(description="The single most important concept to prioritise next session.")
    topics_covered: List[str] = Field(description="Topics actually covered during this session.")
    performance_notes: str = Field(description="Brief assessment of student engagement and understanding.")


class SessionProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="What was covered so far in 1-2 sentences.")
    topics_covered: List[str] = Field(description="Topics actually covered so far in this session.")
    performance_notes: str = Field(description="Brief note on student understanding and engagement.")


# ── JSON schema accessors ─────────────────────────────────────────────────────

def session_plan_llm_schema() -> dict:
    """Strict-compatible schema describing only the fields the LLM should fill for a session."""
    full = SessionPlan.model_json_schema()
    props = {k: v for k, v in full.get("properties", {}).items() if k not in _SYSTEM_FIELDS}
    required = [k for k in full.get("required", []) if k not in _SYSTEM_FIELDS]
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def study_plan_llm_schema() -> dict:
    """Strict-compatible schema for full study plan generation (LLM-fillable fields only)."""
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The specific topic the student chose."},
            "source_summary": {"type": "string", "description": "One sentence describing what the syllabus covers."},
            "assessment_summary": {"type": "string", "description": "One sentence summarising the student's prior knowledge."},
            "sessions": {
                "type": "array",
                "items": session_plan_llm_schema(),
                "description": "Exactly one introductory session.",
            },
        },
        "required": ["topic", "source_summary", "assessment_summary", "sessions"],
        "additionalProperties": False,
    }


def topic_list_schema() -> dict:
    return TopicList.model_json_schema()


def diagnostic_questions_schema() -> dict:
    return DiagnosticQuestions.model_json_schema()


def session_debrief_schema() -> dict:
    return SessionDebrief.model_json_schema()


def session_progress_schema() -> dict:
    return SessionProgress.model_json_schema()
