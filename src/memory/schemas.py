"""
Pydantic schemas for study plan generation and persistence.

SessionPlan  — all details about a single study session.
               LLM fills: focus, topics, key_concepts, practice.
               System fills: session_id (UUID), status (lifecycle state).

StudyPlan    — the full plan: a list of SessionPlans plus top-level metadata.

Usage:
  plan = StudyPlan.model_validate_json(raw_llm_output)
  # session_id and status are auto-defaulted for every session.
"""

from __future__ import annotations

import uuid
from typing import List, Literal

from pydantic import BaseModel, Field

# Fields the LLM must not populate — handled entirely by the system.
_SYSTEM_FIELDS = {"session_id", "status"}


class SessionPlan(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    focus: str
    topics: List[str]
    key_concepts: List[str]
    practice: str
    status: Literal["not_started", "in_progress", "completed"] = "not_started"


class StudyPlan(BaseModel):
    total_sessions: int
    source_summary: str
    sessions: List[SessionPlan]


def session_plan_llm_schema() -> dict:
    """Return a JSON-schema dict describing only the fields the LLM should fill."""
    full = SessionPlan.model_json_schema()
    props = {k: v for k, v in full.get("properties", {}).items() if k not in _SYSTEM_FIELDS}
    required = [k for k in full.get("required", []) if k not in _SYSTEM_FIELDS]
    return {"type": "object", "properties": props, "required": required}
