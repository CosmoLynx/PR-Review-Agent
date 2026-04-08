"""Pydantic v2 models for the PR Review Triage environment.

Defines Observation, Action, Reward, StepResult, and related types
following strict Pydantic v2 conventions with ConfigDict(extra='forbid').

This environment supports multi-step interaction:
  - Agents can request additional context before making a final review decision.
  - Actions have an action_type discriminator: "request_context" or "review".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Valid enum values ──────────────────────────────────────────────────
VALID_LABELS: list[str] = [
    "bug",
    "security",
    "enhancement",
    "documentation",
    "breaking-change",
    "needs-tests",
    "trivial",
    "urgent",
    "performance",
    "memory-leak",
]

VALID_DECISIONS: list[str] = ["approve", "request_changes", "close"]

VALID_PRIORITIES: list[str] = ["low", "medium", "high", "critical"]

VALID_CONTEXT_TYPES: list[str] = [
    "file_tree",
    "ci_status",
    "git_blame",
    "related_prs",
    "test_coverage",
]

PRIORITY_ORDER: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


# ── Observation ────────────────────────────────────────────────────────
class Observation(BaseModel):
    """What the agent sees after a reset or step."""

    model_config = ConfigDict(extra="forbid")

    pr_id: int = Field(..., description="Pull request identifier")
    title: str = Field(..., description="PR title")
    description: str = Field(..., description="PR description/body")
    diff: str = Field(..., description="Unified diff (30-100+ lines)")
    comments: list[str] = Field(
        ..., description="2-6 inline reviewer comments"
    )
    files_changed: list[str] = Field(..., description="List of changed files")
    author: str = Field(..., description="PR author username")
    base_branch: str = Field(..., description="Target branch")
    additions: int = Field(..., ge=0, description="Lines added")
    deletions: int = Field(..., ge=0, description="Lines deleted")
    current_step: int = Field(..., ge=0, description="Current step number")
    max_steps: int = Field(..., gt=0, description="Maximum allowed steps")
    task_name: str = Field(..., description="Task identifier")
    additional_context: dict[str, str] = Field(
        default_factory=dict,
        description="Accumulated context from request_context actions",
    )
    available_contexts: list[str] = Field(
        default_factory=list,
        description="Context types still available to request",
    )


# ── Action ─────────────────────────────────────────────────────────────
class Action(BaseModel):
    """Agent action — either request additional context or submit final review.

    When action_type="request_context":
        - context_type must be set to one of the valid context types
        - decision, labels, priority, review_summary are ignored

    When action_type="review":
        - decision, labels, priority, review_summary must be set
        - context_type is ignored
    """

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["request_context", "review"] = Field(
        default="review",
        description="Action type: gather context or submit final review",
    )

    # ── Context request fields ────────────────────────────────────────
    context_type: Literal[
        "file_tree", "ci_status", "git_blame", "related_prs", "test_coverage"
    ] | None = Field(
        default=None,
        description="Context to request (required when action_type='request_context')",
    )

    # ── Review fields ─────────────────────────────────────────────────
    decision: Literal["approve", "request_changes", "close"] | None = Field(
        default=None,
        description="Triage decision (required when action_type='review')",
    )
    labels: list[str] | None = Field(
        default=None,
        description="Labels from the allowed set (required when action_type='review')",
    )
    priority: Literal["low", "medium", "high", "critical"] | None = Field(
        default=None,
        description="Priority assessment (required when action_type='review')",
    )
    review_summary: str | None = Field(
        default=None,
        description="1-3 sentence written review (required when action_type='review')",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        """Ensure the correct fields are set for each action_type."""
        if self.action_type == "request_context":
            if self.context_type is None:
                raise ValueError(
                    "context_type is required when action_type='request_context'"
                )
        elif self.action_type == "review":
            missing: list[str] = []
            if self.decision is None:
                missing.append("decision")
            if self.labels is None:
                missing.append("labels")
            if self.priority is None:
                missing.append("priority")
            if self.review_summary is None:
                missing.append("review_summary")
            if missing:
                raise ValueError(
                    f"Fields required for action_type='review': {missing}"
                )
        return self


# ── Reward breakdown ──────────────────────────────────────────────────
class RewardBreakdown(BaseModel):
    """Detailed breakdown of the 6 scoring axes."""

    model_config = ConfigDict(extra="forbid")

    decision_score: float = Field(
        ..., ge=0.0, le=1.0, description="Correct decision vs gold"
    )
    label_score: float = Field(
        ..., ge=0.0, le=1.0, description="F1 between predicted and gold labels"
    )
    priority_score: float = Field(
        ..., ge=0.0, le=1.0, description="Ordinal distance penalty"
    )
    summary_score: float = Field(
        ..., ge=0.0, le=1.0, description="Keyword and length heuristic"
    )
    context_efficiency_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How well agent gathered relevant context before deciding",
    )
    reasoning_depth_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Whether the summary references specific code concerns",
    )
    step_penalty: float = Field(
        ..., le=0.0, description="Penalty for extra steps"
    )
    total: float = Field(..., description="Final reward after penalty")


# ── StepResult ─────────────────────────────────────────────────────────
class StepResult(BaseModel):
    """Returned by /step — contains next observation, reward, done flag."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Whether the episode is over")
    info: dict[str, object] = Field(
        default_factory=dict,
        description="Extra info (reward breakdown, etc.)",
    )


# ── Gold standard (internal, not exposed to agent) ────────────────────
class GoldStandard(BaseModel):
    """Ground-truth labels for scoring an action."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "request_changes", "close"]
    labels: list[str]
    priority: Literal["low", "medium", "high", "critical"]
    gold_keywords: list[str] = Field(
        ..., description="Keywords the review summary should mention"
    )
    relevant_contexts: list[str] = Field(
        default_factory=list,
        description="Context types that would help make the correct decision",
    )
    depth_keywords: list[str] = Field(
        default_factory=list,
        description="Technical terms showing deep code analysis",
    )


# ── Task metadata ─────────────────────────────────────────────────────
class TaskMetadata(BaseModel):
    """Metadata returned by /tasks endpoint."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    difficulty: Literal["easy", "medium", "medium-hard", "hard", "expert"]
    max_steps: int = Field(..., gt=0)
    expected_score_range: list[float] = Field(
        ..., min_length=2, max_length=2
    )
