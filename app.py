"""FastAPI application for the PR Review Triage environment.

Endpoints:
    POST /reset   — Start a new episode for a given task
    POST /step    — Submit an action (context request or review) and receive result
    GET  /state   — Get current environment state
    GET  /tasks   — List all available tasks
    GET  /health  — Health check

Sessions are managed via an in-memory store keyed by the
X-Session-ID header (defaults to "default").
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from pr_review_env.env import PRReviewEnv, list_task_metadata
from pr_review_env.models import Action, Observation, StepResult, TaskMetadata

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PR Review Triage Environment",
    description=(
        "OpenEnv environment for pull request code review triage. "
        "Supports multi-step interaction: agents can request additional "
        "context (file tree, CI status, git blame, related PRs, test coverage) "
        "before submitting a final triage decision. 5 tasks spanning "
        "easy to expert difficulty."
    ),
    version="2.0.0",
)

# ── Session store ─────────────────────────────────────────────────────
_sessions: dict[str, PRReviewEnv] = {}


def _get_session(
    x_session_id: Annotated[str | None, Header()] = None,
) -> PRReviewEnv:
    """Resolve or create a session environment by header.

    If no X-Session-ID header is provided, uses "default".
    """
    session_id = x_session_id or "default"
    if session_id not in _sessions:
        _sessions[session_id] = PRReviewEnv()
        logger.info("Created new session: %s", session_id)
    return _sessions[session_id]


# ── Request / Response models ────────────────────────────────────────
class ResetRequest(BaseModel):
    """Body for POST /reset."""

    model_config = ConfigDict(extra="forbid")

    task: str


class ResetResponse(BaseModel):
    """Response from POST /reset — includes session_id and observation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    observation: Observation


class HealthResponse(BaseModel):
    """Response from GET /health."""

    model_config = ConfigDict(extra="forbid")

    status: str


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/reset", response_model=ResetResponse)
def reset_endpoint(
    body: ResetRequest,
    x_session_id: Annotated[str | None, Header()] = None,
) -> ResetResponse:
    """Reset the environment for a new task episode.

    Generates a new session_id if none is provided. Returns the
    initial observation along with the session_id for subsequent calls.
    """
    session_id = x_session_id or str(uuid.uuid4())

    env = PRReviewEnv()
    _sessions[session_id] = env

    try:
        observation = env.reset(body.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Reset session=%s task=%s", session_id, body.task)
    return ResetResponse(session_id=session_id, observation=observation)


@app.post("/step", response_model=StepResult)
def step_endpoint(
    action: Action,
    env: Annotated[PRReviewEnv, Depends(_get_session)],
) -> StepResult:
    """Submit an action (context request or review) and receive the step result.

    Context request actions return reward=0.0 and done=False.
    Review actions return the final reward and done=True.
    """
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.get("/state")
def state_endpoint(
    env: Annotated[PRReviewEnv, Depends(_get_session)],
) -> dict[str, object]:
    """Get the current environment state."""
    return env.get_state()


@app.get("/tasks", response_model=list[TaskMetadata])
def tasks_endpoint() -> list[TaskMetadata]:
    """List all available tasks with metadata."""
    return list_task_metadata()


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    """Health check."""
    return HealthResponse(status="ok")
