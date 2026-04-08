"""PRReviewEnv — full OpenEnv interface for the PR Review Triage environment.

Manages episode state, observation construction, action validation,
reward computation, and step counting.

Supports multi-step interaction:
  - Agents can submit "request_context" actions to gather additional information
    (file tree, CI status, git blame, related PRs, test coverage)
  - Agents submit a final "review" action with their triage decision
  - The reward function considers context gathering efficiency
"""

from __future__ import annotations

import logging
from typing import Any

from pr_review_env.models import (
    Action,
    GoldStandard,
    Observation,
    StepResult,
    TaskMetadata,
    VALID_CONTEXT_TYPES,
)
from pr_review_env.reward import compute_reward, compute_reward_breakdown
from pr_review_env.tasks import easy, medium, docs_api, hard, async_refactor

logger = logging.getLogger(__name__)

# ── Task registry ─────────────────────────────────────────────────────
_TASK_MODULES: dict[str, Any] = {
    "easy": easy,
    "medium": medium,
    "docs_api": docs_api,
    "hard": hard,
    "async_refactor": async_refactor,
}


def get_task_module(task_name: str) -> Any:
    """Look up a task module by name. Raises ValueError if unknown."""
    if task_name not in _TASK_MODULES:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available tasks: {list(_TASK_MODULES.keys())}"
        )
    return _TASK_MODULES[task_name]


def list_task_metadata() -> list[TaskMetadata]:
    """Return metadata for all registered tasks."""
    return [mod.METADATA for mod in _TASK_MODULES.values()]


# ── Environment ───────────────────────────────────────────────────────
class PRReviewEnv:
    """Stateful environment for a single PR review triage episode.

    Lifecycle:
        1. Call reset(task_name) to begin an episode → get first Observation
        2. Optionally call step(request_context) to gather additional context
        3. Call step(review) to submit final triage decision → get reward
        4. When done=True, the episode is over. Call reset() to start again.

    Multi-step interaction:
        The agent can request up to 5 types of additional context before
        making their final review decision:
          - file_tree: directory structure of affected areas
          - ci_status: CI pipeline results and warnings
          - git_blame: authorship and commit history of changed files
          - related_prs: previous PRs related to this change
          - test_coverage: detailed coverage report

        Each context request costs a step but provides information that
        may be critical for making the correct decision. The reward function
        scores context gathering efficiency.
    """

    def __init__(self) -> None:
        self._task_name: str | None = None
        self._fixture: dict[str, object] = {}
        self._gold: GoldStandard | None = None
        self._contexts: dict[str, str] = {}
        self._max_steps: int = 8
        self._current_step: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._requested_contexts: list[str] = []
        self._revealed_contexts: dict[str, str] = {}

    @property
    def is_active(self) -> bool:
        """Whether an episode is in progress."""
        return not self._done

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def task_name(self) -> str | None:
        return self._task_name

    def reset(self, task_name: str) -> Observation:
        """Reset the environment for a new episode on the given task.

        Args:
            task_name: One of the registered task names.

        Returns:
            The initial Observation.
        """
        task_mod = get_task_module(task_name)

        self._task_name = task_name
        self._fixture = task_mod.FIXTURE
        self._gold = task_mod.GOLD
        self._contexts = task_mod.CONTEXTS
        self._max_steps = task_mod.METADATA.max_steps
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._requested_contexts = []
        self._revealed_contexts = {}

        logger.info("Environment reset for task '%s'", task_name)
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Take one step in the environment with the given action.

        Args:
            action: Either a context request or a final review decision.

        Returns:
            StepResult with new observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: If the episode hasn't started or is already done.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )
        if self._gold is None:
            raise RuntimeError("No gold standard loaded. Call reset() first.")

        self._current_step += 1

        # Handle context request
        if action.action_type == "request_context":
            return self._handle_context_request(action)

        # Handle final review
        return self._handle_review(action)

    def _handle_context_request(self, action: Action) -> StepResult:
        """Process a context request action."""
        ctx_type = action.context_type
        assert ctx_type is not None  # validated by model

        if ctx_type in self._requested_contexts:
            # Already requested — provide the same data, no penalty beyond step cost
            logger.info(
                "Step %d: Context '%s' already requested (duplicate)",
                self._current_step, ctx_type,
            )
        elif ctx_type in self._contexts:
            # New context request — reveal the data
            self._requested_contexts.append(ctx_type)
            self._revealed_contexts[ctx_type] = self._contexts[ctx_type]
            logger.info(
                "Step %d: Context '%s' revealed",
                self._current_step, ctx_type,
            )
        else:
            # Context type exists but no data available for this task
            self._requested_contexts.append(ctx_type)
            self._revealed_contexts[ctx_type] = "No additional context available for this item."
            logger.info(
                "Step %d: Context '%s' requested but no data available",
                self._current_step, ctx_type,
            )

        # Check if max steps reached without a review
        done = self._current_step >= self._max_steps

        obs = self._build_observation()
        info: dict[str, object] = {
            "action_type": "request_context",
            "context_type": ctx_type,
            "contexts_gathered": list(self._requested_contexts),
            "steps_remaining": self._max_steps - self._current_step,
        }

        if done:
            info["warning"] = "Max steps reached without submitting a review."
            self._done = True

        return StepResult(
            observation=obs,
            reward=0.0,  # No reward for context requests
            done=done,
            info=info,
        )

    def _handle_review(self, action: Action) -> StepResult:
        """Process a final review action."""
        assert self._gold is not None

        # Compute reward
        reward = compute_reward(
            action=action,
            gold=self._gold,
            current_step=self._current_step,
            requested_contexts=self._requested_contexts,
        )
        breakdown = compute_reward_breakdown(
            action=action,
            gold=self._gold,
            current_step=self._current_step,
            requested_contexts=self._requested_contexts,
        )
        self._cumulative_reward += reward

        # Review action is always terminal
        self._done = True

        obs = self._build_observation()

        info: dict[str, object] = {
            "action_type": "review",
            "reward_breakdown": breakdown,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "contexts_gathered": list(self._requested_contexts),
            "total_steps": self._current_step,
        }

        logger.info(
            "Step %d/%d — REVIEW — reward=%.4f, done=True",
            self._current_step,
            self._max_steps,
            reward,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=True,
            info=info,
        )

    def get_state(self) -> dict[str, object]:
        """Return current environment state as a dict (for /state endpoint)."""
        return {
            "task_name": self._task_name,
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "is_active": self.is_active,
            "requested_contexts": list(self._requested_contexts),
            "available_contexts": [
                ct for ct in VALID_CONTEXT_TYPES
                if ct not in self._requested_contexts
            ],
        }

    def _build_observation(self) -> Observation:
        """Construct an Observation from the current fixture state."""
        available = [
            ct for ct in VALID_CONTEXT_TYPES
            if ct not in self._requested_contexts
        ]

        return Observation(
            pr_id=int(self._fixture["pr_id"]),  # type: ignore[arg-type]
            title=str(self._fixture["title"]),
            description=str(self._fixture["description"]),
            diff=str(self._fixture["diff"]),
            comments=list(self._fixture["comments"]),  # type: ignore[arg-type]
            files_changed=list(self._fixture["files_changed"]),  # type: ignore[arg-type]
            author=str(self._fixture["author"]),
            base_branch=str(self._fixture["base_branch"]),
            additions=int(self._fixture["additions"]),  # type: ignore[arg-type]
            deletions=int(self._fixture["deletions"]),  # type: ignore[arg-type]
            current_step=self._current_step,
            max_steps=self._max_steps,
            task_name=self._task_name or "",
            additional_context=dict(self._revealed_contexts),
            available_contexts=available,
        )
