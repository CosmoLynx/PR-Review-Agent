"""Task 2 — Medium: security-sensitive auth middleware refactor.

The PR removes token expiry checking under the guise of "cleanup."
A careful reviewer should catch the security regression.
Context data (git_blame, related_prs) reveals the expiry check was
added due to a P0 security incident.

Exports:
    TASK_ID, FIXTURE, GOLD, CONTEXTS, METADATA, grade
"""

from __future__ import annotations

import json
from pathlib import Path

from pr_review_env.models import Action, GoldStandard, TaskMetadata
from pr_review_env.reward import compute_reward

TASK_ID = "medium"

_FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "pr_medium.json"

with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    FIXTURE: dict[str, object] = json.load(_f)

GOLD = GoldStandard(**FIXTURE["gold"])  # type: ignore[arg-type]
CONTEXTS: dict[str, str] = dict(FIXTURE.get("contexts", {}))  # type: ignore[arg-type]

METADATA = TaskMetadata(
    id=TASK_ID,
    description=(
        "Security-sensitive auth middleware refactor. The PR removes token "
        "expiry checks disguised as legacy cleanup. Split reviewer opinions. "
        "Requires careful diff reading to catch the security regression. "
        "Context data reveals the check was added for a P0 security incident."
    ),
    difficulty="medium",
    max_steps=8,
    expected_score_range=[0.50, 0.75],
)


def grade(action: Action, current_step: int = 1, requested_contexts: list[str] | None = None) -> float:
    """Score an agent action against the gold standard for this task."""
    return compute_reward(
        action=action,
        gold=GOLD,
        current_step=current_step,
        requested_contexts=requested_contexts or [],
    )
