"""Task 5 — Expert: async DB refactor with connection pool poisoning.

A PR refactors database access from sync to async. The diff looks correct
but removes the explicit rollback-on-error pattern from the old code.
Under error conditions, connections are returned to the pool with dirty
transaction state, eventually causing cascading failures.

Context reveals a previous production incident with the exact same root cause
and an ADR mandating transaction context managers.

Exports:
    TASK_ID, FIXTURE, GOLD, CONTEXTS, METADATA, grade
"""

from __future__ import annotations

import json
from pathlib import Path

from pr_review_env.models import Action, GoldStandard, TaskMetadata
from pr_review_env.reward import compute_reward

TASK_ID = "async_refactor"

_FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "pr_async_refactor.json"

with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    FIXTURE: dict[str, object] = json.load(_f)

GOLD = GoldStandard(**FIXTURE["gold"])  # type: ignore[arg-type]
CONTEXTS: dict[str, str] = dict(FIXTURE.get("contexts", {}))  # type: ignore[arg-type]

METADATA = TaskMetadata(
    id=TASK_ID,
    description=(
        "Async database refactor that introduces connection pool poisoning. "
        "The old sync code had proper rollback-on-error via contextmanager. "
        "The new async code releases connections without rolling back failed "
        "transactions, leading to dirty connection state in the pool. Under "
        "sustained error rates, the entire pool becomes poisoned causing "
        "cascading failures. Context reveals a previous incident with the "
        "exact same root cause and an ADR mandating transaction context managers."
    ),
    difficulty="expert",
    max_steps=10,
    expected_score_range=[0.15, 0.45],
)


def grade(action: Action, current_step: int = 1, requested_contexts: list[str] | None = None) -> float:
    """Score an agent action against the gold standard for this task."""
    return compute_reward(
        action=action,
        gold=GOLD,
        current_step=current_step,
        requested_contexts=requested_contexts or [],
    )
