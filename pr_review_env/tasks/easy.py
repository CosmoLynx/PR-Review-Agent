"""Task 1 — Easy: trivial bugfix PR (off-by-one in list slicing).

Exports:
    TASK_ID   – unique task identifier
    FIXTURE   – raw PR data dict loaded from fixture JSON
    GOLD      – GoldStandard instance
    METADATA  – TaskMetadata instance
    CONTEXTS  – dict of available context data
    grade     – convenience scorer: Action → float
"""

from __future__ import annotations

import json
from pathlib import Path

from pr_review_env.models import Action, GoldStandard, TaskMetadata
from pr_review_env.reward import compute_reward

TASK_ID = "easy"

_FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "pr_easy.json"

with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    FIXTURE: dict[str, object] = json.load(_f)

GOLD = GoldStandard(**FIXTURE["gold"])  # type: ignore[arg-type]
CONTEXTS: dict[str, str] = dict(FIXTURE.get("contexts", {}))  # type: ignore[arg-type]

METADATA = TaskMetadata(
    id=TASK_ID,
    description=(
        "Trivial bugfix PR — off-by-one error in a Python list slicing "
        "utility. Clean diff, clear description, unanimous reviewer approval."
    ),
    difficulty="easy",
    max_steps=8,
    expected_score_range=[0.80, 1.00],
)


def grade(action: Action, current_step: int = 1, requested_contexts: list[str] | None = None) -> float:
    """Score an agent action against the gold standard for this task."""
    return compute_reward(
        action=action,
        gold=GOLD,
        current_step=current_step,
        requested_contexts=requested_contexts or [],
    )
