"""Task 4 — Hard: contested rate limiter PR with TOCTOU race condition.

A 70-line PR adds a Redis-based rate limiter. The INCR + EXPIRE pattern
has a known race condition. Four reviewers disagree, author pushes back.
The PR description is intentionally misleading.
Context data reveals a previous PR with the same approach was REJECTED
and an ADR explicitly prohibits non-atomic rate limiting.

Exports:
    TASK_ID, FIXTURE, GOLD, CONTEXTS, METADATA, grade
"""

from __future__ import annotations

import json
from pathlib import Path

from pr_review_env.models import Action, GoldStandard, TaskMetadata
from pr_review_env.reward import compute_reward

TASK_ID = "hard"

_FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "pr_hard.json"

with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    FIXTURE: dict[str, object] = json.load(_f)

GOLD = GoldStandard(**FIXTURE["gold"])  # type: ignore[arg-type]
CONTEXTS: dict[str, str] = dict(FIXTURE.get("contexts", {}))  # type: ignore[arg-type]

METADATA = TaskMetadata(
    id=TASK_ID,
    description=(
        "Contested rate limiter PR with a TOCTOU race condition in the "
        "Redis INCR + EXPIRE pattern. Four reviewers disagree, author "
        "pushes back arguing the race window is negligible. Misleading "
        "PR title calls it a 'minor perf improvement.' Context reveals "
        "a previous PR with the same approach was rejected and an ADR "
        "explicitly prohibits non-atomic rate limiting."
    ),
    difficulty="hard",
    max_steps=8,
    expected_score_range=[0.20, 0.55],
)


def grade(action: Action, current_step: int = 1, requested_contexts: list[str] | None = None) -> float:
    """Score an agent action against the gold standard for this task."""
    return compute_reward(
        action=action,
        gold=GOLD,
        current_step=current_step,
        requested_contexts=requested_contexts or [],
    )
