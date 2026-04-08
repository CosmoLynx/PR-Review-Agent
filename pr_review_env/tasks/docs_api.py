"""Task 3 — Medium-Hard: API docs PR with incorrect code examples.

The PR updates API documentation but all code examples use v1 method
names with v2-style parameters. The examples will throw AttributeError
for any developer who copies them. CI doc-tests fail.

Exports:
    TASK_ID, FIXTURE, GOLD, CONTEXTS, METADATA, grade
"""

from __future__ import annotations

import json
from pathlib import Path

from pr_review_env.models import Action, GoldStandard, TaskMetadata
from pr_review_env.reward import compute_reward

TASK_ID = "docs_api"

_FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "pr_docs_api.json"

with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    FIXTURE: dict[str, object] = json.load(_f)

GOLD = GoldStandard(**FIXTURE["gold"])  # type: ignore[arg-type]
CONTEXTS: dict[str, str] = dict(FIXTURE.get("contexts", {}))  # type: ignore[arg-type]

METADATA = TaskMetadata(
    id=TASK_ID,
    description=(
        "API documentation update PR with subtly incorrect code examples. "
        "All examples use deprecated v1 method names with v2-style parameters. "
        "The examples will throw AttributeError for developers who copy them. "
        "CI doc-tests fail. Requires checking CI status and understanding the "
        "v1→v2 migration to catch the issue."
    ),
    difficulty="medium-hard",
    max_steps=8,
    expected_score_range=[0.40, 0.70],
)


def grade(action: Action, current_step: int = 1, requested_contexts: list[str] | None = None) -> float:
    """Score an agent action against the gold standard for this task."""
    return compute_reward(
        action=action,
        gold=GOLD,
        current_step=current_step,
        requested_contexts=requested_contexts or [],
    )
