"""Tasks package — exports registry of available tasks."""

from __future__ import annotations

from pr_review_env.tasks.easy import TASK_ID as EASY_ID
from pr_review_env.tasks.medium import TASK_ID as MEDIUM_ID
from pr_review_env.tasks.docs_api import TASK_ID as DOCS_API_ID
from pr_review_env.tasks.hard import TASK_ID as HARD_ID
from pr_review_env.tasks.async_refactor import TASK_ID as ASYNC_REFACTOR_ID

TASK_IDS: list[str] = [EASY_ID, MEDIUM_ID, DOCS_API_ID, HARD_ID, ASYNC_REFACTOR_ID]

__all__ = [
    "TASK_IDS",
    "EASY_ID",
    "MEDIUM_ID",
    "DOCS_API_ID",
    "HARD_ID",
    "ASYNC_REFACTOR_ID",
]
