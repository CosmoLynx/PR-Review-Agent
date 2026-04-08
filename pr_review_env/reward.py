"""Dense reward function for the PR Review Triage environment.

Scores agent actions along 6 independent axes:
  1. decision_score       (weight: 0.25) — exact match vs gold decision
  2. label_score          (weight: 0.20) — F1 between predicted and gold labels
  3. priority_score       (weight: 0.15) — ordinal distance penalty
  4. summary_score        (weight: 0.15) — keyword coverage + length heuristic
  5. context_efficiency   (weight: 0.15) — how well agent gathered relevant context
  6. reasoning_depth      (weight: 0.10) — whether summary references specific code concerns

Applies step penalty: -0.01 per step beyond step 1.
(Reduced from 0.02 to accommodate multi-step context gathering.)

This module is a pure function — no side effects, no LLM calls,
fully deterministic.
"""

from __future__ import annotations

from pr_review_env.models import Action, GoldStandard, PRIORITY_ORDER


# ── Axis weights ──────────────────────────────────────────────────────
W_DECISION = 0.25
W_LABEL = 0.20
W_PRIORITY = 0.15
W_SUMMARY = 0.15
W_CONTEXT = 0.15
W_DEPTH = 0.10

STEP_PENALTY = 0.01


def _decision_score(predicted: str, gold: str) -> float:
    """Binary: 1.0 if exact match, 0.0 otherwise."""
    return 1.0 if predicted == gold else 0.0


def _label_f1(predicted: list[str], gold: list[str]) -> float:
    """Compute F1 score between predicted and gold label sets.

    Returns 1.0 if both sets are empty. Provides partial credit
    for partial overlap.
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * (precision * recall) / (precision + recall)


def _priority_score(predicted: str, gold: str) -> float:
    """Ordinal distance penalty.

    exact match  → 1.0
    off by 1     → 0.5
    off by 2     → 0.25
    off by 3     → 0.0
    """
    pred_idx = PRIORITY_ORDER.get(predicted, 0)
    gold_idx = PRIORITY_ORDER.get(gold, 0)
    distance = abs(pred_idx - gold_idx)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.5
    elif distance == 2:
        return 0.25
    else:
        return 0.0


def _summary_score(summary: str, gold_keywords: list[str]) -> float:
    """Heuristic scoring for the review summary.

    Checks:
      - Length penalty: <20 chars → 0.0, >500 chars → halved
      - Keyword coverage: fraction of gold_keywords mentioned (case-insensitive)

    Returns a score in [0.0, 1.0].
    """
    if len(summary) < 20:
        return 0.0

    summary_lower = summary.lower()
    if not gold_keywords:
        keyword_coverage = 1.0
    else:
        hits = sum(1 for kw in gold_keywords if kw.lower() in summary_lower)
        keyword_coverage = hits / len(gold_keywords)

    score = keyword_coverage

    if len(summary) > 500:
        score *= 0.5

    return score


def _context_efficiency_score(
    requested_contexts: list[str],
    relevant_contexts: list[str],
) -> float:
    """Score how well the agent gathered relevant context before deciding.

    If no relevant contexts exist (e.g., easy task), the agent gets full
    marks regardless — we don't penalize for not exploring when exploration
    isn't needed.

    If relevant contexts exist, we compute F1 between requested and relevant:
      - Requesting all relevant contexts → high recall
      - Not requesting irrelevant ones → high precision
      - Balance of both → F1

    Returns a score in [0.0, 1.0].
    """
    if not relevant_contexts:
        # No context needed — agent gets full marks regardless
        return 1.0

    if not requested_contexts:
        # Relevant context exists but agent didn't explore — 0.0
        return 0.0

    req_set = set(requested_contexts)
    rel_set = set(relevant_contexts)

    true_positives = len(req_set & rel_set)
    precision = true_positives / len(req_set) if req_set else 0.0
    recall = true_positives / len(rel_set) if rel_set else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * (precision * recall) / (precision + recall)


def _reasoning_depth_score(summary: str, depth_keywords: list[str]) -> float:
    """Score whether the summary demonstrates deep technical analysis.

    Checks for task-specific technical terms that indicate the reviewer
    actually read and understood the code (e.g., specific variable names,
    function names, error types, design pattern names).

    If no depth_keywords exist, returns 1.0 (no depth required).
    Otherwise, returns the fraction of depth keywords found.

    Returns a score in [0.0, 1.0].
    """
    if not depth_keywords:
        return 1.0

    if len(summary) < 20:
        return 0.0

    summary_lower = summary.lower()
    hits = sum(1 for kw in depth_keywords if kw.lower() in summary_lower)
    return hits / len(depth_keywords)


def compute_reward(
    action: Action,
    gold: GoldStandard,
    current_step: int = 1,
    requested_contexts: list[str] | None = None,
) -> float:
    """Compute the dense reward for an agent action.

    This is a pure function with no side effects. All scoring is
    deterministic — no LLM calls, no randomness.

    Args:
        action: The agent's triage action.
        gold: The ground-truth gold standard for the task.
        current_step: Current step number (1-indexed). Steps beyond 1
            incur a penalty.
        requested_contexts: List of context types the agent requested
            before making the review decision.

    Returns:
        Reward float in approximately [0.0, 1.0] (can go slightly
        negative due to step penalty).
    """
    if action.action_type == "request_context":
        # Context requests don't get a final reward — return 0.0
        # The reward comes on the final review step
        return 0.0

    d_score = _decision_score(action.decision, gold.decision)  # type: ignore[arg-type]
    l_score = _label_f1(action.labels or [], gold.labels)
    p_score = _priority_score(action.priority, gold.priority)  # type: ignore[arg-type]
    s_score = _summary_score(action.review_summary or "", gold.gold_keywords)
    c_score = _context_efficiency_score(
        requested_contexts or [], gold.relevant_contexts
    )
    r_score = _reasoning_depth_score(
        action.review_summary or "", gold.depth_keywords
    )

    # Weighted sum of 6 axes
    raw_reward = (
        W_DECISION * d_score
        + W_LABEL * l_score
        + W_PRIORITY * p_score
        + W_SUMMARY * s_score
        + W_CONTEXT * c_score
        + W_DEPTH * r_score
    )

    # Step penalty: -0.01 per step beyond step 1
    penalty = max(0, current_step - 1) * STEP_PENALTY
    final_reward = max(0.0, raw_reward - penalty)

    return round(final_reward, 4)


def compute_reward_breakdown(
    action: Action,
    gold: GoldStandard,
    current_step: int = 1,
    requested_contexts: list[str] | None = None,
) -> dict[str, float]:
    """Return the full reward breakdown as a dict.

    Useful for the info field in StepResult.
    """
    if action.action_type == "request_context":
        return {
            "decision_score": 0.0,
            "label_score": 0.0,
            "priority_score": 0.0,
            "summary_score": 0.0,
            "context_efficiency_score": 0.0,
            "reasoning_depth_score": 0.0,
            "step_penalty": 0.0,
            "total": 0.0,
            "note": "context_request_step",
        }

    d_score = _decision_score(action.decision, gold.decision)  # type: ignore[arg-type]
    l_score = _label_f1(action.labels or [], gold.labels)
    p_score = _priority_score(action.priority, gold.priority)  # type: ignore[arg-type]
    s_score = _summary_score(action.review_summary or "", gold.gold_keywords)
    c_score = _context_efficiency_score(
        requested_contexts or [], gold.relevant_contexts
    )
    r_score = _reasoning_depth_score(
        action.review_summary or "", gold.depth_keywords
    )

    raw_reward = (
        W_DECISION * d_score
        + W_LABEL * l_score
        + W_PRIORITY * p_score
        + W_SUMMARY * s_score
        + W_CONTEXT * c_score
        + W_DEPTH * r_score
    )
    penalty = max(0, current_step - 1) * STEP_PENALTY
    final_reward = max(0.0, raw_reward - penalty)

    return {
        "decision_score": round(d_score, 4),
        "label_score": round(l_score, 4),
        "priority_score": round(p_score, 4),
        "summary_score": round(s_score, 4),
        "context_efficiency_score": round(c_score, 4),
        "reasoning_depth_score": round(r_score, 4),
        "step_penalty": round(-penalty, 4),
        "total": round(final_reward, 4),
        "weights": {
            "decision": W_DECISION,
            "label": W_LABEL,
            "priority": W_PRIORITY,
            "summary": W_SUMMARY,
            "context": W_CONTEXT,
            "depth": W_DEPTH,
        },
    }
