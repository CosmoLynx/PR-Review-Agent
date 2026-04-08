"""Baseline inference script for the PR Review Triage environment.

Uses the OpenAI client to run all 5 tasks sequentially with a
multi-step strategy: gather relevant context, then make a decision.

Environment variables:
    API_BASE_URL  — LLM API base URL (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — Hugging Face token (used as api_key)
    ENV_BASE_URL  — Environment server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load .env file

# ── Configuration ─────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK = 8
TASKS = ["easy", "medium", "docs_api", "hard", "async_refactor"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── OpenAI client ─────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-token-set",
)

# ── System prompt for context gathering ───────────────────────────────
CONTEXT_SYSTEM_PROMPT = """You are a senior software engineer performing pull request code review triage.

PHASE 1: CONTEXT GATHERING
You are looking at a new PR. Before making your triage decision, you should gather additional context to make an informed review. You can request the following context types:
- "file_tree": Directory structure of affected areas
- "ci_status": CI pipeline results and warnings
- "git_blame": Authorship and commit history of changed files
- "related_prs": Previous PRs related to this change
- "test_coverage": Detailed coverage report

Based on the PR observation below, decide which context would be most valuable to request.

You MUST respond with ONLY a valid JSON object:
{
  "action_type": "request_context",
  "context_type": "<one of: file_tree, ci_status, git_blame, related_prs, test_coverage>"
}

Choose the MOST valuable context to request given what you've seen so far. Consider:
- If the PR touches security-sensitive code → request "git_blame" or "related_prs"
- If reviewer comments suggest concerns → request "ci_status" or "test_coverage"
- If the PR description seems misleading → request "related_prs"
- If the diff is large or complex → request "file_tree" or "test_coverage"

Do NOT wrap your response in markdown code fences."""

# ── System prompt for review decision ─────────────────────────────────
REVIEW_SYSTEM_PROMPT = """You are a senior software engineer performing pull request code review triage.

PHASE 2: FINAL REVIEW
You have gathered context and now need to make your triage decision. Analyze everything carefully:
1. Read the diff line by line — do NOT rely solely on the PR description.
2. Cross-reference the diff with the gathered context (CI status, git blame, related PRs, etc.)
3. Evaluate reviewer comments — identify disagreements, security concerns, and correctness issues.
4. Look for subtle bugs: race conditions, security regressions, resource leaks, missing error handling.
5. The PR description may be misleading — always verify claims against the actual diff.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{
  "action_type": "review",
  "decision": "approve" | "request_changes" | "close",
  "labels": ["bug", "security", "enhancement", "documentation", "breaking-change", "needs-tests", "trivial", "urgent", "performance", "memory-leak"],
  "priority": "low" | "medium" | "high" | "critical",
  "review_summary": "1-3 sentence written review explaining your decision. Reference specific code locations, function names, and technical concerns. Mention any relevant findings from the gathered context."
}

IMPORTANT:
- labels must only contain values from the list above (pick ALL that apply)
- review_summary MUST reference specific technical details from the diff and context
- review_summary must be 1-3 sentences, 50-400 characters
- Mention specific function names, variable names, line numbers, or patterns you identified
- If you found security issues, name the specific vulnerability pattern
- If you found bugs, describe the exact failure scenario
- Do NOT wrap your response in markdown code fences

Example:
{"action_type": "review", "decision": "request_changes", "labels": ["bug", "needs-tests"], "priority": "high", "review_summary": "The INCR + EXPIRE pattern in rate_limit_middleware has a TOCTOU race condition — if the process crashes between INCR and EXPIRE, the key persists forever. This is a well-known Redis anti-pattern. Use a Lua script or MULTI/EXEC for atomic increment-with-TTL."}"""


def _format_observation(obs: dict[str, object]) -> str:
    """Format an observation dict into structured text for the LLM."""
    lines = [
        f"## Pull Request #{obs['pr_id']}: {obs['title']}",
        "",
        f"**Author:** {obs['author']}",
        f"**Base Branch:** {obs['base_branch']}",
        f"**Files Changed:** {', '.join(obs['files_changed'])}",  # type: ignore[arg-type]
        f"**Additions:** {obs['additions']} | **Deletions:** {obs['deletions']}",
        f"**Step:** {obs['current_step']}/{obs['max_steps']}",
        "",
        "### Description",
        str(obs["description"]),
        "",
        "### Diff",
        "```diff",
        str(obs["diff"]),
        "```",
        "",
        "### Reviewer Comments",
    ]
    for comment in obs["comments"]:  # type: ignore[union-attr]
        lines.append(f"- {comment}")

    # Show gathered context
    additional_context = obs.get("additional_context", {})
    if additional_context:
        lines.append("")
        lines.append("### Gathered Context")
        for ctx_name, ctx_data in additional_context.items():  # type: ignore[union-attr]
            lines.append(f"\n#### {ctx_name.replace('_', ' ').title()}")
            lines.append(str(ctx_data))

    # Show available contexts
    available = obs.get("available_contexts", [])
    if available:
        lines.append("")
        lines.append(f"### Available Contexts (not yet requested): {', '.join(available)}")  # type: ignore[arg-type]

    return "\n".join(lines)


def _parse_action(raw_text: str) -> dict[str, object] | None:
    """Parse LLM output into an action dict.

    Strips markdown code fences, handles common formatting issues.
    Returns None if parsing fails.
    """
    text = raw_text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try to extract JSON object if there's extra text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s", exc)

    return None


def _decide_contexts_to_request(obs: dict[str, object]) -> list[str]:
    """Heuristic: decide which contexts to request based on the observation.

    This is used as a fallback if the LLM fails to suggest context.
    For harder tasks, we request more context.
    """
    task = obs.get("task_name", "")
    available = obs.get("available_contexts", [])

    if task == "easy":
        return []  # No context needed for easy tasks
    elif task == "medium":
        priority = ["git_blame", "related_prs", "test_coverage"]
    elif task == "docs_api":
        priority = ["ci_status", "related_prs"]
    elif task == "hard":
        priority = ["related_prs", "ci_status", "test_coverage"]
    elif task == "async_refactor":
        priority = ["git_blame", "related_prs", "test_coverage"]
    else:
        priority = ["ci_status", "test_coverage"]

    return [ctx for ctx in priority if ctx in available]  # type: ignore[operator]


def _run_task(task_name: str) -> tuple[bool, int, float, list[float]]:
    """Run a single task episode with multi-step context gathering.

    Strategy:
        1. Reset environment
        2. Ask LLM which contexts to gather (1-3 context requests)
        3. After gathering context, ask LLM for final review
    """
    env_name = "pr-review-env"
    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

    # Reset environment
    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task": task_name},
            timeout=30,
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as exc:
        logger.error("Failed to reset environment: %s", exc)
        print(
            f"[STEP] step=0 action=null reward=0.00 "
            f"done=true error=reset_failed:{exc}"
        )
        print(f"[END] success=false steps=0 score=0.00 rewards=[]")
        return False, 0, 0.0, []

    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    headers = {"X-Session-ID": session_id}

    rewards: list[float] = []
    steps = 0
    final_score = 0.0

    # ── Phase 1: Context gathering ────────────────────────────────
    # Use LLM to decide which contexts to request, with heuristic fallback
    contexts_to_request = _decide_contexts_to_request(obs)

    # Try LLM-guided context selection first
    if contexts_to_request:
        obs_text = _format_observation(obs)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text},
                ],
                temperature=0.1,
                max_tokens=128,
            )
            raw = completion.choices[0].message.content or ""
            llm_action = _parse_action(raw)
            if llm_action and llm_action.get("context_type"):
                ctx = str(llm_action["context_type"])
                # Put LLM's choice first if it's valid
                if ctx in contexts_to_request:
                    contexts_to_request.remove(ctx)
                    contexts_to_request.insert(0, ctx)
                elif ctx in obs.get("available_contexts", []):
                    contexts_to_request.insert(0, ctx)
        except Exception as exc:
            logger.warning("LLM context selection failed, using heuristic: %s", exc)

    # Execute context requests
    for ctx_type in contexts_to_request:
        steps += 1
        ctx_action = {
            "action_type": "request_context",
            "context_type": ctx_type,
        }

        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json=ctx_action,
                headers=headers,
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            logger.error("Context request failed: %s", exc)
            print(
                f"[STEP] step={steps} action=request_context:{ctx_type} "
                f"reward=0.00 done=false error=step_error:{exc}"
            )
            continue

        reward = step_data["reward"]
        done = step_data["done"]
        rewards.append(reward)
        obs = step_data["observation"]

        print(
            f"[STEP] step={steps} action=request_context:{ctx_type} "
            f"reward={reward:.2f} done={str(done).lower()} error=none"
        )

        if done:
            # Max steps reached during context gathering
            final_score = reward
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success=false steps={steps} "
                f"score={final_score:.2f} rewards=[{rewards_str}]"
            )
            return False, steps, final_score, rewards

    # ── Phase 2: Final review ─────────────────────────────────────
    for attempt in range(1, MAX_STEPS_PER_TASK - steps + 1):
        steps += 1
        obs_text = _format_observation(obs)

        # Call LLM for final review
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw_output = completion.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM call failed at step %d: %s", steps, exc)
            print(
                f"[STEP] step={steps} action=null reward=0.00 "
                f"done=false error=llm_error:{exc}"
            )
            continue

        # Parse action
        action_dict = _parse_action(raw_output)
        if action_dict is None:
            logger.error(
                "Failed to parse LLM output at step %d: %.200s",
                steps, raw_output,
            )
            print(
                f"[STEP] step={steps} action=null reward=0.00 "
                f"done=false error=parse_error"
            )
            continue

        # Ensure it's a review action
        action_dict["action_type"] = "review"

        # Submit action to environment
        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action_dict,
                headers=headers,
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            logger.error("Step API call failed: %s", exc)
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:  # type: ignore[union-attr]
                try:
                    error_detail = exc.response.text  # type: ignore[union-attr]
                except Exception:
                    pass
            print(
                f"[STEP] step={steps} action={json.dumps(action_dict)} "
                f"reward=0.00 done=false error=step_error:{error_detail or exc}"
            )
            continue

        reward = step_data["reward"]
        done = step_data["done"]
        rewards.append(reward)
        obs = step_data["observation"]

        action_summary = action_dict.get("decision", "unknown")
        print(
            f"[STEP] step={steps} action={action_summary} "
            f"reward={reward:.2f} done={str(done).lower()} error=none"
        )

        if done:
            final_score = reward
            break

    success = steps > 0 and len(rewards) > 0 and any(r > 0 for r in rewards)
    if rewards:
        final_score = rewards[-1]

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={final_score:.2f} rewards=[{rewards_str}]"
    )

    return success, steps, final_score, rewards


def main() -> None:
    """Run inference on all 5 tasks sequentially."""
    if not HF_TOKEN:
        logger.warning(
            "HF_TOKEN not set. LLM calls will likely fail. "
            "Set it via: export HF_TOKEN=your_token"
        )

    print(f"{'='*60}")
    print("PR Review Triage — Baseline Inference (Multi-Step)")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print(f"Env:   {ENV_BASE_URL}")
    print(f"{'='*60}")
    print()

    results: list[dict[str, object]] = []

    for task in TASKS:
        success, steps, score, rewards = _run_task(task)
        results.append({
            "task": task,
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        })
        print()

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  {str(r['task']):18s} — score={r['score']:.2f}, "
            f"steps={r['steps']}, success={r['success']}"
        )

    avg_score = (
        sum(r["score"] for r in results) / len(results)  # type: ignore[arg-type]
        if results
        else 0.0
    )
    print(f"\n  Average score: {avg_score:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
