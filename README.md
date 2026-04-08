# PR Review Triage Environment (`pr-review-env`)

> A multi-step OpenEnv environment that simulates the real-world workflow every engineering team does daily: reading pull request diffs, gathering additional context, evaluating reviewer comments, identifying security regressions and concurrency bugs, making triage decisions, and writing review summaries. Unlike single-turn code review tasks, this environment rewards agents that **gather relevant context before deciding** — just like a real senior engineer would check git blame, CI results, and related PRs before signing off on a change.

---

## What Makes This Environment Unique

1. **Multi-step interaction model**: Agents don't just see the diff — they can request 5 types of additional context (file tree, CI status, git blame, related PRs, test coverage) before making their final decision. This rewards strategic information gathering.

2. **Hidden critical context**: Each task has context data that dramatically changes the correct assessment. For example, the medium task's git blame reveals the "dead code" being removed was added for a P0 security incident. An agent that skips context gathering will miss this.

3. **6-axis dense reward**: Scores decision accuracy, label F1, priority proximity, summary quality, context gathering efficiency, and reasoning depth — providing rich gradient signal.

4. **5 difficulty levels**: Easy → Medium → Medium-Hard → Hard → Expert, calibrated so frontier models score 0.85+ on easy but <0.45 on expert.

5. **Adversarial design**: PR descriptions are intentionally misleading on harder tasks. The diff tells one story; the description tells another. Only agents that read code (not just text) succeed.

---

## Observation Space

| Field               | Type             | Description                                                  |
|---------------------|------------------|--------------------------------------------------------------|
| `pr_id`             | `int`            | Pull request identifier                                      |
| `title`             | `str`            | PR title (may be misleading on harder tasks)                 |
| `description`       | `str`            | PR description / body text                                   |
| `diff`              | `str`            | Unified diff with `@@` headers (30–100+ lines)               |
| `comments`          | `list[str]`      | 2–6 inline reviewer comments                                |
| `files_changed`     | `list[str]`      | List of changed file paths                                   |
| `author`            | `str`            | PR author username                                           |
| `base_branch`       | `str`            | Target branch                                                |
| `additions`         | `int`            | Lines added                                                  |
| `deletions`         | `int`            | Lines deleted                                                |
| `current_step`      | `int`            | Current step in the episode                                  |
| `max_steps`         | `int`            | Maximum allowed steps                                        |
| `task_name`         | `str`            | Task identifier                                              |
| `additional_context`| `dict[str, str]` | Accumulated context from `request_context` actions           |
| `available_contexts`| `list[str]`      | Context types still available to request                     |

## Action Space

Actions use a **discriminated union** pattern with `action_type` as the discriminator:

### Context Request Action (`action_type: "request_context"`)

| Field          | Type     | Description                          |
|----------------|----------|--------------------------------------|
| `action_type`  | `str`    | Must be `"request_context"`          |
| `context_type` | `str`    | One of: `file_tree`, `ci_status`, `git_blame`, `related_prs`, `test_coverage` |

### Review Action (`action_type: "review"`)

| Field            | Type         | Description                        |
|------------------|--------------|------------------------------------|
| `action_type`    | `str`        | Must be `"review"`                 |
| `decision`       | `str`        | `"approve"`, `"request_changes"`, or `"close"` |
| `labels`         | `list[str]`  | From: `{bug, security, enhancement, documentation, breaking-change, needs-tests, trivial, urgent, performance, memory-leak}` |
| `priority`       | `str`        | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `review_summary` | `str`        | 1–3 sentence written review        |

### Available Context Types

| Context Type    | What It Reveals                                                       |
|-----------------|-----------------------------------------------------------------------|
| `file_tree`     | Directory structure of affected areas                                 |
| `ci_status`     | CI pipeline results, test counts, coverage changes, warnings          |
| `git_blame`     | Authorship, commit history, and context for why code was written      |
| `related_prs`   | Previous PRs related to this change (may reveal rejected approaches)  |
| `test_coverage` | Detailed coverage report with missing coverage analysis               |

## Reward Function

The reward is **dense** (not sparse), computed along **6 independent weighted axes**:

| Axis                     | Weight | Scoring Method                                                                        |
|--------------------------|--------|---------------------------------------------------------------------------------------|
| `decision_score`         | 0.25   | Binary: 1.0 if exact match with gold decision, 0.0 otherwise                         |
| `label_score`            | 0.20   | F1 score between predicted and gold label sets (partial credit)                       |
| `priority_score`         | 0.15   | Ordinal distance: exact = 1.0, off-by-1 = 0.5, off-by-2 = 0.25, off-by-3 = 0.0      |
| `summary_score`          | 0.15   | Keyword coverage against gold keywords; length penalties (<20 or >500 chars)          |
| `context_efficiency`     | 0.15   | F1 between requested and relevant contexts (full marks if no context needed)          |
| `reasoning_depth`        | 0.10   | Fraction of technical depth keywords mentioned in the summary                         |

**Step penalty:** −0.01 per step beyond step 1 (reduced from 0.02 to accommodate multi-step context gathering).

**Key design decisions:**
- Context gathering costs steps but can dramatically improve decision quality
- `context_efficiency` rewards agents that request the RIGHT contexts, not just ALL contexts
- For easy tasks, `context_efficiency` = 1.0 regardless (no context needed)
- `reasoning_depth` rewards mentioning specific function names, vulnerability patterns, etc.
- Fully deterministic — no LLM calls in the grader

## Tasks

| Task             | Difficulty  | Scenario                                                                                         | What Makes It Hard                                                                     |
|------------------|-------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `easy`           | Easy        | 12-line bugfix PR fixing off-by-one in list slicing. Clean diff, unanimous approval.             | Nothing — baseline test. GPT-4o should score 0.85+.                                    |
| `medium`         | Medium      | Auth middleware refactor silently removes token expiry checks. Misleading description.           | Must read diff, not just description. Git blame reveals P0 security incident context.  |
| `docs_api`       | Medium-Hard | API doc update with subtly incorrect code examples using v1 methods for v2 API.                  | CI doc-tests fail. Examples throw AttributeError. Requires understanding SDK migration. |
| `hard`           | Hard        | Rate limiter with TOCTOU race in Redis INCR+EXPIRE. 4 conflicting reviewers, author pushback.   | Requires concurrency knowledge. Related PRs reveal same approach was previously rejected. |
| `async_refactor` | Expert      | Async DB refactor drops rollback-on-error, poisoning connection pool under sustained failures.   | Subtle resource leak invisible in happy path. Previous production incident with same root cause. |

## Quick Start

### Build and run with Docker

```bash
# Build the image
docker build -t pr-review-env .

# Run the server
docker run -p 7860:7860 pr-review-env

# Verify it's running
curl http://localhost:7860/health
# → {"status":"ok"}

# List all 5 tasks
curl http://localhost:7860/tasks | python -m json.tool
```

### Multi-step interaction example

```bash
# 1. Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "medium"}'
# → returns session_id and initial observation

# 2. Gather context (optional but recommended for harder tasks)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <session_id>" \
  -d '{"action_type": "request_context", "context_type": "git_blame"}'
# → returns observation with git_blame in additional_context

# 3. Gather more context
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <session_id>" \
  -d '{"action_type": "request_context", "context_type": "related_prs"}'

# 4. Submit final review
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <session_id>" \
  -d '{
    "action_type": "review",
    "decision": "request_changes",
    "labels": ["security", "breaking-change"],
    "priority": "critical",
    "review_summary": "This PR removes the token expiry check that was added for SEC-2024-017. The _validate_token_signature method only checks the signature, not exp. Expired tokens will be accepted forever — this is a security regression, not a cleanup."
  }'
# → returns reward breakdown with dense scoring
```

### Run without Docker

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Running Inference

```bash
# Set your Hugging Face token
export HF_TOKEN=hf_your_token_here

# Optional: customize model and API
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

# Run inference (all 5 tasks with multi-step context gathering)
python inference.py
```

The inference script uses a two-phase strategy:
1. **Context gathering**: Requests relevant context based on task heuristics + LLM guidance
2. **Final review**: Submits triage decision with all gathered context visible

Output format:
```
[START] task=medium env=pr-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=request_context:git_blame reward=0.00 done=false error=none
[STEP] step=2 action=request_context:related_prs reward=0.00 done=false error=none
[STEP] step=3 action=request_context:test_coverage reward=0.00 done=false error=none
[STEP] step=4 action=request_changes reward=0.72 done=true error=none
[END] success=true steps=4 score=0.72 rewards=[0.00,0.00,0.00,0.72]
```

## Baseline Scores

| Task             | Model                      | Score | Steps | Strategy                        |
|------------------|----------------------------|-------|-------|---------------------------------|
| `easy`           | Qwen/Qwen2.5-72B-Instruct  | 0.88  | 1     | Direct review (no context)      |
| `medium`         | Qwen/Qwen2.5-72B-Instruct  | 0.65  | 4     | 3 context requests + review     |
| `docs_api`       | Qwen/Qwen2.5-72B-Instruct  | 0.58  | 3     | 2 context requests + review     |
| `hard`           | Qwen/Qwen2.5-72B-Instruct  | 0.42  | 4     | 3 context requests + review     |
| `async_refactor` | Qwen/Qwen2.5-72B-Instruct  | 0.35  | 4     | 3 context requests + review     |

These are estimated scores. Actual performance depends on model capability and prompt sensitivity.

## HF Space Deployment

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with the **Docker** SDK.
2. Set the `HF_TOKEN` secret in Space settings (needed for inference).
3. Push the code:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/pr-review-env
cd pr-review-env
cp -r /path/to/pr-review-env/* .
git add .
git commit -m "Deploy PR Review Triage Environment v2.0"
git push
```

## API Reference

| Method | Endpoint   | Body                                                     | Response                           |
|--------|------------|----------------------------------------------------------|------------------------------------|
| POST   | `/reset`   | `{"task": "<task_name>"}`                               | `{session_id, observation}`        |
| POST   | `/step`    | Context: `{action_type, context_type}` or Review: `{action_type, decision, labels, priority, review_summary}` | `{observation, reward, done, info}` |
| GET    | `/state`   | —                                                        | Current env state dict             |
| GET    | `/tasks`   | —                                                        | List of task metadata              |
| GET    | `/health`  | —                                                        | `{"status": "ok"}`                 |

**Session management:** Pass `X-Session-ID` header to maintain state across calls. Generated on `/reset` if not provided.

## Architecture

```
pr-review-env/
├── app.py                        # FastAPI server with session management
├── inference.py                  # Multi-step baseline LLM inference
├── pr_review_env/
│   ├── models.py                 # Pydantic v2 data models (Action discriminated union)
│   ├── env.py                    # PRReviewEnv — multi-step episode lifecycle
│   ├── reward.py                 # Dense 6-axis reward function
│   └── tasks/                    # Task definitions + gold standards
│       ├── easy.py               # Off-by-one bugfix
│       ├── medium.py             # Security regression in auth middleware
│       ├── docs_api.py           # Incorrect API documentation examples
│       ├── hard.py               # TOCTOU race condition in rate limiter
│       └── async_refactor.py     # Connection pool poisoning in async refactor
└── fixtures/                     # Realistic PR data with hidden context
    ├── pr_easy.json
    ├── pr_medium.json
    ├── pr_docs_api.json
    ├── pr_hard.json
    └── pr_async_refactor.json
```

## Design Philosophy

This environment is grounded in the daily reality of code review at scale:

- **Information asymmetry**: PR descriptions often understate or misrepresent changes. Agents must verify claims against actual diffs.
- **Context matters**: Real reviewers check git blame to understand why code was written a certain way. They look at CI results. They search for related PRs. This environment rewards that behavior.
- **Disagreement is signal**: When reviewers disagree, something subtle is happening. The harder tasks have conflicting reviewer opinions that test whether the agent can evaluate arguments on both sides.
- **Dense feedback**: Binary pass/fail rewards don't help agents improve. Our 6-axis reward provides rich gradient signal across decision quality, labeling accuracy, priority assessment, summary depth, context efficiency, and technical reasoning.

## License

MIT
