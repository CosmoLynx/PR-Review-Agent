"""Deploy the PR Review Triage environment to Hugging Face Spaces.

Requires:
    HF_TOKEN     — Hugging Face token with write access
    HF_SPACE_ID  — Target space (default: hitanshjain1812/pr-review-env)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main() -> None:
    load_dotenv()
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it to .env or environment variables.")

    space_id = os.environ.get("HF_SPACE_ID", "abhay1304/pr-review-env").strip()
    repo_root = Path(__file__).resolve().parent

    api = HfApi(token=token)

    # Create the Space (Docker SDK, since we have a Dockerfile)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    # Upload everything except sensitive/unnecessary files
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(repo_root),
        ignore_patterns=[
            ".git/*",
            "__pycache__/*",
            "*.pyc",
            ".env",
            ".venv/*",
            "venv/*",
            ".pytest_cache/*",
            "inference_output.txt",
        ],
    )

    print(f"\nDeployed successfully!")
    print(f"URL: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
