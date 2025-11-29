#!/usr/bin/env python3
"""
Small helper to commit and push EA evaluation logs.

Targets the JSON outputs written by train_remote_ea_eval.py into backend/data/ea-logs/json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EA_LOG_DIR = PROJECT_ROOT / "backend" / "data" / "ea-logs" / "json"


def run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True)


def current_branch(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def has_changes(cwd: Path, rel_target: str) -> bool:
    """Return True if the target directory has modified or untracked files."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", rel_target], cwd=cwd, capture_output=True, text=True
    )
    return bool(status.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit and push EA evaluation logs.")
    parser.add_argument(
        "--message",
        "-m",
        default="update ea-eval logs",
        help="Commit message to use.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to origin (still commits locally).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not EA_LOG_DIR.exists():
        print(f"Log directory not found: {EA_LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    rel_target = str(EA_LOG_DIR.relative_to(PROJECT_ROOT))

    if not has_changes(PROJECT_ROOT, rel_target):
        print("No EA log changes to commit. Exiting.")
        sys.exit(0)

    branch = current_branch(PROJECT_ROOT)
    print(f"Pulling latest with rebase on {branch} ...")
    try:
        run(["git", "pull", "--rebase", "--autostash", "origin", branch], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"git pull --rebase failed: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)

    print(f"Staging logs in: {rel_target}")
    run(["git", "add", "--", rel_target], cwd=PROJECT_ROOT)

    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
    if status.returncode == 0:
        print("No changes staged after pull; nothing to commit.")
        sys.exit(0)

    print(f"Committing with message: {args.message!r}")
    run(["git", "commit", "-m", args.message], cwd=PROJECT_ROOT)

    if not args.no_push:
        print(f"Pushing to origin/{branch} ...")
        run(["git", "push", "origin", branch], cwd=PROJECT_ROOT)
    else:
        print("Push skipped (--no-push).")


if __name__ == "__main__":
    main()
