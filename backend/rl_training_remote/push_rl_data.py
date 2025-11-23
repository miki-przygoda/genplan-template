#!/usr/bin/env python3
"""
Small helper to commit and push freshly generated RL training data.

By default it stages the bandit state and episode log, commits with a standard
message, and pushes to the current branch's origin.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TARGETS = [
    PROJECT_ROOT / "backend" / "data" / "rl" / "seed_bandit.json",
    PROJECT_ROOT / "backend" / "data" / "rl" / "episode_log.jsonl",
]


def run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True)


def current_branch(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit and push RL training data artifacts.")
    parser.add_argument(
        "--message",
        "-m",
        default="chore: update rl training data",
        help="Commit message to use.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to origin (still commits locally).",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional files to stage (paths relative to repo root).",
    )
    parser.add_argument(
        "--pull-rebase",
        action="store_true",
        help="Auto-run 'git pull --rebase' before pushing to avoid non-fast-forward errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    targets = [path for path in DEFAULT_TARGETS if path.exists()]
    for extra in args.extra:
        targets.append((PROJECT_ROOT / extra).resolve())

    if not targets:
        print("No RL data files found to commit. Expected defaults are missing.", file=sys.stderr)
        sys.exit(1)

    # Convert to repo-relative paths for nicer git output
    rel_targets = [str(p.relative_to(PROJECT_ROOT)) for p in targets]
    print(f"Staging files: {', '.join(rel_targets)}")
    run(["git", "add", "--"] + rel_targets, cwd=PROJECT_ROOT)

    print(f"Committing with message: {args.message!r}")
    run(["git", "commit", "-m", args.message], cwd=PROJECT_ROOT)

    if not args.no_push:
        branch = current_branch(PROJECT_ROOT)
        if args.pull_rebase:
            print(f"Pulling latest with rebase on {branch} ...")
            run(["git", "pull", "--rebase", "origin", branch], cwd=PROJECT_ROOT)
        print(f"Pushing to origin/{branch} ...")
        run(["git", "push", "origin", branch], cwd=PROJECT_ROOT)
    else:
        print("Push skipped (--no-push).")


if __name__ == "__main__":
    main()
