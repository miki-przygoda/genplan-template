#!/usr/bin/env python3
"""
Utility to wipe RL bandit memory/logs after major logic changes.

Removes:
 - backend/data/rl/seed_bandit.json
 - backend/data/rl/episode_log.jsonl
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    PROJECT_ROOT / "backend" / "data" / "rl" / "seed_bandit.json",
    PROJECT_ROOT / "backend" / "data" / "rl" / "episode_log.jsonl",
]


def main() -> None:
    removed = []
    missing = []
    for path in TARGETS:
        if path.exists():
            path.unlink()
            removed.append(path)
        else:
            missing.append(path)

    if removed:
        print("Removed:")
        for p in removed:
            print(f" - {p.relative_to(PROJECT_ROOT)}")
    else:
        print("No RL memory files found to remove.")

    if missing and removed:
        print("Missing (already absent):")
        for p in missing:
            print(f" - {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
