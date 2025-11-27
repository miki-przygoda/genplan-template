#!/usr/bin/env python3
"""
Utility to wipe EA evaluation logs.

Removes JSON log files in backend/data/ea-logs/json (used by EA_Eval_Compare).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "backend" / "data" / "ea-logs" / "json"


def main() -> None:
    if not LOG_DIR.exists():
        print(f"No EA log directory found: {LOG_DIR.relative_to(PROJECT_ROOT)}")
        sys.exit(0)

    removed = []
    for fpath in sorted(LOG_DIR.glob("*.json")):
        try:
            fpath.unlink()
            removed.append(fpath)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not remove {fpath}: {exc}", file=sys.stderr)

    if removed:
        print("Removed EA log files:")
        for fpath in removed:
            print(f" - {fpath.relative_to(PROJECT_ROOT)}")
    else:
        print("No EA log files to remove.")

    # If directory is empty after deletion, clean it up
    try:
        if LOG_DIR.exists() and not any(LOG_DIR.iterdir()):
            LOG_DIR.rmdir()
            print(f"Removed empty directory: {LOG_DIR.relative_to(PROJECT_ROOT)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
