#!/usr/bin/env python3
"""
Compatibility shim: forwards to backend.rl_training_remote.cli
so you can still run this file directly.
"""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.rl_training_remote.cli import main


if __name__ == "__main__":
    main()
