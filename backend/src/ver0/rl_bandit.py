from __future__ import annotations

"""
Lightweight epsilon-greedy bandit for seeding strategy selection.
State is persisted to JSON so repeated runs can improve choices.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Optional

from .seeders import SEEDING_REGISTRY


SeederFn = Callable[[object, random.Random], object]


@dataclass
class ArmStats:
    pulls: int = 0
    value: float = 0.0


class SeedBandit:
    def __init__(
        self,
        *,
        state_path: Path,
        epsilon: float = 0.15,
        actions: Optional[Dict[str, SeederFn]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.state_path = state_path
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.actions = actions or SEEDING_REGISTRY
        self.arms: Dict[str, ArmStats] = {name: ArmStats() for name in self.actions}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            epsilon = data.get("epsilon")
            if isinstance(epsilon, (int, float)):
                self.epsilon = float(epsilon)
            arms = data.get("arms", {})
            for name, stats in arms.items():
                if name in self.arms and isinstance(stats, dict):
                    self.arms[name] = ArmStats(
                        pulls=int(stats.get("pulls", 0)),
                        value=float(stats.get("value", 0.0)),
                    )
        except Exception:
            # Stay silent; fall back to defaults
            pass

    def save(self) -> None:
        payload = {
            "epsilon": self.epsilon,
            "arms": {name: asdict(stats) for name, stats in self.arms.items()},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2))

    def select(self) -> tuple[str, SeederFn]:
        if self.rng.random() < self.epsilon:
            name = self.rng.choice(list(self.actions.keys()))
            return name, self.actions[name]
        # exploit: pick max value (tie-break randomly)
        max_val = max(self.arms[name].value for name in self.actions)
        best = [name for name in self.actions if self.arms[name].value == max_val]
        name = self.rng.choice(best)
        return name, self.actions[name]

    def update(self, name: str, reward: float) -> None:
        if name not in self.arms:
            return
        stats = self.arms[name]
        stats.pulls += 1
        # incremental mean
        stats.value += (reward - stats.value) / max(1, stats.pulls)
        self.arms[name] = stats
        self.save()


def make_seed_bandit(state_path: Path, epsilon: float = 0.15, rng: Optional[random.Random] = None) -> SeedBandit:
    return SeedBandit(state_path=state_path, epsilon=epsilon, rng=rng)
