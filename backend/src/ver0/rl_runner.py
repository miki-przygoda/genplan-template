from __future__ import annotations

"""
Helper to run one EA episode in a separate process. Designed to be fork-safe
for use with ProcessPoolExecutor from notebooks.
"""

import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple, Any

from .evolver import evolve, EAConfig, mutate
from .mutator import set_light_mutation
from .seeders import SEEDING_REGISTRY
from .grid_encoder import encode_floorplan_to_grid
from .fitness import Weights
import time


def run_episode(job: Tuple[int, int, str, Dict[str, Any], str, int, int, bool | None]) -> Dict[str, Any]:
    """
    Execute one EA episode.

    Args:
        job: (ep_idx, floor_id, seed_name, cfg_dict, project_root_str, grid_size, rotate_k)
    Returns:
        dict with episode, floor_id, seed, best_fitness, history
    """
    if len(job) == 7:
        ep_idx, floor_id, seed_name, cfg_dict, project_root_str, grid_size, rotate_k = job
        light_mutation = False
    else:
        ep_idx, floor_id, seed_name, cfg_dict, project_root_str, grid_size, rotate_k, light_mutation = job
    project_root = Path(project_root_str)

    # Rehydrate config and weights
    weights_dict = cfg_dict.pop("weights", {})
    weights = Weights(**weights_dict)
    cfg = EAConfig(**cfg_dict, weights=weights)

    # Unique seed per episode for reproducibility
    rng_seed = (cfg.random_seed or 0) + ep_idx * 131 + floor_id
    cfg = replace(cfg, random_seed=rng_seed)

    floor_dir = project_root / "backend" / "data" / "processed" / "floor_plans" / f"floor{floor_id:03d}"
    sample = encode_floorplan_to_grid(floor_dir, grid_size=grid_size, rotate_k=rotate_k)

    seed_fn = SEEDING_REGISTRY[seed_name]
    set_light_mutation(bool(light_mutation))
    start = time.perf_counter()
    best, _, history = evolve(sample, cfg=cfg, make_random=seed_fn, mutate_fn=mutate)
    duration = time.perf_counter() - start
    best_f = best.fitness if best.fitness is not None else float("inf")

    return {
        "episode": ep_idx + 1,
        "floor_id": floor_id,
        "seed": seed_name,
        "best_fitness": best_f,
        "history": history,
        "duration_s": duration,
    }
