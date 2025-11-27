#!/usr/bin/env python3
"""
Remote-friendly runner that mirrors the EA_Eval_Compare notebook.

For each requested run it:
 - picks a random floor from the fixed set,
 - executes the EA once with the RL seeder and once with a manual seeder (sequentially),
 - logs both histories/best scores to backend/data/ea-logs/json.

Runs are farmed out to a small process pool (default: 6 workers) to speed up batches,
but the two EA variants for a given floor always execute back-to-back in the same worker.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_MAX_WORKERS = 6
DEFAULT_RUNS = 3

# Fixed floors aligned with RL training / notebook defaults
FIXED_FLOOR_IDS = [
    15,
    35,
    55,
    75,
    95,
    101,
    110,
    120,
    135,
    150,
    160,
    175,
    185,
    190,
    205,
    210,
    230,
    235,
    245,
    260,
    270,
    285,
    295,
    309,
    320,
    340,
    345,
    365,
    370,
    390,
    395,
    412,
    420,
    440,
    445,
    465,
    470,
    490,
    495,
    515,
    523,
    540,
    550,
    565,
    575,
    590,
    600,
    615,
    634,
    640,
    660,
    665,
    685,
    690,
    710,
    715,
    740,
    745,
    765,
    770,
    790,
    795,
    815,
    820,
    840,
    845,
    856,
    865,
    880,
    890,
    905,
    915,
    930,
    940,
    955,
    960,
    967,
]


def find_project_root() -> Path:
    """Walk upwards to locate the repo root that contains backend/src."""
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "backend" / "src").exists():
            return candidate
    raise RuntimeError("Could not locate backend/src. Run from inside the repo.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for EA_Eval_Compare (RL vs manual seeder, sequential per floor)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="How many random floors to evaluate (each runs RL + manual sequentially).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Process workers for parallel runs (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible floor selection and EA seeds.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where to write EA JSON logs (default: backend/data/ea-logs/json).",
    )
    parser.add_argument(
        "--fixed-floors",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for the floor ID pool (defaults to notebook list).",
    )
    return parser.parse_args(argv)


def _config_summary(cfg) -> Dict[str, Any]:
    """Subset of EAConfig fields plus weights, mirroring the notebook."""
    summary_fields = [
        "population_size",
        "generations",
        "crossover_rate",
        "mutation_rate",
        "tournament_k",
        "elite_fraction",
        "random_seed",
        "stagnation_threshold",
        "restart_fraction",
        "mutation_boost",
        "mutation_floor",
        "mutation_ceiling",
        "no_change_penalty",
    ]
    weight_fields = [
        "quadrant",
        "overlap",
        "area",
        "compactness",
        "adjacency",
        "location",
        "section",
        "dispersion",
        "room_usage",
        "budget",
        "section_bbox",
        "mask",
        "relationships",
        "realism",
    ]
    summary = {field: getattr(cfg, field, None) for field in summary_fields}
    summary["weights"] = {field: getattr(getattr(cfg, "weights", None), field, None) for field in weight_fields}
    return summary


def _save_run_log(
    *,
    log_dir: Path,
    run_label: str,
    floor_id: int,
    grid_size: int,
    rotate_k: int,
    seed_name_rl: str,
    manual_seed_name: str,
    hist_rl: Iterable[float],
    hist_manual: Iterable[float],
    best_rl: float,
    best_manual: float,
    cfg,
) -> Path:
    now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{now}-{run_label}"
    payload = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "floor_id": floor_id,
        "grid_size": grid_size,
        "rotate_k": rotate_k,
        "seeders": {"rl": seed_name_rl, "manual": manual_seed_name},
        "history": {"rl": list(hist_rl), "manual": list(hist_manual)},
        "best_fitness": {"rl": best_rl, "manual": best_manual},
        "config": _config_summary(cfg),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ea_run_{run_id}_floor{floor_id:03d}.json"
    log_path.write_text(json.dumps(payload, indent=2))
    return log_path


def _run_ea(sample, cfg, seed_fn, *, cfg_seed_offset: int = 0) -> Tuple[List[float], float]:
    """Execute the EA for one seeding strategy and collect per-generation best fitness."""
    rng = random.Random(cfg.random_seed + cfg_seed_offset)

    from ver0.evolver import evaluate_population, init_population, make_next_generation, mutate  # type: ignore

    pop = init_population(sample, cfg, rng, seed_fn)
    evaluate_population(sample, pop, cfg)
    history: List[float] = []

    for gen in range(cfg.generations):
        if gen > 0:
            pop = make_next_generation(sample, pop, cfg, rng, seed_fn, mutate)
            evaluate_population(sample, pop, cfg)
        best = min(pop, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
        history.append(best.fitness)

    best_final = min(pop, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
    return history, best_final.fitness


def _build_config(rng_seed: int):
    """Create EAConfig mirroring EA_Eval_Compare defaults."""
    from ver0.evolver import EAConfig  # type: ignore
    from ver0.fitness import Weights  # type: ignore
    from ver0.vars import (  # type: ignore
        ADJACENCY_WEIGHT,
        AREA_WEIGHT,
        BUDGET_WEIGHT,
        COMPACTNESS_WEIGHT,
        CROSSOVER_RATE,
        DEFAULT_GRID_SIZE,
        DISPERSION_WEIGHT,
        ELITE_FRACTION,
        LOCATION_WEIGHT,
        MASK_WEIGHT,
        MUTATION_RATE,
        NO_CHANGE_PENALTY,
        OVERLAP_WEIGHT,
        QUADRANT_WEIGHT,
        RANDOM_SEED,
        REALISM_WEIGHT,
        RELATIONSHIP_WEIGHT,
        ROTATE_IMAGE_K,
        SECTION_BBOX_WEIGHT,
        SECTION_WEIGHT,
        ROOM_USAGE_WEIGHT,
        TOURNAMENT_K,
    )

    weights = Weights(
        quadrant=QUADRANT_WEIGHT,
        overlap=OVERLAP_WEIGHT,
        area=AREA_WEIGHT,
        compactness=COMPACTNESS_WEIGHT,
        adjacency=ADJACENCY_WEIGHT,
        location=LOCATION_WEIGHT,
        section=SECTION_WEIGHT,
        dispersion=DISPERSION_WEIGHT,
        room_usage=ROOM_USAGE_WEIGHT,
        budget=BUDGET_WEIGHT,
        section_bbox=SECTION_BBOX_WEIGHT,
        mask=MASK_WEIGHT,
        relationships=RELATIONSHIP_WEIGHT,
        realism=REALISM_WEIGHT,
    )

    return EAConfig(
        population_size=52,  # slightly trimmed for remote speed vs POPULATION_SIZE
        generations=100,
        crossover_rate=0.7 if CROSSOVER_RATE is None else CROSSOVER_RATE,
        mutation_rate=0.25 if MUTATION_RATE is None else MUTATION_RATE,
        tournament_k=3 if TOURNAMENT_K is None else TOURNAMENT_K,
        elite_fraction=0.08 if ELITE_FRACTION is None else ELITE_FRACTION,
        random_seed=rng_seed or RANDOM_SEED,
        weights=weights,
        stagnation_threshold=20,
        restart_fraction=0.30,
        mutation_boost=1.5,
        mutation_floor=0.05,
        mutation_ceiling=0.65,
        no_change_penalty=NO_CHANGE_PENALTY,
    )


def _run_single(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint: run RL + manual EAs back-to-back for one floor."""
    project_root = Path(job["project_root"])
    sys.path.insert(0, str(project_root / "backend" / "src"))

    from ver0.grid_encoder import encode_floorplan_to_grid  # type: ignore
    from ver0.rl_bandit import make_seed_bandit  # type: ignore
    from ver0.seeders import SEEDING_REGISTRY  # type: ignore
    from ver0.vars import DEFAULT_GRID_SIZE, ROTATE_IMAGE_K  # type: ignore

    run_idx = job["run_idx"]
    floor_id = job["floor_id"]
    rng_seed = job["rng_seed"]
    log_dir = Path(job["log_dir"])

    grid_size = job.get("grid_size") or DEFAULT_GRID_SIZE
    rotate_k = job.get("rotate_k")
    if rotate_k is None:
        rotate_k = ROTATE_IMAGE_K

    cfg = _build_config(rng_seed)

    floor_dir = project_root / "backend" / "data" / "processed" / "floor_plans" / f"floor{floor_id:03d}"
    sample = encode_floorplan_to_grid(floor_dir, grid_size=grid_size, rotate_k=rotate_k)

    bandit = make_seed_bandit(project_root / "backend" / "data" / "rl" / "seed_bandit.json", epsilon=0.05, rng=random.Random(rng_seed))
    seed_name_rl, seed_fn_rl = bandit.select()
    manual_seed_name = next(iter(SEEDING_REGISTRY.keys()))
    seed_fn_manual = SEEDING_REGISTRY[manual_seed_name]

    hist_rl, best_rl = _run_ea(sample, cfg, seed_fn_rl, cfg_seed_offset=0)
    hist_manual, best_manual = _run_ea(sample, cfg, seed_fn_manual, cfg_seed_offset=1234)

    log_path = _save_run_log(
        log_dir=log_dir,
        run_label=f"r{run_idx:03d}",
        floor_id=floor_id,
        grid_size=grid_size,
        rotate_k=rotate_k,
        seed_name_rl=seed_name_rl,
        manual_seed_name=manual_seed_name,
        hist_rl=hist_rl,
        hist_manual=hist_manual,
        best_rl=best_rl,
        best_manual=best_manual,
        cfg=cfg,
    )

    return {
        "run_idx": run_idx,
        "floor_id": floor_id,
        "log_path": str(log_path),
        "seeders": {"rl": seed_name_rl, "manual": manual_seed_name},
        "best_fitness": {"rl": best_rl, "manual": best_manual},
    }


def build_jobs(
    *,
    runs: int,
    floor_pool: Sequence[int],
    rng: random.Random,
    log_dir: Path,
    project_root: Path,
    grid_size: int | None,
    rotate_k: int | None,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for idx in range(runs):
        floor_id = rng.choice(floor_pool)
        rng_seed = rng.randint(0, 1_000_000_000)
        jobs.append(
            {
                "run_idx": idx,
                "floor_id": floor_id,
                "rng_seed": rng_seed,
                "log_dir": str(log_dir),
                "project_root": str(project_root),
                "grid_size": grid_size,
                "rotate_k": rotate_k,
            }
        )
    return jobs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = find_project_root()

    log_dir = args.log_dir or (project_root / "backend" / "data" / "ea-logs" / "json")
    floor_pool = args.fixed_floors if args.fixed_floors else FIXED_FLOOR_IDS
    runs = max(1, args.runs)
    max_workers = max(1, args.max_workers or DEFAULT_MAX_WORKERS)
    rng = random.Random(args.seed or int(time.time()))

    jobs = build_jobs(
        runs=runs,
        floor_pool=floor_pool,
        rng=rng,
        log_dir=log_dir,
        project_root=project_root,
        grid_size=None,
        rotate_k=None,
    )

    print(
        f"Starting {runs} run(s) with {max_workers} worker(s). "
        f"Floor pool size: {len(floor_pool)}. Logs -> {log_dir}"
    )

    start = time.time()
    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_run_single, job): job for job in jobs}
        for fut in concurrent.futures.as_completed(future_map):
            job = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
                print(
                    f"[run {res['run_idx']:03d}] floor {res['floor_id']:03d} "
                    f"RL {res['best_fitness']['rl']:.3f} | manual {res['best_fitness']['manual']:.3f} "
                    f"-> {res['log_path']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"run {job['run_idx']:03d} floor {job['floor_id']:03d}: {exc!r}")
                print(f"Run failed: {failures[-1]}")

    elapsed = time.time() - start
    print(f"Finished {len(results)} run(s) with {len(failures)} failure(s) in {elapsed/60:.1f} min.")
    if failures:
        for msg in failures:
            print(f" - {msg}")


if __name__ == "__main__":
    main()
