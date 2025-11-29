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
import multiprocessing
import queue
import copy
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_MAX_WORKERS = 6
DEFAULT_RUNS = 6

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
]


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    mins, sec = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"


def find_project_root() -> Path:
    """Walk upwards to locate the repo root that contains backend/src."""
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "backend" / "src").exists():
            return candidate
    raise RuntimeError("Could not locate backend/src. Run from inside the repo.")


class TerminalUI:
    """Single-line stdout refresher for live status (minimal flicker)."""

    def __init__(self, refresh_s: float = 0.5, enabled: bool = True) -> None:
        self.refresh_s = refresh_s
        self.enabled = enabled
        self._last_print = 0.0
        self._last_len = 0
        self._header_printed = False
        self._last_line = ""

    def print_header(self, total: int, log_dir: Path) -> None:
        if not self.enabled or self._header_printed:
            return
        print(f"EA eval runs (RL then manual), total={total}, logs -> {log_dir}")
        self._header_printed = True

    def render(self, state: Dict[str, Any], *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_print < self.refresh_s):
            return

        active = state.get("active", {})
        line = (
            f"[{format_duration(state['elapsed'])}] "
            f"done {state['completed']}/{state['total']} "
            f"| running {state['running']} | queued {state['queued']} | failed {state['failed']} "
            f"| active RL {active.get('rl', 0)} / manual {active.get('manual', 0)}"
        )
        last = state.get("last_result")
        if last:
            bf = last.get("best_fitness", {})
            rl_val = bf.get("ea_rl", bf.get("rl"))
            manual_val = bf.get("ea_only", bf.get("manual"))
            rl_str = f"{rl_val:.3f}" if isinstance(rl_val, (int, float)) else "n/a"
            manual_str = f"{manual_val:.3f}" if isinstance(manual_val, (int, float)) else "n/a"
            line += (
                f" | last floor {last['floor_id']:03d} "
                f"RL {rl_str} / manual {manual_str}"
            )

        if not force and line == self._last_line:
            return

        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)
        self._last_print = now
        self._last_line = line

    def finish(self) -> None:
        """Move to the next line after the final render."""
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()



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
    parser.add_argument(
        "--refresh-s",
        type=float,
        default=0.5,
        help="Status refresh interval for live logging.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable live status updates (still prints per-run summaries).",
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


def _save_run_log(*, log_dir: Path, base_run_id: str, floor_id: int, grid_size: int, rotate_k: int, runs: List[Dict[str, Any]], cfg) -> Path:
    now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{now}-{base_run_id}"
    history_map: Dict[str, Iterable[float]] = {}
    best_map: Dict[str, float] = {}
    for rec in runs:
        variant = rec.get("variant")
        hist = rec.get("history", [])
        best_val = rec.get("best_fitness_final")
        if variant:
            history_map[variant] = hist
            best_map[variant] = best_val
            # Compatibility aliases
            if variant == "ea_rl":
                history_map["rl"] = hist
                best_map["rl"] = best_val
            if variant == "ea_only":
                history_map["manual"] = hist
                best_map["manual"] = best_val

    payload = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "floor_id": floor_id,
        "grid_size": grid_size,
        "rotate_k": rotate_k,
        "runs": runs,
        "history": history_map,
        "best_fitness": best_map,
        "config": _config_summary(cfg),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ea_run_{run_id}_floor{floor_id:03d}.json"
    log_path.write_text(json.dumps(payload, indent=2))
    return log_path


def _run_ea(sample, cfg, seed_fn, *, cfg_seed_offset: int = 0, variant: str, seeder_name: str, bandit_arm_index: Optional[int], bandit_epsilon: Optional[float] = None) -> Dict[str, Any]:
    """Execute the EA for one seeding strategy and collect detailed stats."""
    rng_seed = cfg.random_seed + cfg_seed_offset
    rng = random.Random(rng_seed)

    from ver0.evolver import evaluate_population, init_population, make_next_generation, mutate  # type: ignore
    from ver0.real_plan_classifier import classify_real_floorplan  # type: ignore
    from ver0.vars import REALISM_THRESHOLD  # type: ignore

    pop = init_population(sample, cfg, rng, seed_fn)
    evaluate_population(sample, pop, cfg)

    def _fitness(g):
        return g.fitness if g.fitness is not None else float("inf")

    history: List[float] = []
    best_so_far = float("inf")
    best_gen = 0
    best_genome_copy = None

    # Initial stats
    best_initial_genome = min(pop, key=_fitness)
    best_fitness_initial = _fitness(best_initial_genome)
    mean_initial = sum(_fitness(g) for g in pop) / max(1, len(pop))

    current_mutation_rate = max(cfg.mutation_floor, cfg.mutation_rate)
    mutation_rates: List[float] = []
    stagnation = 0
    stagnation_events = 0
    diversity_injections = 0
    loop_start = time.time()

    for gen in range(cfg.generations):
        if gen > 0:
            prune_compactness = gen % 10 == 0
            fill_holes = gen % 10 == 0
            enforce_conn = prune_compactness
            pop = make_next_generation(
                sample,
                pop,
                cfg,
                rng,
                seed_fn,
                mutate,
                mutation_rate=current_mutation_rate,
                fill_holes=fill_holes,
                enforce_conn=enforce_conn,
                generation_index=gen,
            )
            evaluate_population(sample, pop, cfg)
        best = min(pop, key=_fitness)
        history.append(_fitness(best))
        if _fitness(best) < best_so_far:
            best_so_far = _fitness(best)
            best_gen = gen
            best_genome_copy = copy.deepcopy(best)
            stagnation = 0
            current_mutation_rate = max(cfg.mutation_floor, cfg.mutation_rate)
        else:
            stagnation += 1
            if cfg.stagnation_threshold > 0 and stagnation >= cfg.stagnation_threshold:
                replaced = False
                from ver0.evolver import inject_diversity  # type: ignore

                replaced = inject_diversity(
                    pop,
                    sample,
                    cfg,
                    rng,
                    seed_fn,
                    fraction=cfg.restart_fraction,
                )
                if replaced:
                    diversity_injections += 1
                    stagnation_events += 1
                    evaluate_population(sample, pop, cfg)
                    best = min(pop, key=_fitness)
                    history[-1] = _fitness(best)
                    if _fitness(best) < best_so_far:
                        best_so_far = _fitness(best)
                        best_gen = gen
                        best_genome_copy = copy.deepcopy(best)
                current_mutation_rate = min(cfg.mutation_ceiling, 0.3, current_mutation_rate * cfg.mutation_boost)
                stagnation = 0
        mutation_rates.append(current_mutation_rate)

    best_genome = best_genome_copy or best_initial_genome
    mean_final = sum(_fitness(g) for g in pop) / max(1, len(pop))
    realism_ok, realism_score, _ = classify_real_floorplan(
        sample, best_genome.layout, threshold=REALISM_THRESHOLD, scores=best_genome.scores
    )

    constraints = {}
    if best_genome.scores:
        constraints = {k: getattr(best_genome.scores, k, None) for k in vars(best_genome.scores)}

    return {
        "run_id": uuid.uuid4().hex,
        "variant": variant,
        "seeder_name": seeder_name,
        "bandit_arm_index": bandit_arm_index,
        "random_seed": rng_seed,
        "history": history,
        "best_fitness_final": best_so_far,
        "mean_fitness_final": mean_final,
        "best_fitness_initial": best_fitness_initial,
        "mean_fitness_initial": mean_initial,
        "gen_at_best": best_gen,
        "num_generations": len(history),
        "best_constraints": constraints,
        "realism_score": realism_score,
        "is_real": bool(realism_ok),
        "duration_s": time.time() - loop_start,
        "stagnation_events": stagnation_events,
        "num_diversity_injections": diversity_injections,
        "avg_mutation_rate": sum(mutation_rates) / max(1, len(mutation_rates)),
        "max_mutation_rate": max(mutation_rates) if mutation_rates else None,
        "bandit_epsilon": bandit_epsilon,
        "bandit_reward": -best_so_far if variant == "ea_rl" else None,
    }


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


def _run_single(job: Dict[str, Any], event_q=None) -> Dict[str, Any]:
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
    seeder_names = list(SEEDING_REGISTRY.keys())
    bandit_arm_index = seeder_names.index(seed_name_rl) if seed_name_rl in seeder_names else None
    manual_seed_name = next(iter(SEEDING_REGISTRY.keys()))
    seed_fn_manual = SEEDING_REGISTRY[manual_seed_name]

    if event_q is not None:
        event_q.put({"type": "stage", "run_idx": run_idx, "floor_id": floor_id, "stage": "rl"})
    run_rl = _run_ea(
        sample,
        cfg,
        seed_fn_rl,
        cfg_seed_offset=0,
        variant="ea_rl",
        seeder_name=seed_name_rl,
        bandit_arm_index=bandit_arm_index,
        bandit_epsilon=bandit.epsilon,
    )
    if event_q is not None:
        event_q.put({"type": "stage", "run_idx": run_idx, "floor_id": floor_id, "stage": "manual"})
    run_manual = _run_ea(
        sample,
        cfg,
        seed_fn_manual,
        cfg_seed_offset=1234,
        variant="ea_only",
        seeder_name=manual_seed_name,
        bandit_arm_index=None,
    )

    log_path = _save_run_log(
        log_dir=log_dir,
        base_run_id=f"r{run_idx:03d}",
        floor_id=floor_id,
        grid_size=grid_size,
        rotate_k=rotate_k,
        runs=[run_rl, run_manual],
        cfg=cfg,
    )

    return {
        "run_idx": run_idx,
        "floor_id": floor_id,
        "log_path": str(log_path),
        "seeders": {"rl": seed_name_rl, "manual": manual_seed_name},
        "best_fitness": {
            "ea_rl": run_rl["best_fitness_final"],
            "ea_only": run_manual["best_fitness_final"],
            "rl": run_rl["best_fitness_final"],
            "manual": run_manual["best_fitness_final"],
        },
    }


def build_jobs(*, runs: int, floor_pool: Sequence[int], rng: random.Random, log_dir: Path, project_root: Path, grid_size: int | None, rotate_k: int | None) -> List[Dict[str, Any]]:
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
    last_result: Optional[Dict[str, Any]] = None

    ui = TerminalUI(refresh_s=args.refresh_s, enabled=not args.no_ui)
    active_stages: Dict[int, str] = {}

    with multiprocessing.Manager() as mp_manager:
        event_q = mp_manager.Queue()
        ui.print_header(total=len(jobs), log_dir=log_dir)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_run_single, job, event_q): job for job in jobs}
            pending = set(future_map.keys())

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=args.refresh_s, return_when=concurrent.futures.FIRST_COMPLETED
                )

                # Drain stage updates
                while True:
                    try:
                        evt = event_q.get_nowait()
                    except queue.Empty:
                        break
                    if evt.get("type") == "stage":
                        active_stages[evt["run_idx"]] = evt.get("stage", "")

                for fut in done:
                    job = future_map[fut]
                    active_stages.pop(job["run_idx"], None)
                    try:
                        res = fut.result()
                        results.append(res)
                        last_result = res
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"run {job['run_idx']:03d} floor {job['floor_id']:03d}: {exc!r}")

                active_counts = {"rl": 0, "manual": 0}
                for stage in active_stages.values():
                    if stage in active_counts:
                        active_counts[stage] += 1

                state = {
                    "elapsed": time.time() - start,
                    "completed": len(results),
                    "total": len(jobs),
                    "running": len(pending),
                    "queued": max(0, len(jobs) - len(results) - len(pending)),
                    "failed": len(failures),
                    "last_result": last_result,
                    "log_dir": str(log_dir),
                    "active": active_counts,
                }
                ui.render(state)

    # Final state render
    ui.render(
        {
            "elapsed": time.time() - start,
            "completed": len(results),
            "total": len(jobs),
            "running": 0,
            "queued": 0,
            "failed": len(failures),
            "last_result": last_result,
            "log_dir": str(log_dir),
            "active": {"rl": 0, "manual": 0},
        },
        force=True,
    )
    ui.finish()

    elapsed = time.time() - start
    print(f"Finished {len(results)} run(s) with {len(failures)} failure(s) in {elapsed/60:.1f} min.")
    if failures:
        for msg in failures:
            print(f" - {msg}")


if __name__ == "__main__":
    main()
