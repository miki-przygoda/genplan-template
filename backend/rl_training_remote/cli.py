#!/usr/bin/env python3
"""
Terminal runner for long RL bandit training sessions.

This script mirrors the RLTraining notebook but adds:
 - CLI arguments for run length and EA configuration
 - A lightweight terminal UI with live CPU/memory usage
 - Graceful shutdown on Ctrl+C or 'q'
 - Safe persistence of bandit state and episode logs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os   
import random
import signal
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import curses
except Exception:
    curses = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


DEFAULT_FIXED_FLOORS = [101, 205, 309, 412, 523, 634, 745, 856, 967, 120, 230, 340]
DEFAULT_REFRESH_S = 0.5
MIN_EPSILON = 0.05


def find_project_root() -> Path:
    """Walk upwards to locate the repo root that contains backend/src."""
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "backend" / "src").exists():
            return candidate
    raise RuntimeError("Could not locate backend/src. Run from inside the repo.")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    mins, sec = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"


class UsageProbe:
    """Collect CPU and memory metrics for the current process tree."""

    def __init__(self) -> None:
        self.psutil = psutil
        self.proc = psutil.Process() if psutil else None
        if self.proc:
            self.proc.cpu_percent(interval=None)
            for child in self.proc.children(recursive=True):
                child.cpu_percent(interval=None)

    def sample(self) -> Dict[str, Any]:
        loadavg = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        if self.proc:
            children = self.proc.children(recursive=True)
            proc_cpu = self.proc.cpu_percent(interval=None)
            child_cpu = sum(child.cpu_percent(interval=None) for child in children)
            cpu_total = psutil.cpu_percent(interval=None)
            mem_mb = self.proc.memory_info().rss / (1024 * 1024)
            return {
                "cpu_total": cpu_total,
                "proc_cpu": proc_cpu,
                "child_cpu": child_cpu,
                "mem_mb": mem_mb,
                "load": loadavg,
                "provider": "psutil",
            }
        # Fallback: only report load average
        return {
            "cpu_total": None,
            "proc_cpu": None,
            "child_cpu": None,
            "mem_mb": None,
            "load": loadavg,
            "provider": "loadavg",
        }


class TerminalUI:
    """Minimal terminal UI; falls back to plain prints if curses unavailable."""

    def __init__(self, refresh_s: float = DEFAULT_REFRESH_S, allow_curses: bool = True) -> None:
        self.refresh_s = refresh_s
        self._screen = None
        self._use_curses = allow_curses and sys.stdout.isatty() and curses is not None
        self._last_print = 0.0
        self._log = deque(maxlen=6)

    def __enter__(self) -> "TerminalUI":
        if self._use_curses:
            self._screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self._screen.nodelay(True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._screen:
            curses.nocbreak()
            self._screen.nodelay(False)
            curses.echo()
            curses.endwin()
            self._screen = None

    def push_event(self, line: str) -> None:
        self._log.appendleft(line)

    def _build_lines(self, state: Dict[str, Any]) -> List[str]:
        lines = [
            "RL bandit training (q to stop, Ctrl+C to interrupt)",
            f"Runtime {format_duration(state['elapsed'])} / limit {state['runtime_limit']}",
            f"Episodes: {state['completed']}/{state['target']} | running {state['running']} | queued {state['queued']} | failed {state['failed']}",
            f"Bandit epsilon: {state['epsilon']:.3f}",
        ]
        cpu = state.get("cpu", {})
        if cpu.get("provider") == "psutil":
            load_display = f"{cpu['load']:.2f}" if cpu.get("load") is not None else "-"
            lines.append(
                f"CPU total {cpu['cpu_total']:.1f}% | proc {cpu['proc_cpu']:.1f}% | children {cpu['child_cpu']:.1f}% | mem {cpu['mem_mb']:.1f} MB | load {load_display}"
            )
        elif cpu.get("load") is not None:
            lines.append(f"Load avg (1m): {cpu['load']:.2f}")
        else:
            lines.append("CPU metrics unavailable (install psutil for detailed stats)")

        if state.get("last_result"):
            res = state["last_result"]
            lines.append(
                f"Last episode #{res['episode']:03d}: floor {res['floor_id']:03d}, seed {res['seed']}, best {res['best_fitness']:.3f}, {res.get('duration_s', 0):.1f}s"
            )
        lines.append(f"Log: {state['log_path']}")
        lines.append(f"State: {state['state_path']}")
        lines.append("Recent events:")
        lines.extend([f"  {msg}" for msg in self._log])
        return lines

    def render(self, state: Dict[str, Any]) -> bool:
        lines = self._build_lines(state)
        if self._use_curses and self._screen:
            self._screen.erase()
            max_y, max_x = self._screen.getmaxyx()
            for row, line in enumerate(lines[: max_y - 1]):
                try:
                    self._screen.addstr(row, 0, line[: max_x - 1])
                except Exception:
                    # Terminal too small; ignore drawing errors
                    pass
            self._screen.refresh()
            ch = self._screen.getch()
            return ch in (ord("q"), ord("Q"))
        # Plain stdout fallback
        now = time.time()
        if now - self._last_print >= self.refresh_s:
            print("\n".join(lines), flush=True)
            print("-" * 60, flush=True)
            self._last_print = now
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long RL bandit training with live monitoring.")
    parser.add_argument("--episodes", type=int, default=50, help="How many episodes to run (upper bound).")
    parser.add_argument("--gens", type=int, default=150, help="Generations per episode.")
    parser.add_argument("--population", type=int, default=None, help="Population size per episode.")
    parser.add_argument("--grid-size", type=int, default=None, help="Grid size override.")
    parser.add_argument("--rotate-k", type=int, default=None, help="Rotation augmentation (k*90 degrees).")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Initial epsilon for the bandit (decayed on start).")
    parser.add_argument("--epsilon-decay", type=float, default=0.98, help="Decay multiplier applied once per run.")
    parser.add_argument("--min-epsilon", type=float, default=MIN_EPSILON, help="Lower bound for epsilon.")
    parser.add_argument("--max-workers", type=int, default=None, help="Process workers (default: half of CPUs).")
    parser.add_argument("--max-runtime-min", type=float, default=None, help="Stop after this many minutes (graceful).")
    parser.add_argument("--refresh-s", type=float, default=DEFAULT_REFRESH_S, help="UI refresh interval.")
    parser.add_argument("--log-path", type=Path, default=None, help="Episode log JSONL path.")
    parser.add_argument("--state-path", type=Path, default=None, help="Bandit state JSON path.")
    parser.add_argument("--reset-state", action="store_true", help="Delete existing bandit/log files before running.")
    parser.add_argument("--random-floors", action="store_true", help="Pick floors randomly instead of cycling fixed ids.")
    parser.add_argument("--fixed-floors", type=int, nargs="*", default=None, help="Specific floor ids to cycle through.")
    parser.add_argument("--no-ui", action="store_true", help="Disable curses UI; print summaries instead.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--light-mutation", action="store_true", help="Use lighter mutation/repair for faster remote runs.")
    return parser.parse_args(argv)


def build_jobs(
    *,
    episodes: int,
    use_fixed_floors: bool,
    fixed_floors: Sequence[int],
    bandit,
    cfg_dict: Dict[str, Any],
    project_root: Path,
    grid_size: int,
    rotate_k: int,
) -> List[Tuple[int, int, str, Dict[str, Any], str, int, int]]:
    jobs = []
    for ep in range(episodes):
        if use_fixed_floors and fixed_floors:
            floor_id = fixed_floors[ep % len(fixed_floors)]
        else:
            floor_id = random.randint(1, 970)
        seed_name, _ = bandit.select()
        jobs.append((ep, floor_id, seed_name, dict(cfg_dict), str(project_root), grid_size, rotate_k))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "backend" / "src"))

    from ver0.evolver import EAConfig  # type: ignore
    from ver0.fitness import Weights  # type: ignore
    from ver0.rl_bandit import make_seed_bandit  # type: ignore
    from ver0.vars import (  # type: ignore
        ADJACENCY_WEIGHT,
        AREA_WEIGHT,
        BUDGET_WEIGHT,
        COMPACTNESS_WEIGHT,
        CROSSOVER_RATE,
        DEFAULT_GRID_SIZE,
        DISPERSION_WEIGHT,
        ELITE_FRACTION,
        GENERATIONS,
        LOCATION_WEIGHT,
        MASK_WEIGHT,
        MUTATION_RATE,
        NO_CHANGE_PENALTY,
        OVERLAP_WEIGHT,
        POPULATION_SIZE,
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

    rng = random.Random(args.seed or RANDOM_SEED)
    grid_size = args.grid_size or DEFAULT_GRID_SIZE
    rotate_k = args.rotate_k if args.rotate_k is not None else ROTATE_IMAGE_K
    episodes = max(1, args.episodes)
    population_size = args.population or POPULATION_SIZE
    max_workers = args.max_workers or max(1, (os.cpu_count() or 2) // 2)

    state_path = args.state_path or (project_root / "backend" / "data" / "rl" / "seed_bandit.json")
    log_path = args.log_path or (project_root / "backend" / "data" / "rl" / "episode_log.jsonl")

    if args.reset_state:
        for path in [state_path, log_path]:
            if path.exists():
                path.unlink()
                print(f"Deleted {path}")

    use_fixed = not args.random_floors
    fixed_floors = args.fixed_floors if args.fixed_floors is not None else DEFAULT_FIXED_FLOORS

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

    cfg = EAConfig(
        population_size=population_size,
        generations=args.gens or GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        tournament_k=TOURNAMENT_K,
        elite_fraction=ELITE_FRACTION,
        random_seed=rng.randint(0, 1_000_000),
        weights=weights,
        stagnation_threshold=20,
        restart_fraction=0.30,
        mutation_boost=1.5,
        mutation_floor=0.05,
        mutation_ceiling=0.65,
        no_change_penalty=NO_CHANGE_PENALTY,
    )
    cfg_dict = asdict(cfg)

    bandit = make_seed_bandit(state_path=state_path, epsilon=args.epsilon, rng=rng)
    bandit.epsilon = max(args.min_epsilon, bandit.epsilon * args.epsilon_decay)

    max_runtime = args.max_runtime_min * 60 if args.max_runtime_min else None
    probe = UsageProbe()
    ui_allow_curses = not args.no_ui

    jobs = build_jobs(
        episodes=episodes,
        use_fixed_floors=use_fixed,
        fixed_floors=fixed_floors,
        bandit=bandit,
        cfg_dict=cfg_dict,
        project_root=project_root,
        grid_size=grid_size,
        rotate_k=rotate_k,
    )
    # Append light_mutation flag to each job for the runner
    jobs = [job + (args.light_mutation,) for job in jobs]

    from ver0.rl_runner import run_episode  # type: ignore

    start_time = time.time()
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    completed: List[Dict[str, Any]] = []
    failures: List[str] = []
    stop_requested = False
    stop_reason = ""
    last_result: Optional[Dict[str, Any]] = None

    def handle_interrupt(signum, frame) -> None:
        nonlocal stop_requested, stop_reason
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        stop_reason = "Keyboard interrupt requested; finishing current jobs."

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    pending: set[concurrent.futures.Future] = set()
    job_iter = iter(jobs)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    jobs_total = len(jobs)
    submitted = 0

    init_msg = (
        f"Init: episodes={jobs_total}, gens={cfg.generations}, pop={cfg.population_size}, "
        f"max_workers={max_workers}, runtime_limit={'none' if max_runtime is None else f'{max_runtime/60:.1f} min'}, "
        f"epsilon={bandit.epsilon:.3f}, light_mutation={args.light_mutation}"
    )

    run_error: Optional[BaseException] = None

    try:
        with TerminalUI(refresh_s=args.refresh_s, allow_curses=ui_allow_curses) as ui:
            ui.push_event(init_msg)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                while True:
                    while not stop_requested and len(pending) < max_workers and submitted < jobs_total:
                        try:
                            job = next(job_iter)
                        except StopIteration:
                            break
                        pending.add(ex.submit(run_episode, job))
                        submitted += 1
                        ui.push_event(f"Submitted episode {job[0]+1}/{jobs_total} (running {len(pending)}/{max_workers})")

                    if not pending and (stop_requested or submitted >= jobs_total):
                        break

                    done, pending = concurrent.futures.wait(
                        pending, timeout=args.refresh_s, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        try:
                            rec = fut.result()
                            completed.append(rec)
                            last_result = rec
                            bandit.update(rec["seed"], -rec["best_fitness"])
                            with log_path.open("a") as f:
                                f.write(json.dumps(rec) + "\n")
                            ui.push_event(
                                f"Episode {rec['episode']:03d} finished: seed {rec['seed']} best {rec['best_fitness']:.3f}"
                            )
                        except Exception as exc:
                            failures.append(repr(exc))
                            ui.push_event(f"Job failed: {exc!r}")

                    elapsed = time.time() - start_time
                    cpu = probe.sample()
                    state = {
                        "elapsed": elapsed,
                        "runtime_limit": format_duration(max_runtime) if max_runtime else "no limit",
                        "completed": len(completed),
                        "target": jobs_total,
                        "running": len(pending),
                        "queued": max(0, jobs_total - submitted),
                        "failed": len(failures),
                        "epsilon": bandit.epsilon,
                        "cpu": cpu,
                        "last_result": last_result,
                        "log_path": str(log_path),
                        "state_path": str(state_path),
                    }

                    if ui.render(state):
                        stop_requested = True
                        stop_reason = "Stopped by user request (q)."

                    if max_runtime and elapsed >= max_runtime:
                        stop_requested = True
                        stop_reason = f"Reached runtime limit {format_duration(max_runtime)}."

                    if stop_requested and not pending:
                        break
    except KeyboardInterrupt:
        stop_requested = True
        stop_reason = stop_reason or "Keyboard interrupt received; exiting."
    except BaseException as exc:  # noqa: BLE001
        run_error = exc
        stop_reason = stop_reason or f"Aborted due to error: {exc!r}"
    finally:
        signal.signal(signal.SIGINT, old_handler)
        bandit.save()

    if stop_requested and stop_reason:
        print(stop_reason)
    end_time = time.time()
    end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed = end_time - start_time
    print(f"Completed {len(completed)} episode(s); {len(failures)} failure(s).")
    print(f"Started at: {start_ts}")
    print(f"Finished at: {end_ts}")
    print(f"Elapsed: {format_duration(elapsed)}")
    if failures:
        for msg in failures:
            print(f" - {msg}")
    if run_error:
        raise run_error


if __name__ == "__main__":
    main()
