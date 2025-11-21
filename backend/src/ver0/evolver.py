from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Callable
import random
import copy

from pathlib import Path

from .grid_encoder import GridSample, encode_floorplan_to_grid
from .constraints import CandidateLayout, ConstraintScores, Cell, manhattan, centroid_of_cells
from .fitness import evaluate, Weights
from .text_to_support_text import section_bounds, section_to_cell
from .vars import (
    POPULATION_SIZE,
    GENERATIONS,
    CROSSOVER_RATE,
    MUTATION_RATE,
    TOURNAMENT_K,
    ELITE_FRACTION,
    RANDOM_SEED,
    DEFAULT_GRID_SIZE,
    NO_CHANGE_PENALTY,
)

# Type aliases
MakeRandomFn = Callable[[GridSample, random.Random], CandidateLayout]
MutateFn = Callable[[CandidateLayout, random.Random, float], None]
GrowMutateFn = Callable[[CandidateLayout, random.Random, Dict[str, int]], None]

def layout_signature(layout: CandidateLayout) -> tuple:
    """Canonical representation of a layout for change detection."""
    signature: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for room, cells in layout.placement.items():
        signature.append((room, tuple(sorted(cells))))
    signature.sort(key=lambda item: item[0])
    return tuple(signature)

def make_sample(floor_id: str | int) -> GridSample:
    """Load a GridSample from a processed floor plan directory."""
    repo_root = Path(__file__).resolve().parents[3]  # .../genplan-template
    floor_dir = repo_root / "backend" / "data" / "processed" / "floor_plans" / f"floor{int(floor_id):03d}"
    return encode_floorplan_to_grid(floor_dir, grid_size=DEFAULT_GRID_SIZE)

def make_random_layout(sample: GridSample, rng: random.Random) -> CandidateLayout:
    """Seed the population with large, section-aware room blocks.

    Rooms are placed near their hinted section/target cell in descending size order
    so the initial population already covers big areas of the map. This gives the EA
    something close to the desired layout to prune (remove) or extend (add) cells from.
    """
    grid_size = sample.grid_size
    allowed = sample.active_room_names
    allowed_set = set(allowed) if allowed is not None else None
    placement: Dict[str, List[Cell]] = {spec.name: [] for spec in sample.rooms}
    occupied: set[Cell] = set()

    def _fill_region(center: Cell, target: int, bounds: tuple[int, int, int, int]) -> list[Cell]:
        """Pick cells nearest to center within bounds, avoiding current occupancy."""
        r0, r1, c0, c1 = bounds
        candidates: list[tuple[int, Cell]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rc = (r, c)
                if rc in occupied:
                    continue
                candidates.append((manhattan(rc, center), rc))
        # Prefer closer cells; small jitter to avoid symmetry
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0])
        chosen: list[Cell] = []
        for _, rc in candidates:
            chosen.append(rc)
            occupied.add(rc)
            if len(chosen) >= target:
                break
        return chosen

    # Focus on active/text rooms only; others stay empty.
    ordered_specs = sorted(
        [s for s in sample.rooms if (allowed_set is None or s.name in allowed_set)],
        key=lambda s: s.expected_cells or s.min_cells,
        reverse=True,
    )
    room_count = max(1, len(ordered_specs))
    # Allow each room to initially claim a generous share of the grid within its section.
    for room_spec in ordered_specs:
        target_cells = max(4, room_spec.expected_cells or room_spec.min_cells or 4)
        center = room_spec.target_cell or section_to_cell(room_spec.section, grid_size=grid_size)
        # share-based desired size so rooms start big: roughly 60% of equal partition
        share = int(0.6 * (grid_size * grid_size) / room_count)
        desired = min(grid_size * grid_size // 2, max(target_cells * 2, share))
        bounds = section_bounds(room_spec.section, grid_size=grid_size, half_span=max(4, grid_size // 3))
        cells = _fill_region(center, desired, bounds)
        if len(cells) < desired:
            # widen search to full grid if the section region is crowded
            global_bounds = (0, grid_size - 1, 0, grid_size - 1)
            cells.extend(_fill_region(center, desired - len(cells), global_bounds))
        placement[room_spec.name] = cells
    return CandidateLayout(
        placement=placement,
        active_rooms=allowed_set.copy() if allowed_set is not None else None,
    )

@dataclass
class Genome:
    """One candidate solution in the EA population."""
    layout: CandidateLayout
    fitness: float | None = None
    scores: ConstraintScores | None = None

@dataclass
class EAConfig:
    population_size: int = POPULATION_SIZE
    generations: int = GENERATIONS
    crossover_rate: float = CROSSOVER_RATE
    mutation_rate: float = MUTATION_RATE
    tournament_k: int = TOURNAMENT_K
    elite_fraction: float = ELITE_FRACTION
    random_seed: int | None = RANDOM_SEED
    weights: Weights = Weights()
    stagnation_threshold: int = 20
    restart_fraction: float = 0.25
    mutation_boost: float = 1.5
    mutation_floor: float = 0.02
    mutation_ceiling: float = 0.8
    no_change_penalty: float = NO_CHANGE_PENALTY

def _get_rng(seed: int | None) -> random.Random:
    return random.Random(seed)

def _fitness_value(genome: Genome) -> float:
    return genome.fitness if genome.fitness is not None else float("inf")

def evaluate_population(sample: GridSample, population: list[Genome], cfg: EAConfig, weights: Weights | None = None) -> None:
    for genome in population:
        if genome.fitness is None:
            f, scores = evaluate(sample, genome.layout, weights or cfg.weights)
            genome.fitness = f
            genome.scores = scores

def tournament_select(population: list[Genome], k: int, rng: random.Random) -> Genome:
    competitors = rng.sample(population, k)
    return min(competitors, key=lambda g: g.fitness if g.fitness is not None else float("inf"))

def select_parents(population: list[Genome], cfg: EAConfig, rng: random.Random) -> list[Genome]:
    parents: list[Genome] = []
    for _ in range(cfg.population_size):
        parents.append(tournament_select(population, cfg.tournament_k, rng))
    return parents

def init_population(sample: GridSample, cfg: EAConfig, rng: random.Random, make_random: MakeRandomFn) -> list[Genome]:
    population: list[Genome] = []
    for _ in range(cfg.population_size):
        layout = make_random(sample, rng)
        population.append(Genome(layout=layout))
    return population

def crossover(c1: CandidateLayout, c2: CandidateLayout, rng: random.Random) -> None:
    """
    Uniform crossover over rooms: for each room present in either parent,
    swap the room's cell list between parents with 50% probability.
    """
    rooms = set(c1.placement.keys()) | set(c2.placement.keys())
    for room in rooms:
        if rng.random() < 0.5:
            cells1 = copy.deepcopy(c1.placement.get(room, []))
            cells2 = copy.deepcopy(c2.placement.get(room, []))
            c1.placement[room] = cells2
            c2.placement[room] = cells1
        if c1.active_rooms is not None and room not in c1.active_rooms:
            c1.placement[room] = []
        if c2.active_rooms is not None and room not in c2.active_rooms:
            c2.placement[room] = []

def _infer_grid_size(layout: CandidateLayout, fallback: int = DEFAULT_GRID_SIZE) -> int:
    """Best-effort grid size guess from existing cell coordinates."""
    coords = [rc for cells in layout.placement.values() for rc in cells]
    if not coords:
        return fallback
    max_rc = max(max(r, c) for r, c in coords)
    return max(fallback, max_rc + 1)

def _jitter_weights(w: Weights, rng: random.Random, gen: int) -> Weights:
    """
    Mild noise every 10 generations to escape plateaus.
    """
    if gen % 10 != 0:
        return w
    factor = lambda: 1 + rng.uniform(-0.03, 0.03)
    return Weights(
        quadrant=max(0.0, w.quadrant * factor()),
        overlap=max(0.0, w.overlap * factor()),
        area=max(0.0, w.area * factor()),
        compactness=max(0.0, w.compactness * factor()),
        adjacency=max(0.0, w.adjacency * factor()),
        location=max(0.0, w.location * factor()),
    )

def mutate(layout: CandidateLayout, rng: random.Random, mutation_rate: float) -> None:
    """
    Mutate a layout by occasionally moving a single cell in a room to a 
    new grid position and optionally shuffling the room's cell ordering.
    """
    grid_size = _infer_grid_size(layout)
    allowed = layout.active_rooms
    for room_name, cells in layout.placement.items():
        if allowed is not None and room_name not in allowed:
            layout.placement[room_name] = []
            continue
        if not cells:
            continue
        if rng.random() < mutation_rate:
            idx = rng.randrange(len(cells))
            new_cell = (rng.randrange(grid_size), rng.randrange(grid_size))
            cells[idx] = new_cell
        if rng.random() < mutation_rate and len(cells) > 1:
            rng.shuffle(cells)

def grow_mutation(layout: CandidateLayout, rng: random.Random, target_size: Dict[str, int]) -> None:
    """
    Targeted growth/shrink: adjust rooms toward their target size near current shape.
    """
    grid_size = _infer_grid_size(layout)
    allowed = layout.active_rooms
    for room, cells in layout.placement.items():
        if allowed is not None and room not in allowed:
            layout.placement[room] = []
            continue
        tgt = target_size.get(room, None)
        if tgt is None:
            continue
        if not cells:
            continue
        if len(cells) < tgt:
            needed = tgt - len(cells)
            rs = [r for r,_ in cells]
            cs = [c for _,c in cells]
            rmin, rmax = max(0, min(rs)-1), min(grid_size-1, max(rs)+1)
            cmin, cmax = max(0, min(cs)-1), min(grid_size-1, max(cs)+1)
            candidates = [(r,c) for r in range(rmin, rmax+1) for c in range(cmin, cmax+1)]
            rng.shuffle(candidates)
            for rc in candidates:
                if len(cells) >= tgt:
                    break
                if rc not in cells:
                    cells.append(rc)
        elif len(cells) > tgt:
            # Trim farthest cells first to encourage compactness around centroid
            ctr = centroid_of_cells(cells)
            to_remove = len(cells) - tgt
            cells.sort(key=lambda rc: manhattan(rc, ctr), reverse=True)
            del cells[:to_remove]

def copy_layout(layout: CandidateLayout) -> CandidateLayout:
    """Create a deep copy of a CandidateLayout."""
    return CandidateLayout(
        placement=copy.deepcopy(layout.placement),
        active_rooms=set(layout.active_rooms) if layout.active_rooms is not None else None,
    )

def reproduce(parents: list[Genome], cfg: EAConfig, rng: random.Random) -> list[Genome]:
    offspring: list[Genome] = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % len(parents)]
        c1 = copy_layout(p1.layout)
        c2 = copy_layout(p2.layout)
        if rng.random() < cfg.crossover_rate:
            crossover(c1, c2, rng)
        offspring.append(Genome(layout=c1))
        offspring.append(Genome(layout=c2))
    return offspring

def make_next_generation(
    sample: GridSample,
    population: list[Genome],
    cfg: EAConfig,
    rng: random.Random,
    make_random: MakeRandomFn,
    mutate_fn: MutateFn = mutate,
    grow_mutate_fn: GrowMutateFn = grow_mutation,
    mutation_rate: float | None = None,
) -> list[Genome]:
    """Create the next generation using elitism, tournament selection, crossover, and mutation.
    Assumes the incoming population already has fitness evaluated.
    """
    sorted_pop = sorted(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
    num_elites = max(1, int(cfg.population_size * cfg.elite_fraction))
    new_population: list[Genome] = []
    for i in range(num_elites):
        elite_layout = copy_layout(sorted_pop[i].layout)
        new_population.append(Genome(layout=elite_layout, fitness=sorted_pop[i].fitness, scores=sorted_pop[i].scores))
    
    while len(new_population) < cfg.population_size:
        parent1 = tournament_select(population, cfg.tournament_k, rng)
        parent2 = tournament_select(population, cfg.tournament_k, rng)
        
        child_layout = copy_layout(parent1.layout)
        if rng.random() < cfg.crossover_rate:
            parent2_layout = copy_layout(parent2.layout)
            crossover(child_layout, parent2_layout, rng)
        effective_mutation = mutation_rate if mutation_rate is not None else cfg.mutation_rate
        effective_mutation = max(cfg.mutation_floor, min(cfg.mutation_ceiling, effective_mutation))
        mutate_fn(child_layout, rng, effective_mutation)
        # targeted growth pass
        target_sizes: Dict[str, int] = {}
        allowed = sample.active_room_names
        for spec in sample.rooms:
            if (allowed is not None and spec.name not in allowed) or not getattr(spec, "is_active", True):
                target_sizes[spec.name] = 0
            else:
                target_sizes[spec.name] = max(4, spec.expected_cells or spec.min_cells)
        grow_mutate_fn(child_layout, rng, target_sizes)

        new_population.append(Genome(layout=child_layout))
    return new_population

def inject_diversity(
    population: list[Genome],
    sample: GridSample,
    cfg: EAConfig,
    rng: random.Random,
    make_random: MakeRandomFn,
    fraction: float,
) -> bool:
    count = max(1, int(len(population) * max(0.0, min(1.0, fraction))))
    indexed = list(enumerate(population))
    indexed.sort(key=lambda pair: _fitness_value(pair[1]), reverse=True)
    replaced = False
    for idx, _ in indexed[:count]:
        layout = make_random(sample, rng)
        population[idx] = Genome(layout=layout)
        replaced = True
    return replaced

def evolve(sample: GridSample,cfg: EAConfig = EAConfig(),*,make_random: MakeRandomFn = make_random_layout,mutate_fn: MutateFn = mutate,) -> tuple[Genome, list[Genome], dict[str, list[float]]]:
    """
    Run the evolutionary algorithm and return:
      - best genome
      - final population
      - history dict with per-generation metrics
    """
    rng = _get_rng(cfg.random_seed)
    population = init_population(sample, cfg, rng, make_random)
    evaluate_population(sample, population, cfg)
    best = min(population, key=_fitness_value)

    history_best: list[float] = []
    history_mean: list[float] = []
    stagnation = 0
    current_mutation_rate = max(cfg.mutation_floor, cfg.mutation_rate)
    last_best_signature: tuple | None = None

    # record generation 0
    fitnesses = [g.fitness for g in population if g.fitness is not None]
    history_best.append(min(fitnesses))
    history_mean.append(sum(fitnesses) / len(fitnesses))

    for gen in range(cfg.generations):
        dynamic_weights = _jitter_weights(cfg.weights, rng, gen)
        population = make_next_generation(
            sample,
            population,
            cfg,
            rng,
            make_random,
            mutate_fn,
            mutation_rate=current_mutation_rate,
        )
        evaluate_population(sample, population, cfg, weights=dynamic_weights)

        fitnesses = [g.fitness for g in population if g.fitness is not None]
        gen_best = min(fitnesses)
        gen_mean = sum(fitnesses) / len(fitnesses)

        history_best.append(gen_best)
        history_mean.append(gen_mean)

        candidate = min(population, key=_fitness_value)
        penalized = False
        candidate_signature = layout_signature(candidate.layout)
        if (
            last_best_signature is not None
            and candidate_signature == last_best_signature
            and cfg.no_change_penalty > 0
            and candidate.fitness is not None
        ):
            candidate.fitness += cfg.no_change_penalty
            penalized = True
        if penalized:
            fitnesses = [g.fitness for g in population if g.fitness is not None]
            if fitnesses:
                gen_best = min(fitnesses)
                gen_mean = sum(fitnesses) / len(fitnesses)
                history_best[-1] = gen_best
                history_mean[-1] = gen_mean
            candidate = min(population, key=_fitness_value)
            stagnation = max(stagnation, cfg.stagnation_threshold)
        improved = False
        if candidate.fitness is not None and (best.fitness is None or candidate.fitness < best.fitness):
            best = candidate
            improved = True

        if improved:
            stagnation = 0
            current_mutation_rate = max(cfg.mutation_floor, cfg.mutation_rate)
        else:
            stagnation += 1
            if cfg.stagnation_threshold > 0 and stagnation >= cfg.stagnation_threshold:
                replaced = inject_diversity(
                    population,
                    sample,
                    cfg,
                    rng,
                    make_random,
                    fraction=cfg.restart_fraction,
                )
                if replaced:
                    evaluate_population(sample, population, cfg, weights=dynamic_weights)
                    fitnesses = [g.fitness for g in population if g.fitness is not None]
                    gen_best = min(fitnesses)
                    gen_mean = sum(fitnesses) / len(fitnesses)
                    history_best[-1] = gen_best
                    history_mean[-1] = gen_mean
                    candidate = min(population, key=_fitness_value)
                    if candidate.fitness is not None and (best.fitness is None or candidate.fitness < best.fitness):
                        best = candidate
                current_mutation_rate = min(cfg.mutation_ceiling, current_mutation_rate * cfg.mutation_boost)
                stagnation = 0
        if best.fitness is not None:
            last_best_signature = layout_signature(best.layout)

    history = {"best": history_best, "mean": history_mean}
    return best, population, history
