from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable, Callable
import random
import copy

from pathlib import Path

from .grid_encoder import GridSample, encode_floorplan_to_grid
from .constraints import CandidateLayout, ConstraintScores, Cell
from .fitness import evaluate, Weights
from .vars import (
    POPULATION_SIZE,
    GENERATIONS,
    CROSSOVER_RATE,
    MUTATION_RATE,
    TOURNAMENT_K,
    ELITE_FRACTION,
    RANDOM_SEED,
    DEFAULT_GRID_SIZE,
)

# Type aliases
MakeRandomFn = Callable[[GridSample, random.Random], CandidateLayout]
MutateFn = Callable[[CandidateLayout, random.Random, float], None]

def make_sample(floor_id: str | int) -> GridSample:
    """Load a GridSample from a processed floor plan directory."""
    repo_root = Path(__file__).resolve().parents[3]  # .../genplan-template
    floor_dir = repo_root / "backend" / "data" / "processed" / "floor_plans" / f"floor{int(floor_id):03d}"
    return encode_floorplan_to_grid(floor_dir, grid_size=DEFAULT_GRID_SIZE)

def make_random_layout(sample: GridSample, rng: random.Random) -> CandidateLayout:
    """Generate a random layout by assigning random cells to each room."""
    grid_size = sample.grid_size
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    rng.shuffle(all_cells)
    
    placement: Dict[str, List[Cell]] = {}
    cell_idx = 0
    
    for room_spec in sample.rooms:
        # Assign min_cells cells to this room
        num_cells = max(1, room_spec.min_cells)
        room_cells = all_cells[cell_idx:cell_idx + num_cells]
        placement[room_spec.name] = room_cells
        cell_idx += num_cells
        
        # If we've used all cells, reshuffle and continue
        if cell_idx >= len(all_cells):
            rng.shuffle(all_cells)
            cell_idx = 0
    
    return CandidateLayout(placement=placement)

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

def _get_rng(seed: int | None) -> random.Random:
    return random.Random(seed)

def evaluate_population(sample: GridSample, population: list[Genome], cfg: EAConfig) -> None:
    for genome in population:
        if genome.fitness is None:
            f, scores = evaluate(sample, genome.layout, cfg.weights)
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

def _infer_grid_size(layout: CandidateLayout, fallback: int = DEFAULT_GRID_SIZE) -> int:
    """Best-effort grid size guess from existing cell coordinates."""
    coords = [rc for cells in layout.placement.values() for rc in cells]
    if not coords:
        return fallback
    max_rc = max(max(r, c) for r, c in coords)
    return max(fallback, max_rc + 1)

def mutate(layout: CandidateLayout, rng: random.Random, mutation_rate: float) -> None:
    """
    Mutate a layout by occasionally moving a single cell in a room to a 
    new grid position and optionally shuffling the room's cell ordering.
    """
    grid_size = _infer_grid_size(layout)
    for cells in layout.placement.values():
        if not cells:
            continue
        if rng.random() < mutation_rate:
            idx = rng.randrange(len(cells))
            new_cell = (rng.randrange(grid_size), rng.randrange(grid_size))
            cells[idx] = new_cell
        if rng.random() < mutation_rate and len(cells) > 1:
            rng.shuffle(cells)

def copy_layout(layout: CandidateLayout) -> CandidateLayout:
    """Create a deep copy of a CandidateLayout."""
    return CandidateLayout(placement=copy.deepcopy(layout.placement))

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

def make_next_generation(sample: GridSample,population: list[Genome],cfg: EAConfig,rng: random.Random,make_random: MakeRandomFn,mutate_fn: MutateFn = mutate) -> list[Genome]:
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
        mutate_fn(child_layout, rng, cfg.mutation_rate)
        
        new_population.append(Genome(layout=child_layout))
    return new_population

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
    best = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))

    history_best: list[float] = []
    history_mean: list[float] = []

    # record generation 0
    fitnesses = [g.fitness for g in population if g.fitness is not None]
    history_best.append(min(fitnesses))
    history_mean.append(sum(fitnesses) / len(fitnesses))

    for _ in range(cfg.generations):
        population = make_next_generation(sample, population, cfg, rng, make_random, mutate_fn)
        evaluate_population(sample, population, cfg)

        fitnesses = [g.fitness for g in population if g.fitness is not None]
        gen_best = min(fitnesses)
        gen_mean = sum(fitnesses) / len(fitnesses)

        history_best.append(gen_best)
        history_mean.append(gen_mean)

        candidate = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
        if candidate.fitness is not None and (best.fitness is None or candidate.fitness < best.fitness):
            best = candidate

    history = {"best": history_best, "mean": history_mean}
    return best, population, history
