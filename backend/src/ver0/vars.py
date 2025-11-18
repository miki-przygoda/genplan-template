from __future__ import annotations

import random

# Grid defaults
DEFAULT_GRID_SIZE = 4

# EA configuration defaults
POPULATION_SIZE = 64
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3
TOURNAMENT_K = 3
ELITE_FRACTION = 0.05
RANDOM_SEED = random.randint(0, 1_000_000)

# Fitness weights (lower is better; weights are penalties)
QUADRANT_WEIGHT = 1.0
OVERLAP_WEIGHT = 3.0
AREA_WEIGHT = 1.0
COMPACTNESS_WEIGHT = 0.5
ADJACENCY_WEIGHT = 0.5
