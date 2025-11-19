from __future__ import annotations

import random

# Grid defaults (increased to 16x16 = 4x4 of 4x4 cells)
DEFAULT_GRID_SIZE = 16

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
OVERLAP_WEIGHT = 4.0
AREA_WEIGHT = 3.0        # stronger push to hit target cell counts (min 2x2)
COMPACTNESS_WEIGHT = 3.0 # penalize fragmented/gappy rooms more
ADJACENCY_WEIGHT = 0.5
LOCATION_WEIGHT = 1.0    # pulls rooms toward their target location
