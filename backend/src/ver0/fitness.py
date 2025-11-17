# backend/src/ver0/fitness.py
from __future__ import annotations
from dataclasses import dataclass
from .constraints import score_constraints, CandidateLayout, ConstraintScores
from .grid_encoder import GridSample

# Fitness vars:

QUADRANT_WEIGHT = 1.0
OVERLAP_WEIGHT = 3.0
AREA_WEIGHT = 1.0
COMPACTNESS_WEIGHT = 0.5
ADJACENCY_WEIGHT = 0.5



@dataclass(frozen=True)
class Weights:
    quadrant: float = QUADRANT_WEIGHT
    overlap: float = OVERLAP_WEIGHT
    area: float = AREA_WEIGHT
    compactness: float = COMPACTNESS_WEIGHT
    adjacency: float = ADJACENCY_WEIGHT

def scalarize(scores: ConstraintScores, w: Weights) -> float:
    return (
        w.quadrant   * scores.quadrant +
        w.overlap    * scores.overlap +
        w.area       * scores.area +
        w.compactness* scores.compactness +
        w.adjacency  * scores.adjacency
    )

def evaluate(sample: GridSample, cand: CandidateLayout, w: Weights = Weights()) -> tuple[float, ConstraintScores]:
    scores = score_constraints(sample, cand)
    return scalarize(scores, w), scores