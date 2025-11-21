# backend/src/ver0/fitness.py
from __future__ import annotations
from dataclasses import dataclass

from .constraints import score_constraints, CandidateLayout, ConstraintScores
from .grid_encoder import GridSample
from .vars import (
    QUADRANT_WEIGHT,
    OVERLAP_WEIGHT,
    AREA_WEIGHT,
    COMPACTNESS_WEIGHT,
    ADJACENCY_WEIGHT,
    LOCATION_WEIGHT,
    SECTION_WEIGHT,
    DISPERSION_WEIGHT,
    ROOM_USAGE_WEIGHT,
)


@dataclass(frozen=True)
class Weights:
    quadrant: float = QUADRANT_WEIGHT
    overlap: float = OVERLAP_WEIGHT
    area: float = AREA_WEIGHT
    compactness: float = COMPACTNESS_WEIGHT
    adjacency: float = ADJACENCY_WEIGHT
    location: float = LOCATION_WEIGHT
    section: float = SECTION_WEIGHT
    dispersion: float = DISPERSION_WEIGHT
    room_usage: float = ROOM_USAGE_WEIGHT


def scalarize(scores: ConstraintScores, w: Weights) -> float:
    return (
        w.quadrant * scores.quadrant
        + w.overlap * scores.overlap
        + w.area * scores.area
        + w.compactness * scores.compactness
        + w.adjacency * scores.adjacency
        + w.location * getattr(scores, "location", 0.0)
        + w.section * getattr(scores, "section", 0.0)
        + w.dispersion * getattr(scores, "dispersion", 0.0)
        + w.room_usage * getattr(scores, "room_usage", 0.0)
    )


def evaluate(sample: GridSample, cand: CandidateLayout, w: Weights = Weights()) -> tuple[float, ConstraintScores]:
    scores = score_constraints(sample, cand)
    return scalarize(scores, w), scores
