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
    BUDGET_WEIGHT,
    SECTION_BBOX_WEIGHT,
    MASK_WEIGHT,
    REALISM_WEIGHT,
    REALISM_THRESHOLD,
    RELATIONSHIP_WEIGHT,
    HOLE_WEIGHT,
)
from .real_plan_classifier import classify_real_floorplan


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
    budget: float = BUDGET_WEIGHT
    section_bbox: float = SECTION_BBOX_WEIGHT
    mask: float = MASK_WEIGHT
    realism: float = REALISM_WEIGHT
    relationships: float = RELATIONSHIP_WEIGHT
    holes: float = HOLE_WEIGHT


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
        + w.budget * getattr(scores, "budget", 0.0)
        + w.section_bbox * getattr(scores, "section_bbox", 0.0)
        + w.mask * getattr(scores, "mask", 0.0)
        + w.relationships * getattr(scores, "relationships", 0.0)
        + w.holes * getattr(scores, "holes", 0.0)
    )


def evaluate(sample: GridSample, cand: CandidateLayout, w: Weights = Weights()) -> tuple[float, ConstraintScores]:
    scores = score_constraints(sample, cand)
    realism_ok, realism_score, _ = classify_real_floorplan(sample, cand, threshold=REALISM_THRESHOLD, scores=scores)
    realism_penalty = max(0.0, realism_score - REALISM_THRESHOLD)
    total = scalarize(scores, w) + w.realism * realism_penalty
    return total, scores
