from __future__ import annotations

from .constraints import score_constraints, CandidateLayout, ConstraintScores
from .grid_encoder import GridSample


def _expected_total_cells(sample: GridSample) -> int:
    total = 0
    for spec in sample.rooms:
        if getattr(spec, "is_active", True):
            total += max(4, spec.expected_cells or spec.min_cells or 4)
    return max(1, total)


def realism_score(sample: GridSample, cand: CandidateLayout, scores: ConstraintScores | None = None) -> tuple[float, dict[str, float]]:
    scores = scores or score_constraints(sample, cand)
    total_expected = _expected_total_cells(sample)
    overlap_norm = scores.overlap / max(1, sample.grid_size * sample.grid_size)
    budget = getattr(scores, "budget", 0.0)
    area_norm = scores.area / total_expected
    section_miss = getattr(scores, "section_bbox", 0.0) / total_expected
    dispersion_norm = getattr(scores, "dispersion", 0.0) / total_expected
    compact_norm = scores.compactness / max(1, len(sample.rooms))
    mask_norm = getattr(scores, "mask", 0.0) / total_expected
    detail = {
        "overlap_norm": overlap_norm,
        "budget": budget,
        "area_norm": area_norm,
        "section_miss": section_miss,
        "dispersion_norm": dispersion_norm,
        "compact_norm": compact_norm,
        "mask_norm": mask_norm,
    }
    score = (
        3.0 * overlap_norm
        + 2.0 * budget
        + 2.0 * area_norm
        + 1.5 * section_miss
        + 0.5 * dispersion_norm
        + 0.5 * compact_norm
        + 0.5 * mask_norm
    )
    return score, detail


def classify_real_floorplan(sample: GridSample, cand: CandidateLayout, threshold: float = 1.0, scores: ConstraintScores | None = None) -> tuple[bool, float, dict[str, float]]:
    score, detail = realism_score(sample, cand, scores=scores)
    return score <= threshold, score, detail
