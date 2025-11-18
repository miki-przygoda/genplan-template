from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .grid_encoder import GridSample, RoomSpec
from .vars import DEFAULT_GRID_SIZE

Cell = tuple[int, int]  # (r, c)

@dataclass
class CandidateLayout:
    """room_name -> list of (r, c) cells in a 4x4 grid."""
    placement: Dict[str, List[Cell]]

@dataclass
class ConstraintScores:
    quadrant: float
    overlap: float
    area: float
    compactness: float
    adjacency: float

# ---------- helpers ----------
def quadrant_of_cell(rc: Cell, grid_size: int = DEFAULT_GRID_SIZE) -> str:
    r, c = rc
    ns = "N" if r < grid_size // 2 else "S"
    ew = "W" if c < grid_size // 2 else "E"
    return ns + ew

def centroid_of_cells(cells: List[Cell]) -> Cell:
    r = int(round(sum(x for x, _ in cells) / max(1, len(cells))))
    c = int(round(sum(y for _, y in cells) / max(1, len(cells))))
    return (r, c)

def perimeter_of_cells(cells: List[Cell]) -> int:
    """4-neighbour perimeter on a grid."""
    S = set(cells)
    per = 0
    for (r, c) in S:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (r+dr, c+dc) not in S:
                per += 1
    return per

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ---------- penalties ----------
def quadrant_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """Penalty for rooms whose centroid cell is outside target quadrant (distance to nearest in-quadrant cell)."""
    g = sample.grid_size
    total = 0.0
    for spec in sample.rooms:
        cells = cand.placement.get(spec.name, [])
        if not cells:
            continue
        ctr = centroid_of_cells(cells)
        q = quadrant_of_cell(ctr, g)
        if spec.section in {"NW","NE","SW","SE"}:
            if q != spec.section:
                q_center = {
                    "NW": (g//4, g//4),
                    "NE": (g//4, g - 1 - g//4),
                    "SW": (g - 1 - g//4, g//4),
                    "SE": (g - 1 - g//4, g - 1 - g//4),
                }[spec.section]
                total += manhattan(ctr, q_center)
    return total

def overlap_penalty(cand: CandidateLayout, grid_size: int = DEFAULT_GRID_SIZE) -> float:
    occ = np.zeros((grid_size, grid_size), dtype=int)
    for cells in cand.placement.values():
        for (r, c) in cells:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                occ[r, c] += 1
    return float((occ > 1).sum())

def area_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    total = 0.0
    for spec in sample.rooms:
        assigned = len(cand.placement.get(spec.name, []))
        total += abs(assigned - spec.min_cells)
    return float(total)

def compactness_penalty(cand: CandidateLayout) -> float:
    """Bounding-box perimeter / max(1, area) per room; sum across rooms."""
    total = 0.0
    for cells in cand.placement.values():
        if not cells:
            continue
        rs = [r for (r, _) in cells]
        cs = [c for (_, c) in cells]
        h = (max(rs) - min(rs) + 1)
        w = (max(cs) - min(cs) + 1)
        bbox_perimeter = 2 * (h + w)
        area = len(cells)
        total += bbox_perimeter / max(1, area)
    return float(total)

def adjacency_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """
    Optional v0 signal: if text implied adjacency (not available yet),
    approximate with 'common pairs' that often co-occur (kitchen-dining, bedroom-bathroom):
    penalize large centroid distances.
    """
    COMMON = [
        ("kitchen_1", "dining_1"),
        ("bedroom_1", "bathroom_1"),
    ]
    total = 0.0
    for a, b in COMMON:
        ca = centroid_of_cells(cand.placement.get(a, [])) if cand.placement.get(a) else None
        cb = centroid_of_cells(cand.placement.get(b, [])) if cand.placement.get(b) else None
        if ca and cb:
            total += max(0, manhattan(ca, cb) - 1)
    return total

# ---------- main API ----------
def score_constraints(sample: GridSample, cand: CandidateLayout) -> ConstraintScores:
    return ConstraintScores(
        quadrant=quadrant_penalty(sample, cand),
        overlap=overlap_penalty(cand, sample.grid_size),
        area=area_penalty(sample, cand),
        compactness=compactness_penalty(cand),
        adjacency=adjacency_penalty(sample, cand),
    )
