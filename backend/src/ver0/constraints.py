from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .grid_encoder import GridSample, RoomSpec, cell_to_nested
from .vars import DEFAULT_GRID_SIZE

Cell = tuple[int, int]  # (r, c)

@dataclass
class CandidateLayout:
    """room_name -> list of (r, c) cells in the working grid."""
    placement: Dict[str, List[Cell]]

@dataclass
class ConstraintScores:
    quadrant: float
    overlap: float
    area: float
    compactness: float
    adjacency: float
    location: float = 0.0

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

def hierarchical_reference(cand: CandidateLayout, grid_size: int = DEFAULT_GRID_SIZE, base: int = 4) -> dict[str, list[list[tuple[int, int]]]]:
    """
    Build a nested cell reference for each room: for every (r,c) return its hierarchical path
    across base x base tiles (e.g., 16x16 -> 4x4 blocks-of-4).
    """
    levels = 1
    size = base
    while size < grid_size and levels < 5:
        size *= base
        levels += 1

    return {
        room: [cell_to_nested(rc, base=base, levels=levels) for rc in cells]
        for room, cells in cand.placement.items()
    }

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
        target = spec.expected_cells or spec.min_cells or 4
        target = max(4, target)  # enforce 2x2 minimum
        total += abs(assigned - target)
    return float(total)

def compactness_penalty(cand: CandidateLayout) -> float:
    """
    Bounding-box perimeter / area plus fragmentation penalties:
    - penalize multiple components
    - penalize empty space inside bounding box
    """
    total = 0.0
    for cells in cand.placement.values():
        if not cells:
            continue
        rs = [r for (r, _) in cells]
        cs = [c for (_, c) in cells]
        h = (max(rs) - min(rs) + 1)
        w = (max(cs) - min(cs) + 1)
        bbox_area = h * w
        bbox_perimeter = 2 * (h + w)
        area = len(cells)
        # connected components penalty
        k = _num_components(cells)
        comp_penalty = max(0, k - 1)
        # empty space inside bbox
        empty_frac = max(0.0, (bbox_area - area) / max(1, bbox_area))
        total += (bbox_perimeter / max(1, area)) + comp_penalty + empty_frac
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
            dist = manhattan(ca, cb)
            if dist > 2:
                total += (dist - 2) / sample.grid_size  # normalized
    return total

def location_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """Pull room centroids toward their target cells (from metadata)."""
    total = 0.0
    for spec in sample.rooms:
        tgt = spec.target_cell
        cells = cand.placement.get(spec.name, [])
        if not tgt or not cells:
            continue
        ctr = centroid_of_cells(cells)
        dist = manhattan(ctr, tgt) / sample.grid_size
        total += dist
    return total

def _num_components(cells: List[Cell]) -> int:
    """Connected components count via DFS."""
    if not cells:
        return 0
    S = set(cells)
    visited = set()
    comps = 0
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for cell in list(S):
        if cell in visited:
            continue
        comps += 1
        stack = [cell]
        while stack:
            r,c = stack.pop()
            if (r,c) in visited:
                continue
            visited.add((r,c))
            for dr,dc in dirs:
                nb = (r+dr, c+dc)
                if nb in S and nb not in visited:
                    stack.append(nb)
    return comps

# ---------- main API ----------
def score_constraints(sample: GridSample, cand: CandidateLayout) -> ConstraintScores:
    g = sample.grid_size
    # normalize overlap by grid cells
    overlap = overlap_penalty(cand, g) / max(1, g * g)
    compact = compactness_penalty(cand) / max(1, len(cand.placement) or 1)
    area = area_penalty(sample, cand) / max(1, len(sample.rooms))
    loc = location_penalty(sample, cand) / max(1, len(sample.rooms))
    quad = quadrant_penalty(sample, cand) / max(1, g)
    adj = adjacency_penalty(sample, cand)
    return ConstraintScores(
        quadrant=quad,
        overlap=overlap,
        area=area,
        compactness=compact,
        adjacency=adj,
        location=loc,
    )
