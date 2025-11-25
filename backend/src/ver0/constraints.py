from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .grid_encoder import GridSample, RoomSpec, cell_to_nested
from .vars import DEFAULT_GRID_SIZE
from .text_to_support_text import section_to_cell, section_bounds

Cell = tuple[int, int]  # (r, c)

@dataclass
class CandidateLayout:
    """room_name -> list of (r, c) cells in the working grid."""
    placement: Dict[str, List[Cell]]
    active_rooms: set[str] | None = None
    target_cells: dict[str, Cell] | None = None
    relationships: dict[str, list[tuple[str, str]]] | None = None

@dataclass
class ConstraintScores:
    quadrant: float
    overlap: float
    area: float
    compactness: float
    adjacency: float
    location: float = 0.0
    section: float = 0.0
    dispersion: float = 0.0
    room_usage: float = 0.0
    budget: float = 0.0
    section_bbox: float = 0.0
    mask: float = 0.0
    relationships: float = 0.0
    holes: float = 0.0

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


def find_room_holes(cells: List[Cell]) -> List[List[Cell]]:
    """Return hole components (background regions fully enclosed by the room) in absolute coords."""
    if not cells:
        return []
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    grid = [[0] * w for _ in range(h)]
    for r, c in cells:
        grid[r - rmin][c - cmin] = 1

    visited = [[False] * w for _ in range(h)]
    holes: list[list[Cell]] = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for rr in range(h):
        for cc in range(w):
            if grid[rr][cc] == 1 or visited[rr][cc]:
                continue
            comp: list[tuple[int, int]] = []
            touches_border = False
            stack = [(rr, cc)]
            visited[rr][cc] = True
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_border = True
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes.append([(r + rmin, c + cmin) for r, c in comp])
    return holes

def hierarchical_reference(cand: CandidateLayout, grid_size: int = DEFAULT_GRID_SIZE, base: int = 4) -> dict[str, list[list[tuple[int, int]]]]:
    """
    Build a nested cell reference for each room: for every (r,c) return its hierarchical path
    across base x base tiles (e.g., 32x32 -> stacked 4x4 blocks).
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
    """Penalty for rooms whose centroid cell is outside target quadrant/half (distance to section centre)."""
    g = sample.grid_size
    total = 0.0
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
        cells = cand.placement.get(spec.name, [])
        if not cells:
            continue
        ctr = centroid_of_cells(cells)
        if not spec.section:
            continue
        section_center = section_to_cell(spec.section, grid_size=g)
        q = quadrant_of_cell(ctr, g)
        # Encourage exact quadrant matches first
        if spec.section in {"NW","NE","SW","SE"}:
            if q != spec.section:
                total += manhattan(ctr, section_center)
            continue
        # Cardinal/central sections also incur distance proportional penalty
        if spec.section in {"N", "S", "E", "W", "C"}:
            total += manhattan(ctr, section_center)
    return total

def section_alignment_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """Distance of each room centroid to the coarse section centre with a tolerance window."""
    total = 0.0
    g = sample.grid_size
    tolerance = max(1, g // 6)
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
        if not spec.section:
            continue
        cells = cand.placement.get(spec.name, [])
        if not cells:
            continue
        ctr = centroid_of_cells(cells)
        section_center = section_to_cell(spec.section, grid_size=g)
        dist = manhattan(ctr, section_center)
        total += max(0, dist - tolerance)
    return total / max(1, g)

def overlap_penalty(cand: CandidateLayout, grid_size: int = DEFAULT_GRID_SIZE) -> float:
    room_sizes = {room: len(cells) for room, cells in cand.placement.items()}
    occ_rooms: dict[tuple[int, int], list[str]] = {}
    for room, cells in cand.placement.items():
        for (r, c) in cells:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                occ_rooms.setdefault((r, c), []).append(room)
    total = 0.0
    for rc, rooms in occ_rooms.items():
        if len(rooms) <= 1:
            continue
        size_sum = sum(room_sizes.get(r, 0) for r in rooms)
        total += (len(rooms) - 1) * size_sum
    return float(total)

def area_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    total = 0.0
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
        assigned = len(cand.placement.get(spec.name, []))
        target = spec.expected_cells or spec.min_cells or 4
        target = max(4, target)  # enforce 2x2 minimum
        low = int(0.9 * target)
        high = int(1.1 * target)
        if low <= assigned <= high:
            continue
        delta = abs(assigned - target)
        total += (delta ** 2) / max(1, target)
    return float(total)

def compactness_penalty(cand: CandidateLayout) -> float:
    """
    Bounding-box perimeter / area plus fragmentation penalties:
    - penalize multiple components
    - penalize empty space inside bounding box
    """
    total = 0.0
    allowed = cand.active_rooms
    for room_name, cells in cand.placement.items():
        if allowed is not None and room_name not in allowed:
            continue
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
        comp_penalty = 2 * max(0, k - 1)  # heavier penalty for fragmentation
        # empty space inside bbox (scaled up to kill gaps)
        empty_frac = max(0.0, (bbox_area - area) / max(1, bbox_area))
        empty_penalty = 4 * empty_frac
        total += 1.5 * (bbox_perimeter / max(1, area)) + comp_penalty + empty_penalty
    return float(total)

def dispersion_penalty(cand: CandidateLayout, grid_size: int) -> float:
    """Penalise layouts whose cells are spread far from their centroid (random scatter)."""
    total = 0.0
    allowed = cand.active_rooms
    for room_name, cells in cand.placement.items():
        if allowed is not None and room_name not in allowed:
            continue
        if len(cells) <= 1:
            continue
        ctr = centroid_of_cells(cells)
        spread = sum(manhattan(cell, ctr) for cell in cells) / max(1, len(cells))
        total += spread / max(1, grid_size)
    return total

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

def relationship_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """
    Penalize violations of directional relationships extracted from text.
    Uses room centroids and normalizes by grid size to keep values comparable.
    """
    total = 0.0
    g = sample.grid_size
    centroids = {room: centroid_of_cells(cells) for room, cells in cand.placement.items() if cells}
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
        rels = getattr(spec, "relationships", None)
        if not rels:
            continue
        src_ctr = centroids.get(spec.name)
        if not src_ctr:
            continue
        for rel_type, target_name in rels:
            dst_ctr = centroids.get(target_name)
            if not dst_ctr:
                continue
            dr = src_ctr[0] - dst_ctr[0]
            dc = src_ctr[1] - dst_ctr[1]
            dir_pen = 0
            if rel_type == "north_of":
                dir_pen = max(0, dr + 1)
            elif rel_type == "south_of":
                dir_pen = max(0, -dr + 1)
            elif rel_type == "east_of":
                dir_pen = max(0, -dc + 1)
            elif rel_type == "west_of":
                dir_pen = max(0, dc + 1)
            dist_pen = max(0, manhattan(src_ctr, dst_ctr) - 1)
            total += (dir_pen + 0.25 * dist_pen) / max(1, g)
    return total

def location_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """Pull room centroids toward their target cells (from metadata)."""
    total = 0.0
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
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

def room_usage_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    """Penalty for using rooms beyond the text-derived quota or leaving required slots empty."""
    active_names = getattr(sample, "active_room_names", None)
    target = getattr(sample, "active_room_target", None)
    if not active_names or target is None:
        return 0.0
    used_active = 0
    inactive_used = 0
    for spec in sample.rooms:
        cells = cand.placement.get(spec.name, [])
        if not cells:
            continue
        if getattr(spec, "is_active", True):
            used_active += 1
        else:
            inactive_used += 1
    missing_active = max(0, min(target, len(active_names)) - used_active)
    return float(inactive_used + missing_active)

def budget_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    target_total = 0
    for spec in sample.rooms:
        if getattr(spec, "is_active", True):
            target_total += max(4, spec.expected_cells or spec.min_cells or 4)
    target_total = max(1, target_total)
    assigned_total = sum(len(cells) for cells in cand.placement.values())
    return abs(assigned_total - target_total) / target_total

def section_bbox_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    total = 0.0
    g = sample.grid_size
    for spec in sample.rooms:
        if not getattr(spec, "is_active", True):
            continue
        cells = cand.placement.get(spec.name, [])
        if not cells or not spec.section:
            continue
        r0, r1, c0, c1 = section_bounds(spec.section, grid_size=g, half_span=max(2, g // 6))
        inside = [(r, c) for (r, c) in cells if r0 <= r <= r1 and c0 <= c <= c1]
        if not inside:
            total += len(cells)
    return total

def mask_penalty(sample: GridSample, cand: CandidateLayout) -> float:
    mask = getattr(sample, "target_mask", None)
    if mask is None:
        return 0.0
    g = sample.grid_size
    layout = np.zeros((g, g), dtype=np.uint8)
    for cells in cand.placement.values():
        for (r, c) in cells:
            if 0 <= r < g and 0 <= c < g:
                layout[r, c] = 1
    missing = np.logical_and(mask == 1, layout == 0).sum()
    extra = np.logical_and(mask == 0, layout == 1).sum()
    return float(missing + extra)

# ---------- main API ----------
def score_constraints(sample: GridSample, cand: CandidateLayout) -> ConstraintScores:
    g = sample.grid_size
    # keep overlap as a raw count to strongly punish any collision
    overlap = float(overlap_penalty(cand, g))
    compact = compactness_penalty(cand)
    area = area_penalty(sample, cand)
    loc = location_penalty(sample, cand)
    quad = quadrant_penalty(sample, cand) / max(1, g)
    section = section_alignment_penalty(sample, cand)
    dispersion = dispersion_penalty(cand, g)
    room_usage = room_usage_penalty(sample, cand)
    budget = budget_penalty(sample, cand)
    section_b = section_bbox_penalty(sample, cand)
    mask = mask_penalty(sample, cand)
    adj = adjacency_penalty(sample, cand)
    holes = 0.0
    for _, cells in cand.placement.items():
        comps = find_room_holes(cells)
        if not comps:
            continue
        rs = [r for r, _ in cells] if cells else [0]
        cs = [c for _, c in cells] if cells else [0]
        bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        area_norm = len(cells) / max(1, bbox_area)
        holes += len(comps) * area_norm
    return ConstraintScores(
        quadrant=quad,
        overlap=overlap,
        area=area,
        compactness=compact,
        adjacency=adj,
        location=loc,
        section=section,
        dispersion=dispersion,
        room_usage=room_usage,
        budget=budget,
        section_bbox=section_b,
        mask=mask,
        relationships=relationship_penalty(sample, cand),
        holes=holes,
    )
