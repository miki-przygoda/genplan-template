from __future__ import annotations

from typing import Dict, List, Optional
import random
from collections import deque

from .constraints import CandidateLayout, Cell, centroid_of_cells, manhattan


def _infer_grid_size(layout: CandidateLayout, fallback: int = 64) -> int:
    """Best-effort grid size guess from existing cell coordinates."""
    coords = [rc for cells in layout.placement.values() for rc in cells]
    if not coords:
        return fallback
    max_rc = max(max(r, c) for r, c in coords)
    return max(fallback, max_rc + 1)


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _attempt_blob_shift(
    layout: CandidateLayout,
    room: str,
    cells: list[Cell],
    dr: int,
    dc: int,
    grid_size: int,
    mask=None,
) -> list[Cell] | None:
    """Try to translate an entire room by (dr, dc). Fail if it would collide or leave bounds/mask."""
    if dr == 0 and dc == 0:
        return None
    if mask is None:
        room_masks = getattr(layout, "room_masks", None)
        mask = room_masks.get(room) if isinstance(room_masks, dict) else None
    occupied_other = {
        rc for rname, rc_list in layout.placement.items() if rname != room for rc in rc_list
    }
    shifted: list[Cell] = []
    for (r, c) in cells:
        nr = _clamp(r + dr, 0, grid_size - 1)
        nc = _clamp(c + dc, 0, grid_size - 1)
        new_rc = (nr, nc)
        if new_rc in occupied_other:
            return None
        if mask is not None:
            if nr >= mask.shape[0] or nc >= mask.shape[1]:
                return None
            if mask[nr, nc] == 0:
                return None
        shifted.append(new_rc)
    return shifted


def _resolve_overlaps(layout: CandidateLayout, grid_size: int) -> None:
    """Remove duplicate occupancy by trimming from larger rooms first."""
    room_sizes = {room: len(cells) for room, cells in layout.placement.items()}
    occ: dict[Cell, list[str]] = {}
    for room, cells in layout.placement.items():
        for rc in cells:
            if 0 <= rc[0] < grid_size and 0 <= rc[1] < grid_size:
                occ.setdefault(rc, []).append(room)
    for rc, rooms in occ.items():
        if len(rooms) <= 1:
            continue
        rooms_sorted = sorted(rooms, key=lambda r: room_sizes.get(r, 0))
        keeper = rooms_sorted[0]
        for room in rooms_sorted[1:]:
            cells = layout.placement.get(room, [])
            if rc in cells:
                cells.remove(rc)


DIR_VECTORS = {
    "north_of": (-1, 0),
    "south_of": (1, 0),
    "east_of": (0, 1),
    "west_of": (0, -1),
}


def _relation_step(anchor_ctr: Cell, mobile_ctr: Cell, rel_vec: Cell) -> tuple[int, int]:
    desired_r = anchor_ctr[0] + rel_vec[0]
    desired_c = anchor_ctr[1] + rel_vec[1]
    dr = _clamp(desired_r - mobile_ctr[0], -1, 1)
    dc = _clamp(desired_c - mobile_ctr[1], -1, 1)
    if dr == 0 and dc == 0:
        dr, dc = rel_vec[0], rel_vec[1]
    return dr, dc


def relation_based_mutation(layout: CandidateLayout, rng: random.Random, mutation_rate: float) -> None:
    """Pull related room pairs together in the correct compass direction."""
    relationships = getattr(layout, "relationships", None)
    if not relationships or rng.random() > mutation_rate:
        return
    # collect (mobile, rel, anchor)
    pairs: list[tuple[str, str, str]] = []
    for mobile, rels in relationships.items():
        for rel_type, anchor in rels:
            if rel_type in DIR_VECTORS:
                pairs.append((mobile, rel_type, anchor))
    if not pairs:
        return
    mobile, rel_type, anchor = rng.choice(pairs)
    mobile_cells = layout.placement.get(mobile, [])
    anchor_cells = layout.placement.get(anchor, [])
    if not mobile_cells or not anchor_cells:
        return
    # anchor = bigger room
    if len(mobile_cells) > len(anchor_cells):
        mobile, anchor = anchor, mobile
        mobile_cells = layout.placement.get(mobile, [])
        anchor_cells = layout.placement.get(anchor, [])
        # invert direction
        inv = {"north_of": "south_of", "south_of": "north_of", "east_of": "west_of", "west_of": "east_of"}
        rel_type = inv.get(rel_type, rel_type)
    rel_vec = DIR_VECTORS.get(rel_type)
    if rel_vec is None:
        return
    grid_size = _infer_grid_size(layout)
    m_ctr = centroid_of_cells(mobile_cells)
    a_ctr = centroid_of_cells(anchor_cells)
    dr, dc = _relation_step(a_ctr, m_ctr, rel_vec)
    new_cells = _attempt_blob_shift(layout, mobile, mobile_cells, dr, dc, grid_size)
    if new_cells:
        layout.placement[mobile] = new_cells


def mutate(layout: CandidateLayout, rng: random.Random, mutation_rate: float) -> None:
    """
    Section-aware mutation that nudges cells around target centres, blob-shifts rooms, and trims overlaps.
    """
    grid_size = _infer_grid_size(layout)
    allowed = layout.active_rooms
    targets = layout.target_cells or {}
    room_masks = getattr(layout, "room_masks", None)

    def _sample_near(room: str) -> Cell:
        """
        Pick a cell near existing geometry.
        Bias toward neighbors of current cells; otherwise sample within a
        small window around the room's bounding box (fallback to target cell window).
        """
        cells = layout.placement.get(room, [])
        mask = room_masks.get(room) if isinstance(room_masks, dict) else None
        if cells:
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            span_r = rmax - rmin + 1
            span_c = cmax - cmin + 1
            # Tight window around current bbox
            pad = max(1, min(span_r, span_c) // 3)
            rr0 = max(0, rmin - pad)
            rr1 = min(grid_size - 1, rmax + pad)
            cc0 = max(0, cmin - pad)
            cc1 = min(grid_size - 1, cmax + pad)
            # First, try a neighbor of existing geometry
            neighbor_pool: list[Cell] = []
            for rc in cells:
                for nb in _neighbors(rc, grid_size):
                    if nb in cells or nb in neighbor_pool:
                        continue
                    if mask is not None:
                        r, c = nb
                        if r >= mask.shape[0] or c >= mask.shape[1] or mask[r, c] == 0:
                            continue
                    neighbor_pool.append(nb)
            if neighbor_pool and rng.random() < 0.85:
                return rng.choice(neighbor_pool)
            # Otherwise sample within the bbox window
            return (rng.randint(rr0, rr1), rng.randint(cc0, cc1))

        # No geometry yet: fall back to target cell window or global random
        center = targets.get(room)
        if center is None:
            return (rng.randrange(grid_size), rng.randrange(grid_size))
        half = max(1, grid_size // 16)
        r0 = max(0, center[0] - half)
        r1 = min(grid_size - 1, center[0] + half)
        c0 = max(0, center[1] - half)
        c1 = min(grid_size - 1, center[1] + half)
        return (rng.randint(r0, r1), rng.randint(c0, c1))

    for room_name, cells in layout.placement.items():
        if allowed is not None and room_name not in allowed:
            layout.placement[room_name] = []
            continue
        if not cells:
            continue
        # blob shift toward target
        if rng.random() < mutation_rate:
            target = targets.get(room_name)
            if target is not None:
                ctr = centroid_of_cells(cells)
                dr = _clamp(target[0] - ctr[0], -1, 1)
                dc = _clamp(target[1] - ctr[1], -1, 1)
                mask = room_masks.get(room_name) if isinstance(room_masks, dict) else None
                new_cells = _attempt_blob_shift(layout, room_name, cells, dr, dc, grid_size, mask=mask)
                if new_cells:
                    cells[:] = new_cells
        # single-cell jitter near target
        if rng.random() < mutation_rate:
            idx = rng.randrange(len(cells))
            cells[idx] = _sample_near(room_name)
        # small whole-room shift to preserve compactness
        if rng.random() < 0.25 * mutation_rate and len(cells) > 1:
            dr = rng.choice([-1, 0, 1])
            dc = rng.choice([-1, 0, 1])
            shifted = []
            for (r, c) in cells:
                nr = min(grid_size - 1, max(0, r + dr))
                nc = min(grid_size - 1, max(0, c + dc))
                shifted.append((nr, nc))
            layout.placement[room_name] = shifted
        if rng.random() < mutation_rate and len(cells) > 1:
            rng.shuffle(cells)
    # relationship-based tug after local tweaks
    relation_based_mutation(layout, rng, mutation_rate)
    _resolve_overlaps(layout, grid_size)


def grow_mutation(layout: CandidateLayout, rng: random.Random, target_size: Dict[str, int]) -> None:
    """
    Targeted growth/shrink: adjust rooms toward their target size near current shape.
    """
    grid_size = _infer_grid_size(layout)
    allowed = layout.active_rooms
    room_masks = getattr(layout, "room_masks", None)
    for room, cells in layout.placement.items():
        if allowed is not None and room not in allowed:
            layout.placement[room] = []
            continue
        tgt = target_size.get(room, None)
        if tgt is None:
            continue
        if not cells:
            continue
        mask = room_masks.get(room) if isinstance(room_masks, dict) else None
        if len(cells) < tgt:
            needed = tgt - len(cells)
            rs = [r for r,_ in cells]
            cs = [c for _,c in cells]
            rmin, rmax = max(0, min(rs)-1), min(grid_size-1, max(rs)+1)
            cmin, cmax = max(0, min(cs)-1), min(grid_size-1, max(cs)+1)
            candidates = [(r,c) for r in range(rmin, rmax+1) for c in range(cmin, cmax+1)]
            rng.shuffle(candidates)
            for rc in candidates:
                if len(cells) >= tgt:
                    break
                if rc in cells:
                    continue
                if mask is not None:
                    r,c = rc
                    if r >= mask.shape[0] or c >= mask.shape[1] or mask[r,c] == 0:
                        continue
                cells.append(rc)
        elif len(cells) > tgt:
            ctr = centroid_of_cells(cells)
            to_remove = len(cells) - tgt
            cells.sort(key=lambda rc: manhattan(rc, ctr), reverse=True)
            del cells[:to_remove]

def _neighbors(rc: Cell, grid_size: int) -> list[Cell]:
    r, c = rc
    neigh: list[Cell] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neigh.append((nr, nc))
    return neigh


def _connected_components(cells: list[Cell], grid_size: int) -> list[list[Cell]]:
    """
    Split a room's cells into 4-connected components.
    """
    if not cells:
        return []

    remaining = set(cells)
    components: list[list[Cell]] = []

    while remaining:
        start = remaining.pop()
        comp = [start]
        q: deque[Cell] = deque([start])

        while q:
            rc = q.popleft()
            for nb in _neighbors(rc, grid_size):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.append(nb)
                    q.append(nb)

        components.append(comp)

    return components

def _enforce_connected(
    layout: CandidateLayout,
    grid_size: int,
    target_size: Optional[Dict[str, int]] = None,
) -> None:
    """
    For each room, keep only the largest connected component and,
    if target_size is provided, regrow back toward the desired size
    around the kept blob (respecting masks if available).
    """
    room_masks = getattr(layout, "room_masks", None)

    for room, cells in layout.placement.items():
        if not cells:
            continue

        comps = _connected_components(cells, grid_size)
        if not comps:
            continue

        # 1) keep largest connected component
        largest = max(comps, key=len)
        new_cells = list(largest)

        # 2) optional regrow toward target size
        tgt = None if target_size is None else target_size.get(room)
        if tgt is not None and tgt > len(new_cells):
            mask = room_masks.get(room) if isinstance(room_masks, dict) else None
            needed = tgt - len(new_cells)
            frontier = set(new_cells)

            # simple ring-growth from the blob
            while needed > 0 and frontier:
                next_frontier: set[Cell] = set()
                for rc in frontier:
                    for nb in _neighbors(rc, grid_size):
                        if nb in new_cells:
                            continue
                        if mask is not None:
                            r, c = nb
                            if (
                                r >= mask.shape[0]
                                or c >= mask.shape[1]
                                or mask[r, c] == 0
                            ):
                                continue
                        new_cells.append(nb)
                        needed -= 1
                        next_frontier.add(nb)
                        if needed <= 0:
                            break
                    if needed <= 0:
                        break
                frontier = next_frontier

        layout.placement[room] = new_cells


def enforce_connected(
    layout: CandidateLayout, grid_size: Optional[int] = None, target_size: Optional[Dict[str, int]] = None
) -> None:
    """
    Public helper to enforce 4-connected rooms and optionally regrow toward targets.
    """
    gs = grid_size if grid_size is not None else _infer_grid_size(layout)
    _enforce_connected(layout, gs, target_size)
