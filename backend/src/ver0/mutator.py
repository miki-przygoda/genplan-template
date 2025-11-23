from __future__ import annotations

from typing import Dict, List, Optional
import random

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


def _attempt_blob_shift(room: str, cells: list[Cell], dr: int, dc: int, layout: CandidateLayout, grid_size: int) -> list[Cell] | None:
    """Try to translate an entire room by (dr, dc). Fail if it would collide or leave bounds/mask."""
    if dr == 0 and dc == 0:
        return None
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
    new_cells = _attempt_blob_shift(mobile, mobile_cells, dr, dc, layout, grid_size)
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

    def _attempt_blob_shift(room: str, cells: list[Cell], dr: int, dc: int) -> list[Cell] | None:
        if dr == 0 and dc == 0:
            return None
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

    def _sample_near(room: str) -> Cell:
        center = targets.get(room)
        if center is None:
            return (rng.randrange(grid_size), rng.randrange(grid_size))
        half = max(1, grid_size // 10)
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
                new_cells = _attempt_blob_shift(room_name, cells, dr, dc)
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
