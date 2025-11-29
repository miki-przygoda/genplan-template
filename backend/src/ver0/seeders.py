from __future__ import annotations

"""
Seeder strategies for the EA. These can be plugged into RL/bandit policies to
choose how the initial population is created.

All seeders share the signature (sample, rng) -> CandidateLayout.
"""

import random
from typing import Dict, List
import numpy as np

from .evolver import make_random_layout, CandidateLayout, Cell, _infer_grid_size
from .constraints import centroid_of_cells, manhattan
from .text_to_support_text import section_bounds, section_to_cell


def section_seed(sample, rng: random.Random) -> CandidateLayout:
    """Default seeding strategy (section-aware large blocks)."""
    return make_random_layout(sample, rng)


def _cells_from_mask(mask: np.ndarray, desired: int, rng: random.Random) -> list[Cell]:
    coords = list(zip(*np.where(mask == 1)))
    rng.shuffle(coords)
    return coords[:desired]

def _layout_from_mask(sample, rng: random.Random, fill_ratio: float = 0.8) -> CandidateLayout:
    """Fill rooms by sampling from per-room masks or global mask; fallback to section-based if mask absent."""
    mask = getattr(sample, "target_mask", None)
    room_masks = getattr(sample, "room_masks", None) or {}
    if mask is None or not np.any(mask):
        return make_random_layout(sample, rng)

    grid_size = mask.shape[0]
    allowed = sample.active_room_names
    allowed_set = set(allowed) if allowed is not None else None
    placement: Dict[str, List[Cell]] = {spec.name: [] for spec in sample.rooms}
    target_cells: Dict[str, Cell] = {}

    coords = list(zip(*np.where(mask == 1)))
    rng.shuffle(coords)

    def _sample_cells_for_room(room_name: str, target_count: int, bounds: tuple[int, int, int, int]) -> list[Cell]:
        r0, r1, c0, c1 = bounds
        room_mask = room_masks.get(room_name)
        if room_mask is not None and np.any(room_mask):
            pool = _cells_from_mask(room_mask, target_count * 2, rng)
        else:
            pool = [(r, c) for (r, c) in coords if r0 <= r <= r1 and c0 <= c <= c1]
        if not pool:
            pool = coords
        rng.shuffle(pool)
        return pool[:target_count]

    ordered_specs = sorted(
        [s for s in sample.rooms if (allowed_set is None or s.name in allowed_set)],
        key=lambda s: s.expected_cells or s.min_cells,
        reverse=True,
    )
    for spec in ordered_specs:
        tgt = max(4, spec.expected_cells or spec.min_cells or 4)
        desired = int(tgt * fill_ratio)
        center = spec.target_cell or section_to_cell(spec.section, grid_size=grid_size)
        bounds = section_bounds(spec.section, grid_size=grid_size, half_span=max(3, grid_size // 4))
        cells = _sample_cells_for_room(spec.name, desired, bounds)
        if len(cells) < desired and coords:
            need = desired - len(cells)
            cells += coords[:need]
        placement[spec.name] = cells
        target_cells[spec.name] = center

    return CandidateLayout(
        placement=placement,
        active_rooms=allowed_set.copy() if allowed_set is not None else None,
        target_cells=target_cells,
    )


def mask_guided_seed(sample, rng: random.Random) -> CandidateLayout:
    """Seed heavily from the provided target mask (if available)."""
    return _layout_from_mask(sample, rng, fill_ratio=0.9)


def compact_seed(sample, rng: random.Random) -> CandidateLayout:
    """Seed with smaller, tightly packed regions near targets to diversify search."""
    mask = getattr(sample, "target_mask", None)
    if mask is not None and np.any(mask):
        return _layout_from_mask(sample, rng, fill_ratio=0.5)

    allowed = sample.active_room_names
    allowed_set = set(allowed) if allowed is not None else None
    placement: Dict[str, List[Cell]] = {spec.name: [] for spec in sample.rooms}
    target_cells: Dict[str, Cell] = {}
    for spec in sample.rooms:
        if allowed_set is not None and spec.name not in allowed_set:
            continue
        center = spec.target_cell or section_to_cell(spec.section, grid_size=sample.grid_size)
        target_cells[spec.name] = center
        target = max(4, spec.expected_cells or spec.min_cells or 4)
        desired = max(4, int(target * 0.6))
        bounds = section_bounds(spec.section, grid_size=sample.grid_size, half_span=max(2, sample.grid_size // 6))
        r0, r1, c0, c1 = bounds
        candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        rng.shuffle(candidates)
        placement[spec.name] = candidates[:desired]
    return CandidateLayout(
        placement=placement,
        active_rooms=allowed_set.copy() if allowed_set is not None else None,
        target_cells=target_cells,
    )

def section_tight_mask_seed(sample, rng):
    """Mask-aware seeding with compact sections; favors mask overlap."""
    return _layout_from_mask(sample, rng, fill_ratio=0.7)

def mask_sparse_seed(sample, rng):
    """Lightweight mask-aware seed (~50% size) to encourage exploration."""
    return _layout_from_mask(sample, rng, fill_ratio=0.5)


SEEDING_REGISTRY = {
    "section": section_seed,
    "mask_guided": mask_guided_seed,
    "compact": compact_seed,
    "section_tight_mask": section_tight_mask_seed,
    "mask_sparse": mask_sparse_seed,
}
