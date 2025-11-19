from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import numpy as np

from .vars import DEFAULT_GRID_SIZE

Section = str

@dataclass
class RoomSpec:
    name: str
    section: Section
    min_cells: int
    expected_cells: int | None = None
    target_cell: tuple[int, int] | None = None
    room_type: str | None = None  # optional label from metadata

@dataclass
class GridSample:
    floor_id: int
    grid_size: int
    rooms: List[RoomSpec]
    target_mask: Optional[np.ndarray]
    base: int = 4  # nested grid branching factor (per dimension)
    levels: int = 1  # how many nested 4x4 levels describe grid_size


def _load_metadata(floor_dir: Path) -> dict:
    return json.loads((floor_dir / "metadata.json").read_text())

def _room_name(room: dict) -> str:
    return f"room_{room['room_id']}"

def _centroid_to_cell(centroid: Tuple[int, int], img_wh=(512, 512), grid_size=DEFAULT_GRID_SIZE) -> Tuple[int, int]:
    x, y = centroid
    w, h = img_wh
    c = min(grid_size-1, max(0, int(x * grid_size / w)))
    r = min(grid_size-1, max(0, int(y * grid_size / h)))
    return (r, c)

def _cell_to_section(rc: Tuple[int, int], grid_size=DEFAULT_GRID_SIZE) -> Section:
    r, c = rc
    ns = "N" if r < grid_size//2 else "S"
    ew = "W" if c < grid_size//2 else "E"
    return f"{ns}{ew}"

def _area_to_min_cells(area_px: int, img_wh=(512, 512), grid_size=DEFAULT_GRID_SIZE) -> int:
    total_px = img_wh[0]*img_wh[1]
    ratio = area_px / total_px
    # Scale desired cells to grid resolution (e.g., 256 cells at 16x16).
    ideal = ratio * (grid_size * grid_size)
    target = max(3, int(round(ideal * 0.6)))  # enforce ~60% of ideal, min 3 (>=2x2 later)
    # Keep within reasonable bounds so a single room doesn't consume the entire grid.
    return min(max(4, target), max(4, grid_size * grid_size // 2))

def _text_section_override(name_to_text_section: Dict[str, Section], name: str, fallback: Section) -> Section:
    return name_to_text_section.get(name, fallback)

def _make_target_mask(specs: List[RoomSpec], grid_size=DEFAULT_GRID_SIZE) -> np.ndarray:
    """
    Build a richer target mask: for each room, paint a compact block around its target cell
    sized by expected/min cells (min 2x2). Helps visual comparison on 16x16 grids.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for spec in specs:
        rc = spec.target_cell or (grid_size // 2, grid_size // 2)
        size = max(2, int(round((spec.expected_cells or spec.min_cells or 4) ** 0.5)))
        half = size // 2
        r0 = max(0, rc[0] - half)
        c0 = max(0, rc[1] - half)
        r1 = min(grid_size - 1, r0 + size - 1)
        c1 = min(grid_size - 1, c0 + size - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mask[r, c] = 1
    return mask

def _levels_for_grid(grid_size: int, base: int) -> int:
    """Small helper to describe grid_size as nested base x base tiles (e.g., 16 -> base=4, levels=2)."""
    levels = 1
    size = base
    while size < grid_size and levels < 5:  # avoid runaway
        levels += 1
        size *= base
    return levels

def cell_to_nested(rc: Tuple[int, int], base: int, levels: int) -> list[Tuple[int, int]]:
    """Return hierarchical (r,c) tiles from coarse->fine for a flattened grid."""
    r, c = rc
    path: list[Tuple[int, int]] = []
    for _ in range(levels):
        path.append((r // base, c // base))
        r, c = r % base, c % base
    return path

# ------------ main API ------------
def encode_floorplan_to_grid(
    floor_dir: Path,
    grid_size: int = DEFAULT_GRID_SIZE,
    *,
    name_to_text_section: Optional[Dict[str, Section]] = None,  # from parser (optional)
) -> GridSample:
    """
    Build a grid sample from a processed floor folder (default 16x16 = nested 4x4 blocks).
    Prefers text-derived sections if provided; otherwise uses centroid-based quadrants.
    """
    meta = _load_metadata(floor_dir)
    img_wh = (meta["image_size"]["width"], meta["image_size"]["height"])
    rooms = meta["rooms"]

    centroid_cells, specs = [], []
    text_sections = name_to_text_section or {}
    levels = _levels_for_grid(grid_size, base=4)

    for rmeta in rooms:
        name = _room_name(rmeta)
        centroid = tuple(rmeta["centroid"])  # (x,y)
        rc = _centroid_to_cell(centroid, img_wh=img_wh, grid_size=grid_size)

        inferred_section = _cell_to_section(rc, grid_size=grid_size)
        section = _text_section_override(text_sections, name, inferred_section)

        min_cells = _area_to_min_cells(rmeta["area_px"], img_wh=img_wh, grid_size=grid_size)
        expected_cells = min_cells
        # If metadata provides a bounding box, refine expected size
        bbox = rmeta.get("bbox") or rmeta.get("bounding_box")
        if bbox and isinstance(bbox, dict) and "width" in bbox and "height" in bbox:
            px_area = bbox["width"] * bbox["height"]
            expected_cells = _area_to_min_cells(px_area, img_wh=img_wh, grid_size=grid_size)

        specs.append(
            RoomSpec(
                name=name,
                section=section,
                min_cells=min_cells,
                expected_cells=expected_cells,
                target_cell=rc,
                room_type=rmeta.get("room_type"),
            )
        )
        centroid_cells.append(rc)

    target_mask = _make_target_mask(specs, grid_size=grid_size)

    return GridSample(
        floor_id=int(meta["floor_id"]),
        grid_size=grid_size,
        base=4,
        levels=levels,
        rooms=specs,
        target_mask=target_mask,
    )
