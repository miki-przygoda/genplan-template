from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import numpy as np

Section = str

@dataclass
class RoomSpec:
    name: str
    section: Section
    min_cells: int

@dataclass
class GridSample:
    floor_id: int
    grid_size: int
    rooms: List[RoomSpec]
    target_mask: Optional[np.ndarray]


def _load_metadata(floor_dir: Path) -> dict:
    return json.loads((floor_dir / "metadata.json").read_text())

def _room_name(room: dict) -> str:
    return f"room_{room['room_id']}"

def _centroid_to_cell(centroid: Tuple[int,int], img_wh=(512,512), grid_size=4) -> Tuple[int,int]:
    x, y = centroid
    w, h = img_wh
    c = min(grid_size-1, max(0, int(x * grid_size / w)))
    r = min(grid_size-1, max(0, int(y * grid_size / h)))
    return (r, c)

def _cell_to_section(rc: Tuple[int,int], grid_size=4) -> Section:
    r, c = rc
    ns = "N" if r < grid_size//2 else "S"
    ew = "W" if c < grid_size//2 else "E"
    return f"{ns}{ew}"

def _area_to_min_cells(area_px: int, img_wh=(512,512), grid_size=4) -> int:
    total_px = img_wh[0]*img_wh[1]
    ratio = area_px / total_px
    if ratio < 0.04: return 1
    if ratio < 0.09: return 2
    return 3

def _text_section_override(name_to_text_section: Dict[str, Section], name: str, fallback: Section) -> Section:
    return name_to_text_section.get(name, fallback)

def _make_target_mask(centroid_cells: List[Tuple[int,int]], grid_size=4) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for (r,c) in centroid_cells:
        mask[r, c] = 1
    return mask

# ------------ main API ------------
def encode_floorplan_to_grid(
    floor_dir: Path,
    grid_size: int = 4,
    *,
    name_to_text_section: Optional[Dict[str, Section]] = None,  # from parser (optional)
) -> GridSample:
    """
    Build a 4x4 grid sample from a processed floor folder.
    Prefers text-derived sections if provided; otherwise uses centroid-based quadrants.
    """
    meta = _load_metadata(floor_dir)
    img_wh = (meta["image_size"]["width"], meta["image_size"]["height"])
    rooms = meta["rooms"]

    centroid_cells, specs = [], []
    text_sections = name_to_text_section or {}

    for rmeta in rooms:
        name = _room_name(rmeta)
        centroid = tuple(rmeta["centroid"])  # (x,y)
        rc = _centroid_to_cell(centroid, img_wh=img_wh, grid_size=grid_size)

        inferred_section = _cell_to_section(rc, grid_size=grid_size)
        section = _text_section_override(text_sections, name, inferred_section)

        min_cells = _area_to_min_cells(rmeta["area_px"], img_wh=img_wh, grid_size=grid_size)
        specs.append(RoomSpec(name=name, section=section, min_cells=min_cells))
        centroid_cells.append(rc)

    target_mask = _make_target_mask(centroid_cells, grid_size=grid_size)

    return GridSample(
        floor_id=int(meta["floor_id"]),
        grid_size=grid_size,
        rooms=specs,
        target_mask=target_mask,
    )