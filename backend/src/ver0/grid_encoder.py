from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import numpy as np
from PIL import Image, ImageDraw

from .vars import DEFAULT_GRID_SIZE, ROTATE_IMAGE_K
from .room_memory import room_size_for

Section = str

@dataclass
class RoomSpec:
    name: str
    section: Section
    min_cells: int
    expected_cells: int | None = None
    target_cell: tuple[int, int] | None = None
    room_type: str | None = None  # optional label from metadata
    polygon: Optional[list[tuple[float, float]]] = None  # room contour in image coords
    is_active: bool = True

@dataclass
class GridSample:
    floor_id: int
    grid_size: int
    rooms: List[RoomSpec]
    target_mask: Optional[np.ndarray]
    base: int = 4  # nested grid branching factor (per dimension)
    levels: int = 1  # how many nested 4x4 levels describe grid_size
    text_room_total: int | None = None
    text_section_counts: dict[str, int] | None = None
    active_room_names: set[str] | None = None
    active_room_target: int | None = None


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

def _section_to_cell(section: Section | None, grid_size: int = DEFAULT_GRID_SIZE) -> Tuple[int, int]:
    quarter = grid_size // 4
    centers = {
        "NW": (quarter, quarter),
        "NE": (quarter, grid_size - 1 - quarter),
        "SW": (grid_size - 1 - quarter, quarter),
        "SE": (grid_size - 1 - quarter, grid_size - 1 - quarter),
        "N": (quarter, grid_size // 2),
        "S": (grid_size - 1 - quarter, grid_size // 2),
        "E": (grid_size // 2, grid_size - 1 - quarter),
        "W": (grid_size // 2, quarter),
        "C": (grid_size // 2, grid_size // 2),
    }
    return centers.get(section or "C", centers["C"])

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _rotate_point(pt: Tuple[int, int], img_wh=(512, 512), k: int = 0) -> Tuple[int, int]:
    """Rotate (x,y) in image coords by k * 90 degrees clockwise."""
    x, y = pt
    w, h = img_wh
    k = k % 4
    if k == 0:
        return (x, y)
    if k == 1:  # 90 deg CW
        return (h - y, x)
    if k == 2:  # 180
        return (w - x, h - y)
    if k == 3:  # 270 CW
        return (y, w - x)
    return (x, y)

def _area_to_min_cells(area_px: int, img_wh=(512, 512), grid_size=DEFAULT_GRID_SIZE) -> int:
    total_px = img_wh[0]*img_wh[1]
    ratio = area_px / total_px
    # Scale desired cells to grid resolution (e.g., 1024 cells at 32x32).
    ideal = ratio * (grid_size * grid_size)
    target = max(3, int(round(ideal * 0.6)))  # enforce ~60% of ideal, min 3 (>=2x2 later)
    # Keep within reasonable bounds so a single room doesn't consume the entire grid.
    return min(max(4, target), max(4, grid_size * grid_size // 2))

def _text_section_override(name_to_text_section: Dict[str, Section], name: str, fallback: Section) -> Section:
    return name_to_text_section.get(name, fallback)

def _scale_rotate_polygon(
    polygon: list[tuple[float, float]],
    *,
    img_wh: tuple[int, int],
    grid_size: int,
    rotate_k: int,
) -> list[tuple[float, float]]:
    """Rotate polygon in image coords, then scale it into grid coordinates."""
    w, h = img_wh
    rotated = [_rotate_point((float(x), float(y)), img_wh=img_wh, k=rotate_k) for x, y in polygon]
    return [(x * grid_size / w, y * grid_size / h) for x, y in rotated]

def _fallback_block(mask: np.ndarray, rc: tuple[int, int], size: int) -> None:
    """Paint a simple block centred on rc into mask (in-place)."""
    grid_size = mask.shape[0]
    half = size // 2
    r0 = max(0, rc[0] - half)
    c0 = max(0, rc[1] - half)
    r1 = min(grid_size - 1, r0 + size - 1)
    c1 = min(grid_size - 1, c0 + size - 1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = 1

def _make_target_mask(
    specs: List[RoomSpec],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    img_wh: tuple[int, int] = (512, 512),
    rotate_k: int = 0,
) -> np.ndarray:
    """
    Build a target mask from true room polygons when available; fall back to compact
    blocks around a room's target cell if polygon data is missing.
    """
    mask_img = Image.new("L", (grid_size, grid_size), 0)
    draw = ImageDraw.Draw(mask_img)
    fallback_specs: list[RoomSpec] = []

    for spec in specs:
        polygon = spec.polygon
        if polygon and len(polygon) >= 3:
            scaled_poly = _scale_rotate_polygon(
                polygon,
                img_wh=img_wh,
                grid_size=grid_size,
                rotate_k=rotate_k,
            )
            draw.polygon(scaled_poly, outline=1, fill=1)
        else:
            fallback_specs.append(spec)

    mask = (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)

    # If some rooms have no polygon data, retain previous behaviour for them.
    for spec in fallback_specs:
        rc = spec.target_cell or (grid_size // 2, grid_size // 2)
        size = max(2, int(round((spec.expected_cells or spec.min_cells or 4) ** 0.5)))
        _fallback_block(mask, rc, size)

    return mask

def _levels_for_grid(grid_size: int, base: int) -> int:
    """Small helper to describe grid_size as nested base x base tiles (e.g., 32 -> base=4, levels=3)."""
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

# ---- support text helpers ----
def _parse_support_rooms(text: str | None) -> list:
    if not text:
        return []
    try:
        from .text_to_support_text import parse_supporting_text  # type: ignore
        rooms = parse_supporting_text(text)
    except Exception:
        return []
    return rooms or []

def _activate_rooms_from_text(specs: list[RoomSpec], text_rooms: list, grid_size: int) -> set[str]:
    if not text_rooms:
        for spec in specs:
            spec.is_active = True
        return {spec.name for spec in specs}
    for spec in specs:
        spec.is_active = False
    available = set(range(len(specs)))
    active_names: set[str] = set()
    for room in text_rooms:
        if not available:
            break
        section = getattr(room, "section", None)
        target_center = _section_to_cell(section, grid_size=grid_size)
        def _score(idx: int) -> int:
            rc = specs[idx].target_cell or target_center
            return _manhattan(rc, target_center)
        best_idx = min(available, key=_score)
        specs[best_idx].is_active = True
        active_names.add(specs[best_idx].name)
        available.remove(best_idx)
    for idx in available:
        specs[idx].is_active = False
        specs[idx].min_cells = 0
        specs[idx].expected_cells = 0
    return active_names

def _rooms_from_text_only(text: str, grid_size: int) -> tuple[list[RoomSpec], set[str], int | None]:
    text_rooms = _parse_support_rooms(text)
    specs: list[RoomSpec] = []
    active_names: set[str] = set()
    for idx, room in enumerate(text_rooms):
        exp, mn = room_size_for(room.room_type, grid_size)
        name = f"{room.room_type}_{room.ordinal or idx + 1}"
        section = room.section or "C"
        target_cell = _section_to_cell(section, grid_size=grid_size)
        specs.append(
            RoomSpec(
                name=name,
                section=section,
                min_cells=mn,
                expected_cells=exp,
                target_cell=target_cell,
                room_type=room.room_type,
                polygon=None,
                is_active=True,
            )
        )
        active_names.add(name)
    active_target = len(active_names) if active_names else None
    return specs, active_names, active_target

# ------------ main API ------------
def encode_floorplan_to_grid(
    floor_dir: Path,
    grid_size: int = DEFAULT_GRID_SIZE,
    *,
    name_to_text_section: Optional[Dict[str, Section]] = None,  # from parser (optional)
    rotate_k: int = ROTATE_IMAGE_K,  # multiples of 90 deg clockwise
    text_override: str | None = None,
) -> GridSample:
    """
    Build a grid sample from support text only; metadata geometry is ignored to avoid leakage.
    """
    meta = None
    if text_override:
        support_text = text_override
        floor_id = -1
    else:
        meta = _load_metadata(floor_dir)
        support_text = (
            meta.get("supporting_text")
            or meta.get("support_text")
            or meta.get("text")
            or meta.get("scene_description")
        )
        floor_id = int(meta.get("floor_id", -1)) if meta else -1

    text_rooms = _parse_support_rooms(support_text)
    section_counts = Counter(room.section for room in text_rooms if getattr(room, "section", None))
    specs, active_names, active_target = _rooms_from_text_only(support_text or "", grid_size)
    levels = _levels_for_grid(grid_size, base=4)

    return GridSample(
        floor_id=floor_id,
        grid_size=grid_size,
        base=4,
        levels=levels,
        rooms=specs,
        target_mask=None,
        text_room_total=len(text_rooms) or None,
        text_section_counts=dict(section_counts) if section_counts else None,
        active_room_names=active_names,
        active_room_target=active_target,
    )


def load_target_mask(floor_dir: Path, grid_size: int = DEFAULT_GRID_SIZE, rotate_k: int = ROTATE_IMAGE_K) -> Optional[np.ndarray]:
    if not floor_dir.exists():
        return None
    meta = _load_metadata(floor_dir)
    img_wh = (meta["image_size"]["width"], meta["image_size"]["height"])
    rooms = meta["rooms"]
    specs: list[RoomSpec] = []
    for rmeta in rooms:
        name = _room_name(rmeta)
        centroid = tuple(rmeta["centroid"])
        rot_centroid = _rotate_point(centroid, img_wh=img_wh, k=rotate_k)
        rc = _centroid_to_cell(rot_centroid, img_wh=img_wh, grid_size=grid_size)
        min_cells = _area_to_min_cells(rmeta["area_px"], img_wh=img_wh, grid_size=grid_size)
        expected_cells = min_cells
        bbox = rmeta.get("bbox") or rmeta.get("bounding_box")
        if bbox and isinstance(bbox, dict) and "width" in bbox and "height" in bbox:
            px_area = bbox["width"] * bbox["height"]
            expected_cells = _area_to_min_cells(px_area, img_wh=img_wh, grid_size=grid_size)
        polygon = None
        raw_polygon = rmeta.get("polygon")
        if raw_polygon and isinstance(raw_polygon, list):
            polygon = [tuple(p) for p in raw_polygon if isinstance(p, (list, tuple)) and len(p) == 2]
        specs.append(
            RoomSpec(
                name=name,
                section=None,
                min_cells=min_cells,
                expected_cells=expected_cells,
                target_cell=rc,
                room_type=rmeta.get("room_type"),
                polygon=polygon,
            )
        )
    return _make_target_mask(
        specs,
        grid_size=grid_size,
        img_wh=img_wh,
        rotate_k=rotate_k,
    )
