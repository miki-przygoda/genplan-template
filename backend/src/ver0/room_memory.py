from __future__ import annotations

from .vars import DEFAULT_GRID_SIZE

BASE_GRID = 32
BASE_SIZES = {
    "bedroom": (40, 28),
    "bathroom": (20, 14),
    "living": (60, 40),
    "dining": (36, 24),
    "kitchen": (48, 32),
    "office": (36, 24),
    "garage": (80, 60),
    "storeroom": (24, 16),
    "other": (32, 20),
}

def room_size_for(room_type: str, grid_size: int = DEFAULT_GRID_SIZE) -> tuple[int, int]:
    exp, mn = BASE_SIZES.get(room_type, BASE_SIZES["other"])
    scale = (grid_size / BASE_GRID) ** 2
    return max(4, int(exp * scale)), max(4, int(mn * scale))
