import pytest
import json
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.src.ver0.grid_encoder import (
    encode_floorplan_to_grid,
    RoomSpec,
    GridSample,
    _centroid_to_cell,
    _cell_to_section,
    _area_to_min_cells,
    _make_target_mask,
    _text_section_override,
)


def test_centroid_to_cell():
    """Test centroid to cell conversion for different positions."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing centroid_to_cell with various positions...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Converting centroids to grid cells...")
    
    test_cases = [
        ((0, 0), (512, 512), 4, (0, 0)),
        ((256, 256), (512, 512), 4, (2, 2)),
        ((512, 512), (512, 512), 4, (3, 3)),
        ((128, 128), (512, 512), 4, (1, 1)),
        ((100, 400), (512, 512), 4, (3, 0)),
        ((400, 100), (512, 512), 4, (0, 3)),
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for centroid, img_wh, grid_size, expected in test_cases:
        result = _centroid_to_cell(centroid, img_wh=img_wh, grid_size=grid_size)
        print(f"  Centroid {centroid} -> Cell {result} (expected {expected})")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Centroid {centroid}: expected {expected}, got {result}")
        else:
            print(f"    ✓ Correct")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ All centroid to cell conversions correct")


def test_centroid_to_cell_boundary_conditions():
    """Test centroid to cell with boundary conditions and clamping."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing boundary conditions (values outside image bounds)...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Converting centroids that are outside image bounds...")
    
    test_cases = [
        ((-10, -10), (512, 512), 4, (0, 0)),
        ((600, 600), (512, 512), 4, (3, 3)),
        ((1000, 1000), (512, 512), 4, (3, 3)),
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for centroid, img_wh, grid_size, expected in test_cases:
        result = _centroid_to_cell(centroid, img_wh=img_wh, grid_size=grid_size)
        print(f"  Centroid {centroid} -> Cell {result} (expected {expected}, clamped)")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Centroid {centroid}: expected {expected}, got {result}")
        else:
            print(f"    ✓ Correctly clamped")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ All boundary conditions handled correctly")


def test_cell_to_section():
    """Test cell to section conversion for all quadrants."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing cell to section conversion for 4x4 grid...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Converting grid cells to sections...")
    
    test_cases = [
        ((0, 0), 4, "NW"),
        ((0, 1), 4, "NW"),
        ((0, 2), 4, "NE"),
        ((0, 3), 4, "NE"),
        ((1, 0), 4, "NW"),
        ((1, 1), 4, "NW"),
        ((1, 2), 4, "NE"),
        ((1, 3), 4, "NE"),
        ((2, 0), 4, "SW"),
        ((2, 1), 4, "SW"),
        ((2, 2), 4, "SE"),
        ((2, 3), 4, "SE"),
        ((3, 0), 4, "SW"),
        ((3, 1), 4, "SW"),
        ((3, 2), 4, "SE"),
        ((3, 3), 4, "SE"),
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for cell, grid_size, expected in test_cases:
        result = _cell_to_section(cell, grid_size=grid_size)
        print(f"  Cell {cell} -> Section '{result}' (expected '{expected}')")
        if result != expected:
            print(f"    ✗ Expected '{expected}', got '{result}'")
            failures.append(f"Cell {cell}: expected '{expected}', got '{result}'")
        else:
            print(f"✓ {result} is correct")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ All cell to section conversions correct")


def test_area_to_min_cells():
    """Test area to minimum cells calculation based on area ratios."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing area to min_cells conversion...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Converting room areas to minimum cell requirements...")
    
    img_wh = (512, 512)
    total_px = img_wh[0] * img_wh[1]  # 262144
    
    # Test cases: (area_px, expected_min_cells, description)
    test_cases = [
        (1000, 1, "Very small room (< 4%)"),  # ~0.38%
        (10000, 1, "Small room (< 4%)"),  # ~3.8%
        (11000, 2, "Medium-small room (4-9%)"),  # ~4.2%
        (20000, 2, "Medium room (4-9%)"),  # ~7.6%
        (24000, 3, "Large room (>= 9%)"),  # ~9.2%
        (50000, 3, "Very large room (>= 9%)"),  # ~19%
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for area_px, expected, description in test_cases:
        result = _area_to_min_cells(area_px, img_wh=img_wh, grid_size=4)
        ratio = area_px / total_px
        print(f"  Area {area_px}px ({ratio:.2%}) -> {result} cells (expected {expected}) - {description}")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Area {area_px}px: expected {expected}, got {result}")
        else:
            print(f"    ✓ Correct")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ All area to min_cells conversions correct")


def test_make_target_mask():
    """Test target mask creation from room polygons with fallbacks."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing target mask creation from polygons and fallback blocks...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Creating target masks from polygon data...")
    
    img_wh = (4, 4)
    specs = [
        RoomSpec(
            name="room_1",
            section="NW",
            min_cells=4,
            target_cell=(0, 0),
            polygon=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
        ),
        RoomSpec(
            name="room_2",
            section="SE",
            min_cells=4,
            target_cell=(2, 2),
            polygon=[(2.1, 2.1), (2.9, 2.1), (2.9, 2.9), (2.1, 2.9)],
        ),
        RoomSpec(  # No polygon -> fallback block
            name="room_3",
            section="S",
            min_cells=4,
            target_cell=(3, 0),
            polygon=None,
        ),
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    result = _make_target_mask(specs, grid_size=4, img_wh=img_wh, rotate_k=0)
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0, 0] = 1
    expected[2, 2] = 1
    expected[2:4, 0:2] = 1  # fallback block for room_3
    print(f"Mask shape {result.shape}")
    print(f"Mask:\n{result}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if not np.array_equal(result, expected):
        print("  ✗ Masks don't match expected output")
        print(f"    Expected:\n{expected}")
        pytest.fail("Target mask does not match expected output")
    else:
        print("✓ Target mask created correctly")


def test_text_section_override():
    """Test text section override functionality."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing text section override...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Testing override with provided text sections...")
    
    name_to_text_section = {
        "room_1": "NE",
        "room_2": "SW",
    }
    
    test_cases = [
        ("room_1", "NW", "NE"),  # Should override NW with NE
        ("room_2", "SE", "SW"),  # Should override SE with SW
        ("room_3", "NW", "NW"),  # No override, should use fallback
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for name, fallback, expected in test_cases:
        result = _text_section_override(name_to_text_section, name, fallback)
        print(f"  Room '{name}' with fallback '{fallback}' -> '{result}' (expected '{expected}')")
        if result != expected:
            print(f"    ✗ Expected '{expected}', got '{result}'")
            failures.append(f"Room '{name}': expected '{expected}', got '{result}'")
        else:
            print(f"✓ {result} is correct")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("✓ All text section overrides work correctly")


def test_encode_floorplan_to_grid_basic():
    """Test basic encoding of floorplan to grid without text sections."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing basic encode_floorplan_to_grid without text sections...")
    
    with TemporaryDirectory() as tmpdir:
        floor_dir = Path(tmpdir)
        
        # Create mock metadata.json
        metadata = {
            "floor_id": 1,
            "image_size": {"width": 512, "height": 512},
            "rooms": [
                {
                    "room_id": 1,
                    "centroid": [128, 128],  # Should map to cell (1, 1) -> NW
                    "area_px": 15000,  # ~5.7% -> 2 min_cells
                    "polygon": [
                        [140, 140],
                        [180, 140],
                        [180, 180],
                        [140, 180],
                    ],
                },
                {
                    "room_id": 2,
                    "centroid": [384, 384],  # Should map to cell (3, 3) -> SE
                    "area_px": 25000,  # ~9.5% -> 3 min_cells
                    "polygon": [
                        [400, 400],
                        [440, 400],
                        [440, 440],
                        [400, 440],
                    ],
                },
            ],
        }
        
        (floor_dir / "metadata.json").write_text(json.dumps(metadata))
        
        print("\n" + "=" * 60)
        print("PROCESS")
        print("=" * 60)
        print("Encoding floorplan to grid...")
        result = encode_floorplan_to_grid(floor_dir, grid_size=4)
        
        print("\n" + "=" * 60)
        print("OUTPUT")
        print("=" * 60)
        print(f"Floor ID: {result.floor_id}")
        print(f"Grid Size: {result.grid_size}")
        print(f"Number of rooms: {len(result.rooms)}")
        for room in result.rooms:
            print(f"  {room.name}: section={room.section}, min_cells={room.min_cells}")
        print(f"Target mask shape: {result.target_mask.shape}")
        print(f"Target mask:\n{result.target_mask}")
        
        print("\n" + "=" * 60)
        print("EXPECTED")
        print("=" * 60)
        print("  floor_id: 1")
        print("  grid_size: 4")
        print("  room_1: section='NW', min_cells=2")
        print("  room_2: section='SE', min_cells=3")
        print("  target_mask: should have 1s at cells (1,1) and (3,3)")
        
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        failures = []
        
        if result.floor_id != 1:
            print(f"  ✗ floor_id: expected 1, got {result.floor_id}")
            failures.append(f"floor_id: expected 1, got {result.floor_id}")
        else:
            print(f"  ✓ floor_id is 1")
        
        if result.grid_size != 4:
            print(f"  ✗ grid_size: expected 4, got {result.grid_size}")
            failures.append(f"grid_size: expected 4, got {result.grid_size}")
        else:
            print(f"  ✓ grid_size is 4")
        
        if len(result.rooms) != 2:
            print(f"  ✗ Number of rooms: expected 2, got {len(result.rooms)}")
            failures.append(f"Number of rooms: expected 2, got {len(result.rooms)}")
        else:
            print(f"  ✓ Number of rooms is 2")
        
        room_1 = next((r for r in result.rooms if r.name == "room_1"), None)
        if room_1:
            if room_1.section != "NW":
                print(f"  ✗ room_1 section: expected 'NW', got '{room_1.section}'")
                failures.append(f"room_1 section: expected 'NW', got '{room_1.section}'")
            else:
                print(f"  ✓ room_1 section is 'NW'")
            
            if room_1.min_cells != 2:
                print(f"  ✗ room_1 min_cells: expected 2, got {room_1.min_cells}")
                failures.append(f"room_1 min_cells: expected 2, got {room_1.min_cells}")
            else:
                print(f"  ✓ room_1 min_cells is 2")
        else:
            print("  ✗ room_1 not found")
            failures.append("room_1 not found")
        
        room_2 = next((r for r in result.rooms if r.name == "room_2"), None)
        if room_2:
            if room_2.section != "SE":
                print(f"  ✗ room_2 section: expected 'SE', got '{room_2.section}'")
                failures.append(f"room_2 section: expected 'SE', got '{room_2.section}'")
            else:
                print(f"  ✓ room_2 section is 'SE'")
            
            if room_2.min_cells != 3:
                print(f"  ✗ room_2 min_cells: expected 3, got {room_2.min_cells}")
                failures.append(f"room_2 min_cells: expected 3, got {room_2.min_cells}")
            else:
                print(f"  ✓ room_2 min_cells is 3")
        else:
            print("  ✗ room_2 not found")
            failures.append("room_2 not found")
        
        # Check target mask
        expected_mask = np.zeros((4, 4), dtype=np.uint8)
        expected_mask[1, 1] = 1  # room_1 at (1, 1)
        expected_mask[3, 3] = 1  # room_2 at (3, 3)
        
        if not np.array_equal(result.target_mask, expected_mask):
            print(f"  ✗ target_mask doesn't match expected")
            print(f"    Expected:\n{expected_mask}")
            print(f"    Got:\n{result.target_mask}")
            failures.append("target_mask doesn't match expected")
        else:
            print(f"  ✓ target_mask is correct")
        
        if failures:
            print("\n" + "-" * 60)
            print("SUMMARY OF ISSUES")
            print("-" * 60)
            for i, failure in enumerate(failures, 1):
                print(f"  {i}. {failure}")
            print("-" * 60)
            pytest.fail(f"Test had {len(failures)} issue(s): {'; '.join(failures)}")


def test_encode_floorplan_to_grid_with_text_sections():
    """Test encoding with text section overrides."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing encode_floorplan_to_grid with text section overrides...")
    
    with TemporaryDirectory() as tmpdir:
        floor_dir = Path(tmpdir)
        
        # Create mock metadata.json
        metadata = {
            "floor_id": 2,
            "image_size": {"width": 512, "height": 512},
            "rooms": [
                {
                    "room_id": 1,
                    "centroid": [128, 128],  # Would normally map to NW
                    "area_px": 15000,
                },
                {
                    "room_id": 2,
                    "centroid": [384, 384],  # Would normally map to SE
                    "area_px": 25000,
                },
            ],
        }
        
        (floor_dir / "metadata.json").write_text(json.dumps(metadata))
        
        # Provide text section overrides
        name_to_text_section = {
            "room_1": "NE",  # Override NW -> NE
            "room_2": "SW",  # Override SE -> SW
        }
        
        print("\n" + "=" * 60)
        print("PROCESS")
        print("=" * 60)
        print("Encoding floorplan with text section overrides...")
        result = encode_floorplan_to_grid(
            floor_dir, 
            grid_size=4, 
            name_to_text_section=name_to_text_section
        )
        
        print("\n" + "=" * 60)
        print("OUTPUT")
        print("=" * 60)
        print(f"Number of rooms: {len(result.rooms)}")
        for room in result.rooms:
            print(f"  {room.name}: section={room.section}, min_cells={room.min_cells}")
        
        print("\n" + "=" * 60)
        print("EXPECTED")
        print("=" * 60)
        print("  room_1: section should be 'NE' (overridden from NW)")
        print("  room_2: section should be 'SW' (overridden from SE)")
        
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        failures = []
        
        room_1 = next((r for r in result.rooms if r.name == "room_1"), None)
        if room_1:
            if room_1.section != "NE":
                print(f"  ✗ room_1 section: expected 'NE' (overridden), got '{room_1.section}'")
                failures.append(f"room_1 section: expected 'NE', got '{room_1.section}'")
            else:
                print(f"  ✓ room_1 section is 'NE' (correctly overridden)")
        else:
            print("  ✗ room_1 not found")
            failures.append("room_1 not found")
        
        room_2 = next((r for r in result.rooms if r.name == "room_2"), None)
        if room_2:
            if room_2.section != "SW":
                print(f"  ✗ room_2 section: expected 'SW' (overridden), got '{room_2.section}'")
                failures.append(f"room_2 section: expected 'SW', got '{room_2.section}'")
            else:
                print(f"  ✓ room_2 section is 'SW' (correctly overridden)")
        else:
            print("  ✗ room_2 not found")
            failures.append("room_2 not found")
        
        if failures:
            print("\n" + "-" * 60)
            print("SUMMARY OF ISSUES")
            print("-" * 60)
            for i, failure in enumerate(failures, 1):
                print(f"  {i}. {failure}")
            print("-" * 60)
            pytest.fail(f"Test had {len(failures)} issue(s): {'; '.join(failures)}")


def test_encode_floorplan_to_grid_empty_rooms():
    """Test encoding with empty rooms list."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing encode_floorplan_to_grid with empty rooms...")
    
    with TemporaryDirectory() as tmpdir:
        floor_dir = Path(tmpdir)
        
        # Create mock metadata.json with no rooms
        metadata = {
            "floor_id": 3,
            "image_size": {"width": 512, "height": 512},
            "rooms": [],
        }
        
        (floor_dir / "metadata.json").write_text(json.dumps(metadata))
        
        print("\n" + "=" * 60)
        print("PROCESS")
        print("=" * 60)
        print("Encoding floorplan with no rooms...")
        result = encode_floorplan_to_grid(floor_dir, grid_size=4)
        
        print("\n" + "=" * 60)
        print("OUTPUT")
        print("=" * 60)
        print(f"Number of rooms: {len(result.rooms)}")
        print(f"Target mask shape: {result.target_mask.shape}")
        print(f"Target mask:\n{result.target_mask}")
        
        print("\n" + "=" * 60)
        print("EXPECTED")
        print("=" * 60)
        print("  Number of rooms: 0")
        print("  Target mask: all zeros")
        
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        failures = []
        
        if len(result.rooms) != 0:
            print(f"  ✗ Number of rooms: expected 0, got {len(result.rooms)}")
            failures.append(f"Number of rooms: expected 0, got {len(result.rooms)}")
        else:
            print(f"  ✓ Number of rooms is 0")
        
        expected_mask = np.zeros((4, 4), dtype=np.uint8)
        if not np.array_equal(result.target_mask, expected_mask):
            print(f"  ✗ target_mask should be all zeros")
            failures.append("target_mask should be all zeros")
        else:
            print(f"  ✓ target_mask is all zeros (correct)")
        
        if failures:
            print("\n" + "-" * 60)
            print("SUMMARY OF ISSUES")
            print("-" * 60)
            for i, failure in enumerate(failures, 1):
                print(f"  {i}. {failure}")
            print("-" * 60)
            pytest.fail(f"Test had {len(failures)} issue(s): {'; '.join(failures)}")


def test_encode_floorplan_to_grid_custom_grid_size():
    """Test encoding with custom grid size."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing encode_floorplan_to_grid with custom grid size (8x8)...")
    
    with TemporaryDirectory() as tmpdir:
        floor_dir = Path(tmpdir)
        
        # Create mock metadata.json
        metadata = {
            "floor_id": 4,
            "image_size": {"width": 512, "height": 512},
            "rooms": [
                {
                    "room_id": 1,
                    "centroid": [256, 256],  # Center
                    "area_px": 20000,
                },
            ],
        }
        
        (floor_dir / "metadata.json").write_text(json.dumps(metadata))
        
        print("\n" + "=" * 60)
        print("PROCESS")
        print("=" * 60)
        print("Encoding floorplan with grid_size=8...")
        result = encode_floorplan_to_grid(floor_dir, grid_size=8)
        
        print("\n" + "=" * 60)
        print("OUTPUT")
        print("=" * 60)
        print(f"Grid Size: {result.grid_size}")
        print(f"Target mask shape: {result.target_mask.shape}")
        print(f"Number of rooms: {len(result.rooms)}")
        
        print("\n" + "=" * 60)
        print("EXPECTED")
        print("=" * 60)
        print("  grid_size: 8")
        print("  target_mask shape: (8, 8)")
        
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        failures = []
        
        if result.grid_size != 8:
            print(f"  ✗ grid_size: expected 8, got {result.grid_size}")
            failures.append(f"grid_size: expected 8, got {result.grid_size}")
        else:
            print(f"  ✓ grid_size is 8")
        
        if result.target_mask.shape != (8, 8):
            print(f"  ✗ target_mask shape: expected (8, 8), got {result.target_mask.shape}")
            failures.append(f"target_mask shape: expected (8, 8), got {result.target_mask.shape}")
        else:
            print(f"  ✓ target_mask shape is (8, 8)")
        
        if failures:
            print("\n" + "-" * 60)
            print("SUMMARY OF ISSUES")
            print("-" * 60)
            for i, failure in enumerate(failures, 1):
                print(f"  {i}. {failure}")
            print("-" * 60)
            pytest.fail(f"Test had {len(failures)} issue(s): {'; '.join(failures)}")
