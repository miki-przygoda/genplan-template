import pytest
import numpy as np

from backend.src.ver0.constraints import (
    CandidateLayout,
    ConstraintScores,
    quadrant_of_cell,
    centroid_of_cells,
    perimeter_of_cells,
    manhattan,
    quadrant_penalty,
    overlap_penalty,
    area_penalty,
    compactness_penalty,
    adjacency_penalty,
    score_constraints,
)
from backend.src.ver0.grid_encoder import GridSample, RoomSpec


def test_quadrant_of_cell():
    """Test quadrant calculation for different cell positions."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing quadrant_of_cell for 4x4 grid...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating quadrants for various cells...")
    
    # For 4x4 grid: rows 0-1 are N, rows 2-3 are S; cols 0-1 are W, cols 2-3 are E
    test_cases = [
        ((0, 0), 4, "NW"),  # Top-left
        ((0, 1), 4, "NW"),  # Top-left area
        ((0, 2), 4, "NE"),  # Top-right area
        ((0, 3), 4, "NE"),  # Top-right
        ((1, 0), 4, "NW"),  # Upper-left
        ((1, 1), 4, "NW"),  # Upper-center-left
        ((1, 2), 4, "NE"),  # Upper-center-right
        ((1, 3), 4, "NE"),  # Upper-right
        ((2, 0), 4, "SW"),  # Lower-left
        ((2, 1), 4, "SW"),  # Lower-center-left
        ((2, 2), 4, "SE"),  # Lower-center-right
        ((2, 3), 4, "SE"),  # Lower-right
        ((3, 0), 4, "SW"),  # Bottom-left
        ((3, 1), 4, "SW"),  # Bottom-center-left
        ((3, 2), 4, "SE"),  # Bottom-center-right
        ((3, 3), 4, "SE"),  # Bottom-right
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for cell, grid_size, expected in test_cases:
        result = quadrant_of_cell(cell, grid_size)
        print(f"  Cell {cell} -> Quadrant '{result}' (expected '{expected}')")
        if result != expected:
            print(f"    ✗ Expected '{expected}', got '{result}'")
            failures.append(f"Cell {cell}: expected '{expected}', got '{result}'")
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
        print("  ✓ All quadrant calculations correct")


def test_centroid_of_cells():
    """Test centroid calculation for cell lists."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing centroid_of_cells with various cell groups...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating centroids...")
    
    test_cases = [
        ([(0, 0)], (0, 0)),  # Single cell
        ([(0, 0), (0, 1)], (0, 0)),  # Two cells horizontally: rows=(0+0)/2=0, cols=(0+1)/2=0.5→0
        ([(0, 0), (1, 0)], (0, 0)),  # Two cells vertically: rows=(0+1)/2=0.5→0, cols=(0+0)/2=0
        ([(0, 0), (0, 1), (1, 0), (1, 1)], (0, 0)),  # 2x2 square: rows=(0+0+1+1)/4=0.5→0, cols=(0+1+0+1)/4=0.5→0
        ([(1, 1), (1, 2), (2, 1), (2, 2)], (2, 2)),  # Another 2x2 square: rows=(1+1+2+2)/4=1.5→2, cols=(1+2+1+2)/4=1.5→2
        ([(0, 0), (0, 3), (3, 0), (3, 3)], (2, 2)),  # Four corners: rows=(0+0+3+3)/4=1.5→2, cols=(0+3+0+3)/4=1.5→2
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for cells, expected in test_cases:
        result = centroid_of_cells(cells)
        print(f"  Cells {cells} -> Centroid {result} (expected {expected})")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Cells {cells}: expected {expected}, got {result}")
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
        print("  ✓ All centroid calculations correct")


def test_perimeter_of_cells():
    """Test perimeter calculation for cell groups."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing perimeter_of_cells with various shapes...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating perimeters...")
    
    test_cases = [
        ([(0, 0)], 4),  # Single cell: 4 sides exposed
        ([(0, 0), (0, 1)], 6),  # Two cells horizontally: 6 sides exposed
        ([(0, 0), (1, 0)], 6),  # Two cells vertically: 6 sides exposed
        ([(0, 0), (0, 1), (1, 0), (1, 1)], 8),  # 2x2 square: 8 sides exposed
        ([(0, 0), (0, 1), (0, 2)], 8),  # Three cells in a row: 8 sides
        ([(1, 1), (1, 2), (2, 1)], 8),  # L-shape: 8 sides
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for cells, expected in test_cases:
        result = perimeter_of_cells(cells)
        print(f"  Cells {cells} -> Perimeter {result} (expected {expected})")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Cells {cells}: expected {expected}, got {result}")
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
        print("  ✓ All perimeter calculations correct")


def test_manhattan():
    """Test Manhattan distance calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing manhattan distance...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating Manhattan distances...")
    
    test_cases = [
        ((0, 0), (0, 0), 0),  # Same cell
        ((0, 0), (0, 1), 1),  # Adjacent horizontally
        ((0, 0), (1, 0), 1),  # Adjacent vertically
        ((0, 0), (1, 1), 2),  # Diagonal (1+1)
        ((0, 0), (3, 3), 6),  # Far diagonal (3+3)
        ((1, 2), (3, 4), 4),  # General case (2+2)
    ]
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    for a, b, expected in test_cases:
        result = manhattan(a, b)
        print(f"  Manhattan({a}, {b}) = {result} (expected {expected})")
        if result != expected:
            print(f"    ✗ Expected {expected}, got {result}")
            failures.append(f"Manhattan({a}, {b}): expected {expected}, got {result}")
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
        print("  ✓ All Manhattan distance calculations correct")


def test_quadrant_penalty():
    """Test quadrant penalty calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing quadrant_penalty...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating quadrant penalties...")
    
    # Create a sample with rooms in specific quadrants
    sample = GridSample(
        floor_id=1,
        grid_size=4,
        rooms=[
            RoomSpec(name="room_1", section="NW", min_cells=2),
            RoomSpec(name="room_2", section="SE", min_cells=2),
        ],
        target_mask=None,
    )
    
    # Test case 1: Rooms in correct quadrants (should have 0 penalty)
    cand1 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1)],  # NW quadrant
        "room_2": [(2, 2), (3, 3)],  # SE quadrant
    })
    
    # Test case 2: Rooms in wrong quadrants (should have penalty)
    cand2 = CandidateLayout(placement={
        "room_1": [(2, 2), (3, 3)],  # Should be NW, but in SE
        "room_2": [(0, 0), (0, 1)],  # Should be SE, but in NW
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    penalty1 = quadrant_penalty(sample, cand1)
    print(f"  Correct quadrants penalty: {penalty1} (expected 0.0)")
    if penalty1 != 0.0:
        print(f"    ✗ Expected 0.0, got {penalty1}")
        failures.append(f"Correct quadrants: expected 0.0, got {penalty1}")
    else:
        print(f"    ✓ Correct (no penalty for correct placement)")
    
    penalty2 = quadrant_penalty(sample, cand2)
    print(f"  Wrong quadrants penalty: {penalty2} (expected > 0.0)")
    if penalty2 <= 0.0:
        print(f"    ✗ Expected > 0.0, got {penalty2}")
        failures.append(f"Wrong quadrants: expected > 0.0, got {penalty2}")
    else:
        print(f"    ✓ Correct (penalty for wrong placement)")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ Quadrant penalty calculations correct")


def test_overlap_penalty():
    """Test overlap penalty calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing overlap_penalty...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating overlap penalties...")
    
    # Test case 1: No overlap
    cand1 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1)],
        "room_2": [(2, 2), (3, 3)],
    })
    
    # Test case 2: One overlapping cell
    cand2 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1)],
        "room_2": [(0, 1), (1, 1)],  # (0,1) overlaps
    })
    
    # Test case 3: Multiple overlapping cells
    cand3 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1), (1, 0)],
        "room_2": [(0, 1), (1, 0), (1, 1)],  # (0,1) and (1,0) overlap
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    penalty1 = overlap_penalty(cand1, grid_size=4)
    print(f"  No overlap penalty: {penalty1} (expected 0.0)")
    if penalty1 != 0.0:
        print(f"    ✗ Expected 0.0, got {penalty1}")
        failures.append(f"No overlap: expected 0.0, got {penalty1}")
    else:
        print(f"    ✓ Correct")
    
    penalty2 = overlap_penalty(cand2, grid_size=4)
    print(f"  One overlap penalty: {penalty2} (expected 1.0)")
    if penalty2 != 1.0:
        print(f"    ✗ Expected 1.0, got {penalty2}")
        failures.append(f"One overlap: expected 1.0, got {penalty2}")
    else:
        print(f"    ✓ Correct")
    
    penalty3 = overlap_penalty(cand3, grid_size=4)
    print(f"  Multiple overlaps penalty: {penalty3} (expected 2.0)")
    if penalty3 != 2.0:
        print(f"    ✗ Expected 2.0, got {penalty3}")
        failures.append(f"Multiple overlaps: expected 2.0, got {penalty3}")
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
        print("  ✓ Overlap penalty calculations correct")


def test_area_penalty():
    """Test area penalty calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing area_penalty...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating area penalties...")
    
    sample = GridSample(
        floor_id=1,
        grid_size=4,
        rooms=[
            RoomSpec(name="room_1", section="NW", min_cells=2),
            RoomSpec(name="room_2", section="SE", min_cells=3),
        ],
        target_mask=None,
    )
    
    # Test case 1: Exact match
    cand1 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1)],  # 2 cells (matches min_cells=2)
        "room_2": [(2, 2), (2, 3), (3, 2)],  # 3 cells (matches min_cells=3)
    })
    
    # Test case 2: Too few cells
    cand2 = CandidateLayout(placement={
        "room_1": [(0, 0)],  # 1 cell (needs 2, penalty = 1)
        "room_2": [(2, 2), (2, 3)],  # 2 cells (needs 3, penalty = 1)
    })
    
    # Test case 3: Too many cells
    cand3 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1), (1, 0)],  # 3 cells (needs 2, penalty = 1)
        "room_2": [(2, 2), (2, 3), (3, 2), (3, 3)],  # 4 cells (needs 3, penalty = 1)
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    penalty1 = area_penalty(sample, cand1)
    print(f"  Exact match penalty: {penalty1} (expected 0.0)")
    if penalty1 != 0.0:
        print(f"    ✗ Expected 0.0, got {penalty1}")
        failures.append(f"Exact match: expected 0.0, got {penalty1}")
    else:
        print(f"    ✓ Correct")
    
    penalty2 = area_penalty(sample, cand2)
    print(f"  Too few cells penalty: {penalty2} (expected 2.0)")
    if penalty2 != 2.0:
        print(f"    ✗ Expected 2.0, got {penalty2}")
        failures.append(f"Too few: expected 2.0, got {penalty2}")
    else:
        print(f"    ✓ Correct")
    
    penalty3 = area_penalty(sample, cand3)
    print(f"  Too many cells penalty: {penalty3} (expected 2.0)")
    if penalty3 != 2.0:
        print(f"    ✗ Expected 2.0, got {penalty3}")
        failures.append(f"Too many: expected 2.0, got {penalty3}")
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
        print("  ✓ Area penalty calculations correct")


def test_compactness_penalty():
    """Test compactness penalty calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing compactness_penalty...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating compactness penalties...")
    
    # Test case 1: Compact 2x2 square (low penalty)
    cand1 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2 square: perimeter=8, area=4, ratio=2.0
    })
    
    # Test case 2: Spread out cells (high penalty)
    cand2 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 3), (3, 0), (3, 3)],  # Four corners: perimeter=16, area=4, ratio=4.0
    })
    
    # Test case 3: Linear arrangement (medium penalty)
    cand3 = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1), (0, 2), (0, 3)],  # Row: perimeter=10, area=4, ratio=2.5
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    penalty1 = compactness_penalty(cand1)
    print(f"  Compact square penalty: {penalty1} (expected ~2.0)")
    if not (1.9 <= penalty1 <= 2.1):
        print(f"    ✗ Expected ~2.0, got {penalty1}")
        failures.append(f"Compact square: expected ~2.0, got {penalty1}")
    else:
        print(f"    ✓ Correct")
    
    penalty2 = compactness_penalty(cand2)
    print(f"  Spread out penalty: {penalty2} (expected ~4.0)")
    if not (3.9 <= penalty2 <= 4.1):
        print(f"    ✗ Expected ~4.0, got {penalty2}")
        failures.append(f"Spread out: expected ~4.0, got {penalty2}")
    else:
        print(f"    ✓ Correct")
    
    penalty3 = compactness_penalty(cand3)
    print(f"  Linear arrangement penalty: {penalty3} (expected ~2.5)")
    if not (2.4 <= penalty3 <= 2.6):
        print(f"    ✗ Expected ~2.5, got {penalty3}")
        failures.append(f"Linear: expected ~2.5, got {penalty3}")
    else:
        print(f"    ✓ Correct")
    
    # Verify that spread out has higher penalty than compact
    if penalty2 <= penalty1:
        print(f"    ✗ Spread out should have higher penalty than compact")
        failures.append("Spread out should have higher penalty than compact")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ Compactness penalty calculations correct")


def test_adjacency_penalty():
    """Test adjacency penalty calculation."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing adjacency_penalty...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating adjacency penalties...")
    
    sample = GridSample(
        floor_id=1,
        grid_size=4,
        rooms=[
            RoomSpec(name="kitchen_1", section="NW", min_cells=2),
            RoomSpec(name="dining_1", section="NE", min_cells=2),
            RoomSpec(name="bedroom_1", section="SW", min_cells=2),
            RoomSpec(name="bathroom_1", section="SE", min_cells=1),
        ],
        target_mask=None,
    )
    
    # Test case 1: Truly adjacent rooms (Manhattan distance = 1, should have 0 penalty)
    cand1 = CandidateLayout(placement={
        "kitchen_1": [(0, 0), (0, 1)],  # Centroid: (0, 0)
        "dining_1": [(0, 1), (1, 1)],  # Centroid: (0, 1), Manhattan distance from kitchen = 1
        "bedroom_1": [(2, 0), (2, 1)],  # Centroid: (2, 0)
        "bathroom_1": [(2, 1)],  # Centroid: (2, 1), Manhattan distance from bedroom = 1
    })
    
    # Test case 2: Rooms 2 cells apart (should have penalty = 2)
    cand2 = CandidateLayout(placement={
        "kitchen_1": [(0, 0), (0, 1)],  # Centroid: (0, 0)
        "dining_1": [(0, 2), (0, 3)],  # Centroid: (0, 2), Manhattan distance = 2, penalty = 1
        "bedroom_1": [(2, 0), (2, 1)],  # Centroid: (2, 0)
        "bathroom_1": [(2, 2)],  # Centroid: (2, 2), Manhattan distance = 2, penalty = 1
    })
    
    # Test case 3: Far apart rooms (should have large penalty)
    # kitchen_1 centroid: (0, 0), dining_1 centroid: (3, 2), distance = 5, penalty = 4
    # bedroom_1 centroid: (0, 2), bathroom_1 centroid: (3, 0), distance = 5, penalty = 4
    # Total penalty = 8
    cand3 = CandidateLayout(placement={
        "kitchen_1": [(0, 0), (0, 1)],  # Centroid: (0, 0)
        "dining_1": [(3, 2), (3, 3)],  # Centroid: (3, 2), Manhattan distance = 5, penalty = 4
        "bedroom_1": [(0, 2), (0, 3)],  # Centroid: (0, 2)
        "bathroom_1": [(3, 0)],  # Centroid: (3, 0), Manhattan distance = 5, penalty = 4
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    penalty1 = adjacency_penalty(sample, cand1)
    print(f"  Truly adjacent rooms (distance=1) penalty: {penalty1} (expected 0.0)")
    if penalty1 != 0.0:
        print(f"    ✗ Expected 0.0, got {penalty1}")
        failures.append(f"Adjacent rooms (distance=1): expected 0.0, got {penalty1}")
    else:
        print(f"    ✓ Correct (no penalty for Manhattan distance = 1)")
    
    penalty2 = adjacency_penalty(sample, cand2)
    print(f"  Rooms 2 cells apart penalty: {penalty2} (expected 2.0)")
    print(f"    Note: kitchen-dining centroids are 2 cells apart (penalty=1), bedroom-bathroom are 2 cells apart (penalty=1)")
    if penalty2 != 2.0:
        print(f"    ✗ Expected 2.0, got {penalty2}")
        failures.append(f"Rooms 2 cells apart: expected 2.0, got {penalty2}")
    else:
        print(f"    ✓ Correct (penalty for Manhattan distance > 1)")
    
    penalty3 = adjacency_penalty(sample, cand3)
    print(f"  Far apart rooms penalty: {penalty3} (expected 8.0)")
    print(f"    Note: kitchen-dining centroids are 5 cells apart (penalty=4), bedroom-bathroom are 5 cells apart (penalty=4)")
    if penalty3 != 8.0:
        print(f"    ✗ Expected 8.0, got {penalty3}")
        failures.append(f"Far apart: expected 8.0, got {penalty3}")
    else:
        print(f"    ✓ Correct (penalty for distant placement)")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ Adjacency penalty calculations correct")


def test_score_constraints():
    """Test the main score_constraints function."""
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)
    print("Testing score_constraints (main API)...")
    
    print("\n" + "=" * 60)
    print("PROCESS")
    print("=" * 60)
    print("Calculating constraint scores...")
    
    sample = GridSample(
        floor_id=1,
        grid_size=4,
        rooms=[
            RoomSpec(name="room_1", section="NW", min_cells=2),
            RoomSpec(name="room_2", section="SE", min_cells=2),
        ],
        target_mask=None,
    )
    
    # Good layout: correct quadrants, no overlap, correct areas
    cand_good = CandidateLayout(placement={
        "room_1": [(0, 0), (0, 1)],  # NW, 2 cells
        "room_2": [(2, 2), (3, 3)],  # SE, 2 cells
    })
    
    # Bad layout: wrong quadrants, overlap, wrong areas
    cand_bad = CandidateLayout(placement={
        "room_1": [(2, 2), (3, 3), (3, 2)],  # Should be NW but in SE, 3 cells (needs 2)
        "room_2": [(0, 0), (0, 1), (0, 0)],  # Should be SE but in NW, overlaps with itself, 2 cells (correct)
    })
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    failures = []
    
    scores_good = score_constraints(sample, cand_good)
    print(f"  Good layout scores:")
    print(f"    quadrant: {scores_good.quadrant}")
    print(f"    overlap: {scores_good.overlap}")
    print(f"    area: {scores_good.area}")
    print(f"    compactness: {scores_good.compactness}")
    print(f"    adjacency: {scores_good.adjacency}")
    
    if scores_good.quadrant != 0.0:
        print(f"    ✗ quadrant should be 0.0, got {scores_good.quadrant}")
        failures.append(f"Good layout quadrant: expected 0.0, got {scores_good.quadrant}")
    if scores_good.overlap != 0.0:
        print(f"    ✗ overlap should be 0.0, got {scores_good.overlap}")
        failures.append(f"Good layout overlap: expected 0.0, got {scores_good.overlap}")
    if scores_good.area != 0.0:
        print(f"    ✗ area should be 0.0, got {scores_good.area}")
        failures.append(f"Good layout area: expected 0.0, got {scores_good.area}")
    if scores_good.compactness <= 0.0:
        print(f"    ✗ compactness should be > 0.0, got {scores_good.compactness}")
        failures.append(f"Good layout compactness: expected > 0.0, got {scores_good.compactness}")
    
    scores_bad = score_constraints(sample, cand_bad)
    print(f"\n  Bad layout scores:")
    print(f"    quadrant: {scores_bad.quadrant}")
    print(f"    overlap: {scores_bad.overlap}")
    print(f"    area: {scores_bad.area}")
    print(f"    compactness: {scores_bad.compactness}")
    print(f"    adjacency: {scores_bad.adjacency}")
    
    if scores_bad.quadrant <= 0.0:
        print(f"    ✗ quadrant should be > 0.0, got {scores_bad.quadrant}")
        failures.append(f"Bad layout quadrant: expected > 0.0, got {scores_bad.quadrant}")
    if scores_bad.overlap <= 0.0:
        print(f"    ✗ overlap should be > 0.0, got {scores_bad.overlap}")
        failures.append(f"Bad layout overlap: expected > 0.0, got {scores_bad.overlap}")
    if scores_bad.area <= 0.0:
        print(f"    ✗ area should be > 0.0, got {scores_bad.area}")
        failures.append(f"Bad layout area: expected > 0.0, got {scores_bad.area}")
    
    # Verify that bad layout has higher total penalty
    total_good = sum([
        scores_good.quadrant,
        scores_good.overlap,
        scores_good.area,
        scores_good.compactness,
        scores_good.adjacency,
    ])
    total_bad = sum([
        scores_bad.quadrant,
        scores_bad.overlap,
        scores_bad.area,
        scores_bad.compactness,
        scores_bad.adjacency,
    ])
    
    if total_bad <= total_good:
        print(f"    ✗ Bad layout should have higher total penalty")
        failures.append(f"Bad layout total ({total_bad}) should be > good layout total ({total_good})")
    else:
        print(f"    ✓ Bad layout has higher total penalty ({total_bad} > {total_good})")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if failures:
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        pytest.fail(f"Test had {len(failures)} failure(s): {'; '.join(failures)}")
    else:
        print("  ✓ All constraint scores calculated correctly")
        print(f"  ✓ Good layout has lower penalties than bad layout")

