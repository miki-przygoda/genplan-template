## 1. Data inspection & setup
- [x] Load all 970 samples and check which 35 have no supporting text; save their IDs to `backend/data/processed/no_text_ids.json`.
- [x] Verify image shape is always 512x512 (raise warning if not).
- [x] Inspect 10–20 images to confirm colour-coded rooms are consistent across the dataset. (Functionality exists in `utils/preprocessing_data_extract.py` via `export_sample_images()`)

## 2. Room extraction (colour-based first pass)
- [x] Room extraction implemented (via edge-based contour detection instead of HSV color-based approach)
  - [x] Room polygons extracted using edge detection and contour detection (`utils/data_formater.py` and `utils/preprocessing_edge_mask.py`)
  - [x] Room masks and metadata exported to `backend/data/processed/floor_plans/{room_id}/metadata.json`
  - [x] Includes room polygons, centroids, corner points, and area information
  - Note: Implementation uses edge/contour-based detection rather than HSV color-based, and is located in `utils/` rather than `src/ver0/`
- [] Evaluate "quality" of extraction (precision of borders, leakage between rooms).
- [] If quality is poor, fall back to manual/ML approach (see section 3).

## 3. Small handmade labelled set (fallback)
- [] Create `labelled_samples/` with 20–40 images.
- [] Manually store room coordinates & types in JSON:
  ```json
  {
    "image_id": "xxx.png",
    "rooms": [
      {"type": "bedroom", "bbox": [x1, y1, x2, y2]},
      {"type": "bathroom", "bbox": [x1, y1, x2, y2]}
    ]
  }
  ```
- [] Train a tiny model / script to learn colour-to-room or contour-to-room mapping (optional).
- [] Use this labelled set later to measure extraction error.

## 4. Text → structured metadata
- [x] Create `backend/src/ver0/text_to_support_text.py`.
- [x] Define spatial vocab: `["N","S","E","W","NE","NW","SE","SW","C"]`.
- [x] Implement rule/regex parser for sentences like:
  - "The first bedroom is located at south west."
  - "The first bathroom is located at north east."
  - Supports directional relationships (north_of, south_of, etc.)
  - Supports multi-room phrases and various room type aliases
- [x] Output format: List of `Room` objects with sections, relationships, and metadata
- [x] Comprehensive test suite in `backend/tests/ver0/test_text_to_support_text.py`

## 5. Orientation / rotation normalisation
- [] Add script `backend/src/ver0/orient.py`:
  - [] Detect dominant wall angle (Hough or largest contour).
  - [] Rotate image to the closest 0/90/180/270.
  - [] Save rotation angle in metadata for later use.

## 6. Grid encoder
- [x] Implement 4x4 grid encoder (`backend/src/ver0/grid_encoder.py`)
  - [x] `encode_floorplan_to_grid()` function that converts floor plan metadata to `GridSample`
  - [x] Maps room centroids to grid cells
  - [x] Calculates room sections (NW, NE, SW, SE) based on cell positions
  - [x] Calculates minimum cells based on room area ratios
  - [x] Supports text section overrides from parser
  - [x] Creates target mask for room positions
- [x] Comprehensive test suite in `backend/tests/ver0/test_grid_encoder.py`
- [] Implement 2x2 grid encoder (optional extension)
- [] Save per-image encoded layout to `backend/data/processed/encoded/{image_id}.json` (batch processing script needed)

## 7. Fitness design (soft penalties)
- [x] Define constraint scoring system (`backend/src/ver0/constraints.py`):
  - [x] F1: quadrant penalty (room-in-correct-zone based on text metadata)
  - [x] F2: overlap penalty (room-overlap detection)
  - [x] F3: area penalty (room size matching)
  - [x] F4: compactness penalty (bounding box efficiency)
  - [x] F5: adjacency penalty (common room pairs distance)
- [x] Implement weighted fitness function (`backend/src/ver0/fitness.py`):
  - [x] `score_constraints()` returns `ConstraintScores` with all penalty terms
  - [x] `scalarize()` combines scores with configurable weights
  - [x] `evaluate()` main API for fitness evaluation
  - [x] Default weights: quadrant=1.0, overlap=3.0, area=1.0, compactness=0.5, adjacency=0.5
- [x] Comprehensive test suite in `backend/tests/ver0/test_constraints.py`
- [] Add YAML/JSON configuration file for weights (optional enhancement)

## 8. EA scaffold
- [] Create `backend/src/ver0/evolution.py` with:
  - [] `init_population(...)`
  - [] `evaluate_population(...)`
  - [] `select(...)`
  - [] `crossover(...)`
  - [] `mutate(...)`
- [] Log per-generation best fitness to `logs/evo_run_*.jsonl`.
- [] (Later) plug in grid-encoded layouts as genomes.

## 9. Testing & utilities
- [x] Comprehensive unit tests for core modules:
  - [x] `backend/tests/ver0/test_text_to_support_text.py` (10 test functions)
  - [x] `backend/tests/ver0/test_grid_encoder.py` (10 test functions)
  - [x] `backend/tests/ver0/test_constraints.py` (10 test functions)
- [x] Test infrastructure:
  - [x] Pytest configuration in `pyproject.toml`
  - [x] Test directory structure: `backend/tests/ver0/`
- [] Add a tiny CLI script `tools/run_preprocess.py` to run ALL preprocessing steps for the team.
- [] Add integration tests for:
  - [] Full pipeline: image → grid encoding → fitness evaluation
  - [] Text parsing → grid encoding integration
- [] Document all of this in `backend/src/ver0/README.md`.

