## 1. Data inspection & setup
- [x] Load all 970 samples and check which 35 have no supporting text; save their IDs to `processed/no_text_ids.json`.
- [x] Verify image shape is always 512x512 (raise warning if not).
- [x] Inspect 10–20 images to confirm colour-coded rooms are consistent across the dataset. (Functionality exists in `utils/preprocessing_data_extract.py` via `export_sample_images()`)

## 2. Room extraction (colour-based first pass)
- [x] Room extraction implemented (via edge-based contour detection instead of HSV color-based approach)
  - [x] Room polygons extracted using edge detection and contour detection (`utils/data_formater.py` and `utils/preprocessing_edge_mask.py`)
  - [x] Room masks and metadata exported to `processed/floor_plans/{room_id}/metadata.json`
  - [x] Includes room polygons, centroids, corner points, and area information
  - Note: Implementation uses edge/contour-based detection rather than HSV color-based, and is located in `utils/` rather than `src/evo_floorplan/`
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
- [] Create `src/evo_floorplan/text_parser.py`.
- [] Define spatial vocab: `["N","S","E","W","NE","NW","SE","SW"]`.
- [] Implement simple rule/regex parser for sentences like:
  - “The first bedroom is located at south west.”
  - “The first bathroom is located at north east.”
- [] Output format:
  ```json
  {
    "bedroom_1": {"zone": "SW"},
    "bathroom_1": {"zone": "NE"}
  }
  ```

## 5. Orientation / rotation normalisation
- [] Add script `src/evo_floorplan/orient.py`:
  - [] Detect dominant wall angle (Hough or largest contour).
  - [] Rotate image to the closest 0/90/180/270.
  - [] Save rotation angle in metadata for later use.

## 6. Grid encoder
- [] Implement 2x2 and 4x4 grid encoders over the *cropped* floor-plan area.
- [] For each detected room, map occupied cells:
  - [] `grid_2`: 4 values
  - [] `grid_4`: 16 values
- [] Save per-image encoded layout to `processed/encoded/{image_id}.json`.

## 7. Fitness design (soft penalties)
- [] Define initial fitness terms:
  - [] F1: room-in-correct-zone penalty (based on text metadata)
  - [] F2: room-overlap penalty
  - [] F3: empty-cell penalty / compactness
- [] Combine as weighted sum:
  - [] `fitness = w1*F1 + w2*F2 + w3*F3`
  - [] Keep weights configurable via YAML/JSON.

## 8. EA scaffold
- [] Create `src/evo_floorplan/evolution.py` with:
  - [] `init_population(...)`
  - [] `evaluate_population(...)`
  - [] `select(...)`
  - [] `crossover(...)`
  - [] `mutate(...)`
- [] Log per-generation best fitness to `logs/evo_run_*.jsonl`.
- [] (Later) plug in grid-encoded layouts as genomes.

## 9. Testing & utilities
- [] Add a tiny CLI script `tools/run_preprocess.py` to run ALL preprocessing steps for the team.
- [] Add unit-style checks for:
  - [] image → grayscale → binarized
  - [] text → structured metadata
  - [] encoded grid length == expected
- [] Document all of this in `src/evo_floorplan/README.md`.

