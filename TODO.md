# FloorPlanEA — v0 Development Roadmap

This roadmap reflects the final scope for **v0 (coursework submission)** and marks
any items postponed to post-coursework development (hybrid EA/RL, 2×2 encoder, etc.).

---

## 1. Data inspection & setup
- [x] Load all 970 samples and identify the 35 with no supporting text.
- [x] Save IDs to `backend/data/processed/no_text_ids.json`.
- [x] Verify image shapes (512×512).
- [x] Manually inspect 10–20 images (mask export enabled).
- [ ] Final pass: evaluate room extraction quality (quick visual + spot-check metadata).

---

## 2. Room extraction (edge/contour segmentation)
- [x] Implemented in `utils/data_formater.py`:
  - [x] Room mask generation
  - [x] Room polygons
  - [x] Centroids
  - [x] Corner points
  - [x] Metadata export to `processed/floor_plans/`
- [ ] Evaluate segmentation leakage & missing-room cases.
- [ ] (Optional, for later) fallback ML-based segmentation.

> v0 uses **existing edge-based segmentation**.  
> No need for ML refinement before coursework deadline.

---

## 3. Small handmade labelled set (fallback) — *POSTPONED*
- [ ] Create `labelled_samples/` with 20–40 manually labelled images.
- [ ] Add JSON room annotations.
- [ ] Train tiny model for segmentation correction.
- [ ] Compare to automated masks.

> *Post-coursework; only needed for a v1 pipeline.*

---

## 4. Natural language → structured metadata
- [x] Implement `backend/src/ver0/text_to_support_text.py`.
- [x] Full support for:
  - Multi-room expressions
  - Ordinals (compact, numeric, word-based)
  - Room aliases
  - Directional relationships
  - Section/location parsing
  - Conflict preservation
- [x] Comprehensive test suite (pytest).
- [ ] Add conflict-debugging pretty-printer (optional).

---

## 5. Orientation / rotation normalisation — *LOW PRIORITY*
- [ ] Create `backend/src/ver0/orient.py`
  - [ ] Detect dominant wall orientation
  - [ ] Rotate image to nearest 0/90/180/270
  - [ ] Save rotation info to metadata

> **Optional for v0.**  
> Only implement if we discover rotation inconsistencies in the dataset.

---

## 6. Grid encoder (4×4 v0)
- [x] Implement `backend/src/ver0/grid_encoder.py`
  - [x] Map polygons → centroids → grid cells
  - [x] Section override from text parser
  - [x] Estimate minimum size via area ratio
  - [x] Output `GridSample`
- [x] Test suite in `backend/tests/ver0/test_grid_encoder.py`
- [ ] Batch script to generate encoded layouts to:
      `backend/data/processed/encoded/{id}.json`

> 2×2 grid encoder postponed until RL/EAv2 work.

---

## 7. Fitness design (soft penalties)
- [x] Implement `backend/src/ver0/constraints.py`:
  - [x] quadrant penalty
  - [x] overlap penalty
  - [x] area penalty
  - [x] compactness penalty
  - [x] adjacency penalty
- [x] Implement `backend/src/ver0/fitness.py`:
  - [x] Weight config
  - [x] `evaluate()` for EA
- [x] Test suite

- [ ] Add YAML/JSON config loader for fitness weights (optional).

---

## 8. Evolutionary Algorithm (EA v0)
- [ ] Create `backend/src/ver0/evolver.py`
  - [ ] Define genome format (4×4 occupancy: `room_id → list of cells`)
  - [ ] `init_population()`
  - [ ] `evaluate_population()`
  - [ ] `tournament_select()` or `rank_select()`
  - [ ] `crossover()` (cell swaps / room exchanges)
  - [ ] `mutate()` (cell shuffling, section nudges)
  - [ ] Main loop: produce JSONL logs per generation

- [ ] Add high-level script:
      `tools/run_ea_v0.py` to run a full EA optimisation for a single sample.

---

## 9. Integration tests & utilities
- [x] Unit tests for all core modules.
- [ ] Integration test:
  - [ ] image → grid → fitness (single sample)
  - [ ] text description → grid override → fitness
- [ ] CLI script: `tools/run_preprocess.py` to regenerate entire pipeline.
- [ ] Create `backend/src/ver0/README.md` documenting:
  - Data flow
  - Grid encoder
  - Fitness system
  - EA usage
  - Future v1/v2 roadmap (RL–EA)

---

## 10. Post-coursework (v1+)
- [ ] RL-assisted EA (model B in your diagram)
- [ ] Experience replay of mutated genomes
- [ ] RL critic that predicts fitness residuals
- [ ] v2 synergistic optimisation (mutual EA–RL)
- [ ] 2×2 + hierarchical grid encoder