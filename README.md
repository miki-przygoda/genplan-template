# Genplan Template — EA + RL Floorplan Experiments

A hybrid Evolutionary Algorithm + Reinforcement Learning system for generating constraint-aware architectural floorplans from natural-language instructions. The repo contains data preprocessing, reproducible RL/EA runners, and notebooks used to produce paper figures.

## Repository layout
- `backend/utils/`: data download and preprocessing (`run_full_preprocessing.py` orchestrates audit + mask generation).
- `backend/src/ver0/`: core EA/RL logic  
  - `evolver.py` (EA loop, restart/plateau logic), `mutator.py` (mutation/repair switches), `constraints.py` + `fitness.py` (scoring), `grid_encoder.py` (metadata → grids/masks), `rl_bandit.py` (epsilon-greedy seeding bandit), `rl_runner.py` (episode worker), `seeders.py` (initial layout strategies), `text_to_support_text.py` (NL → structured constraints), `vars.py` (defaults).
- `backend/rl_training_remote/`: CLI entrypoints for long RL runs and remote-friendly EA comparisons (`cli.py`, `train_remote_ea_eval.py`, `push_rl_data.py`, `wipe_rl_memory.py`).
- `backend/src/notebooks/`: analysis and training notebooks (`ver0/EANotebook.ipynb`, `ver0/RLTraining.ipynb`, `EA_Eval_Compare*.ipynb`, `rl_training_results.ipynb`).
- Data: processed floor plans and audit artifacts in `backend/data/processed/`; bandit/log outputs in `backend/data/rl/`; EA comparison logs in `backend/data/ea-logs/json/`.
- `papers/`: Currently includes a formal coursework summary of this system. (A full academic publication is planned, once completed will be added to the codebase.)

## Environment
- Python 3.11+ Recommended.
- Install deps (OpenCV is headless; on some OSes you may need system `libGL`/`libglib` packages):
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
  pip install -r requirements.txt
  ```
  or via conda: Recomended
  ```bash
  conda create -n genplan python=3.11 -y
  conda activate genplan
  pip install -r requirements.txt
  ```

## Fast start (10–15 minutes)
1) **Prep data (download → audit → masks):**
   ```bash
   python backend/utils/run_full_preprocessing.py --sample-count 0
   ```
   - Downloads `HamzaWajid1/FloorPlans970Dataset` to `backend/data/dataset/` if missing.
   - Writes processed floors to `backend/data/processed/floor_plans/floor###/` with `metadata.json`, `room_mask.png`, `overlay.png`.
   - Rebuilds `backend/data/processed/no_text_ids.json` with samples lacking supporting text.

2) **Run a short RL+EA session (curses UI if TTY):**
   ```bash
   python -m backend.rl_training_remote \
     --episodes 5 --gens 60 --population 50 \
     --max-workers 4 --light-mutation --max-runtime-min 10
   ```
   Outputs:
   - bandit state `backend/data/rl/seed_bandit.json`
   - per-episode log `backend/data/rl/episode_log.jsonl`

3) **Visualise results:** open `backend/src/notebooks/rl_training_results.ipynb` and point it at `backend/data/rl/episode_log.jsonl` (plots use log-scale fitness).

## Data pipeline details
- `backend/utils/run_full_preprocessing.py` stitches:
  1. `data_loader.download_datasets()` — pulls the HF dataset to disk (safe to re-run).
  2. `preprocessing_data_extract.run_data_audit()` — resolves column names, checks image sizes, records `no_text_ids.json`, and can export sample PNGs.
  3. `data_formater.process_dataset()` — extracts polygons via OpenCV edge detection, builds room masks/overlays, and writes `metadata.json` (schema versioned, includes supporting text, room polygons/centroids, corner points).
- Processed structure (one per floor, numbered): `backend/data/processed/floor_plans/floor###/{metadata.json,room_mask.png,overlay.png}`.

## Running experiments
- **RL bandit training (primary entrypoint):**
  - CLI: `python -m backend.rl_training_remote [--episodes N --gens G --population P ...]`
  - Helpful flags: `--light-mutation` (skip heavy repairs), `--max-runtime-min` (wall clock cap), `--max-workers` (CPU throttle), `--no-ui` (print summaries), `--reset-state` (wipe prior state/logs), `--random-floors` or `--fixed-floors` (override floor cycle).
  - Logs: `backend/data/rl/episode_log.jsonl`; bandit: `backend/data/rl/seed_bandit.json`.
  - Maintenance: `backend/rl_training_remote/wipe_rl_memory.py` (delete state/log), `backend/rl_training_remote/push_rl_data.py --pull-rebase` (stage/commit/push RL artifacts).

- **EA comparison batches (RL vs manual seeder):**
  - `python backend/rl_training_remote/train_remote_ea_eval.py --runs 6 --max-workers 6`
  - Writes paired EA histories to `backend/data/ea-logs/json/` with lightweight live status.

- **Notebooks (reproducibility + figures):**
  - `backend/src/notebooks/ver0/RLTraining.ipynb`: mirrors the RL CLI with a fixed floor set (~77 ids) and log-scale plots.
  - `backend/src/notebooks/ver0/EANotebook.ipynb`: manual EA runs with the same seeding registry and target floors.
  - `backend/src/notebooks/EA_Eval_Compare*.ipynb`: paper-style RL-vs-manual comparisons.

## Key algorithmic components (for citing/inspection)
- EA loop and restart/stagnation logic: `backend/src/ver0/evolver.py`
- Constraint scoring and penalties (quadrant, overlap, compactness, adjacency, room usage, relationships, hole detection): `backend/src/ver0/constraints.py`
- Fitness aggregation and realism gate: `backend/src/ver0/fitness.py`, `backend/src/ver0/real_plan_classifier.py`
- Grid encoding of processed metadata to EA-ready tensors/masks: `backend/src/ver0/grid_encoder.py`
- Mutations/repairs and light/heavy modes: `backend/src/ver0/mutator.py`
- Seed selection bandit and seeding strategies: `backend/src/ver0/rl_bandit.py`, `backend/src/ver0/seeders.py`
- Text-to-structure parser for supporting text: `backend/src/ver0/text_to_support_text.py`

## Testing and reproducibility
- Unit tests: `pytest backend/tests/ver0` (grid encoder, constraints, text parsing).
- RNG seeds: EA config seeds live in `backend/src/ver0/vars.py`; RL bandit state persists to `backend/data/rl/seed_bandit.json` for repeatability across runs.

## Frequently used paths
- Processed floors: `backend/data/processed/floor_plans/`
- Missing-text ids: `backend/data/processed/no_text_ids.json`
- RL state/logs: `backend/data/rl/seed_bandit.json`, `backend/data/rl/episode_log.jsonl`
- EA comparison logs: `backend/data/ea-logs/json/`
- Remote RL CLI entrypoint: `backend/rl_training_remote/cli.py`
- EA/RL core code: `backend/src/ver0/`