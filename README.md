# Genplan Template — EA + RL Floorplan Experiments

Lightweight guide focused on setup, preprocessing, and how to run the EA/RL workflows (local and remote). Implementation details and long dependency lists are omitted for brevity.

## Setup

1) Create an environment (Python 3.11+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```
2) (Optional) Use conda instead:
   ```bash
   conda create -n genplan python=3.11 -y
   conda activate genplan
   pip install -r requirements.txt
   ```

## Data Prep (one-stop)

Run the full preprocessing pipeline (download → audit → masks):
```bash
python backend/utils/run_full_preprocessing.py --sample-count 0
```
Key artifacts land in `backend/data/processed/` (floor plans, masks, logs).

## Remote RL Training

Main CLI:
```bash
python -m backend.rl_training_remote \
  --episodes 10 --gens 150 --population 100 \
  --max-workers 4 --light-mutation
```
Flags of interest:
- `--light-mutation` trims heavy mutation/repair for faster runs.
- `--max-runtime-min` to bound wall time.
- `--max-workers` to throttle CPU use.
- `--no-ui` to disable curses UI; still prints periodic state.
- `--reset-state` to wipe prior bandit/logs before starting.

Wipe RL memory (state + log):
```bash
python backend/rl_training_remote/wipe_rl_memory.py
```

Push RL artifacts (bandit state + log) with auto pull/rebase:
```bash
python backend/rl_training_remote/push_rl_data.py --pull-rebase
```

## Local EA Notebook

- `backend/src/notebooks/ver0/EANotebook.ipynb` uses the same fixed floor IDs and seeding registry as RL.
- Fitness plots use log scale for visibility.
- Reads bandit state from `backend/data/rl/seed_bandit.json`.

## RL Training Notebook

- `backend/src/notebooks/ver0/RLTraining.ipynb` mirrors the CLI; fixed floor IDs expanded (~77) and log-scaled fitness plot.
- Logs to `backend/data/rl/episode_log.jsonl`, bandit state in `backend/data/rl/seed_bandit.json`.

## Monitoring Tips

- For lighter runs: reduce `--population`, `--gens`, and `--max-workers`; keep `--light-mutation` on.
- Early-stop logic in EA halts episodes on stagnation/plateaus or time budget (see `ver0/evolver.py`).

## Utilities

- `backend/utils/run_full_preprocessing.py`: download + audit + mask generation.
- RL results viewer: `notebooks/rl_training_results.ipynb` plots episode log (log y-scale).

## Paths to know

- Processed data: `backend/data/processed/`
- RL state/log: `backend/data/rl/seed_bandit.json`, `backend/data/rl/episode_log.jsonl`
- Remote runner: `backend/rl_training_remote/cli.py` (module entrypoint)
- EA/RL core: `backend/src/ver0/`
