#!/usr/bin/env python3
"""
Convenience orchestrator that stitches together the utils modules to prep data in one go.

Steps:
1) Ensure the FloorPlans970 dataset is downloaded locally (via data_loader.download_datasets)
2) Run a quick audit to regenerate no_text_ids and optionally dump a few sample images
3) Generate processed floor plan masks/overlays/metadata (via data_formater.process_dataset)

Run from the repo root:
    python backend/utils/run_full_preprocessing.py --sample-count 5 --force-download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "backend" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from data_loader import download_datasets  # type: ignore
from preprocessing_data_extract import (  # type: ignore
    DEFAULT_DATASET_NAME,
    DEFAULT_LOCAL_DATASET_PATH,
    DEFAULT_NO_TEXT_PATH,
    run_data_audit,
)
from data_formater import process_dataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full preprocessing (download, audit, process masks).")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_LOCAL_DATASET_PATH, help="Local HF dataset path.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME, help="HF dataset id to download.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if dataset exists.")
    parser.add_argument(
        "--sample-output-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "processed" / "samples",
        help="Directory to export audit sample images.",
    )
    parser.add_argument("--sample-count", type=int, default=0, help="How many audit samples to export.")
    parser.add_argument(
        "--processed-out",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "processed" / "floor_plans",
        help="Output directory for processed floor plans.",
    )
    return parser.parse_args()


def ensure_dataset(dataset_path: Path, force_download: bool, dataset_name: str) -> None:
    """Download dataset to disk if missing or forced."""
    if force_download or not dataset_path.exists():
        print(f"Downloading dataset to {dataset_path} (force={force_download})...")
        download_datasets()
    else:
        print(f"Dataset already present at {dataset_path}, skipping download.")


def main() -> None:
    args = parse_args()

    ensure_dataset(args.dataset_path, args.force_download, args.dataset_name)

    # Load for audit (resolves columns, rebuilds no_text_ids, optional samples)
    print("\n=== Running data audit ===")
    audit_summary = run_data_audit(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        no_text_output=DEFAULT_NO_TEXT_PATH,
        sample_output_dir=args.sample_output_dir if args.sample_count > 0 else None,
        sample_count=args.sample_count,
    )
    print(json.dumps(audit_summary, indent=2))

    # Generate processed masks/metadata
    print("\n=== Generating processed floor plans ===")
    process_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        image_column=None,
        output_dir=args.processed_out,
        force_download=False,
    )
    print("All preprocessing steps finished.")


if __name__ == "__main__":
    main()
