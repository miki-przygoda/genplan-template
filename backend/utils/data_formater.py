"""
Combined data formatting utility for the FloorPlans970 dataset.

This module provides a complete workflow:
1. Download datasets from Hugging Face if needed
2. Load the dataset from disk or Hugging Face Hub
3. Extract room polygons and create masks
4. Save processed data to the processed directory
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

try:
    from PIL import Image
    PILImage = Image  # Alias for consistency
except ImportError:  # pragma: no cover - pillow is listed in requirements
    Image = None  # type: ignore
    PILImage = None  # type: ignore

# Constants
DEFAULT_DATASET_NAME = "HamzaWajid1/FloorPlans970Dataset"
DEFAULT_LOCAL_DATASET_PATH = Path("backend") / "dataset" / "FloorPlans970Dataset"
DEFAULT_PROCESSED_DIR = Path("backend") / "processed"
DEFAULT_NO_TEXT_PATH = Path("backend") / "processed" / "no_text_ids.json"
EXPECTED_IMAGE_SHAPE = (512, 512)
SCHEMA_VERSION = "1.0.0"

DATASETS = ["HamzaWajid1/FloorPlans970Dataset"]

PREFERRED_TEXT_COLUMNS: Sequence[str] = (
    "supporting_text",
    "supporting_texts",
    "description",
    "text",
)
PREFERRED_ID_COLUMNS: Sequence[str] = (
    "image_id",
    "plan_id",
    "id",
    "file_name",
    "filename",
)
PREFERRED_IMAGE_COLUMNS: Sequence[str] = ("image", "png", "jpg", "jpeg")


class DataInspectionError(RuntimeError):
    """Raised when an expected dataset field cannot be resolved."""


# ============================================================================
# Dataset Download Functions
# ============================================================================

def download_datasets():
    """
    Download datasets from Hugging Face and organize them into folders
    based on the dataset name (second part of the repository path).
    Includes rate limiting protection with random delays.
    """
    os.makedirs("backend/dataset", exist_ok=True)
    
    for i, dataset_path in enumerate(DATASETS):
        dataset_name = dataset_path.split("/")[-1]
        dataset_folder = os.path.join("backend", "dataset", dataset_name)
        
        print(f"Downloading {dataset_path}...")
        print(f"Creating folder: {dataset_folder}")
        
        if i > 0:
            delay = random.uniform(2, 5)
            print(f"⏳ Waiting {delay:.1f} seconds to avoid rate limiting...")
            time.sleep(delay)
        
        try:
            os.makedirs(dataset_folder, exist_ok=True)
            
            print("   Loading dataset (this may take a while for large datasets)...")
            dataset = load_dataset(dataset_path, verification_mode="no_checks")
            
            dataset.save_to_disk(dataset_folder)
            
            print(f"✅ Successfully downloaded and saved {dataset_path} to {dataset_folder}")
            print(f"   Dataset info: {dataset}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_path}: {str(e)}")
            
            if "expected" in str(e) and "recorded" in str(e):
                print("   💡 This appears to be a dataset metadata mismatch error.")
                print("   💡 The dataset may have been updated but metadata is outdated.")
                print("   💡 Try downloading this dataset manually or contact the dataset owner.")
            elif "Connection" in str(e) or "timeout" in str(e).lower():
                print("   💡 This appears to be a network connectivity issue.")
                print("   💡 Check your internet connection and try again.")
            else:
                print("   💡 Unknown error - check the dataset URL and try again.")
            
            print("-" * 50)
            
            if i < len(DATASETS) - 1:
                error_delay = random.uniform(5, 10)
                print(f"⏳ Waiting {error_delay:.1f} seconds before next download...")
                time.sleep(error_delay)


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_floorplans_dataset(
    dataset_path: Optional[Path] = None,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: Optional[str] = "train",
) -> Dataset:
    """
    Load the FloorPlans970 dataset from disk if available; otherwise fall back
    to Hugging Face Hub.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_LOCAL_DATASET_PATH

    dataset: Dataset | DatasetDict
    try:
        dataset = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset from disk: {e}")
        print("Falling back to Hugging Face Hub")
        dataset = load_dataset(dataset_name, verification_mode="no_checks")
        dataset.save_to_disk(str(dataset_path))

    if isinstance(dataset, DatasetDict):
        if split is not None and split in dataset:
            return dataset[split]
        if "train" in dataset:
            return dataset["train"]
        # Fall back to the first available split.
        first_split = next(iter(dataset.keys()))
        return dataset[first_split]

    return dataset


def resolve_column(column_names: Sequence[str], candidates: Sequence[str]) -> str:
    """Pick the first column name that exists within the dataset."""
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    raise DataInspectionError(
        f"None of the candidate columns {candidates} were found in {column_names}."
    )


# ============================================================================
# Mask Extraction Functions
# ============================================================================

def extract_room_polygons(pil_image, min_area_ratio: float = 0.001, corner_quality: float = 0.01):
    """Return filled room mask, room contours, metadata, and snapped corners."""

    # Convert PIL image to OpenCV-friendly arrays
    rgb_image = np.array(pil_image.convert("RGB"))
    gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)

    # Enhance structural edges and connect potential gaps in walls
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 40, 120)

    wall_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thick_walls = cv.dilate(edges, wall_kernel, iterations=2)
    thick_walls = cv.morphologyEx(thick_walls, cv.MORPH_CLOSE, wall_kernel, iterations=2)

    # Invert to isolate open space, then flood fill the exterior to remove background
    open_space = cv.bitwise_not(thick_walls)
    floodfilled = open_space.copy()
    height, width = open_space.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv.floodFill(floodfilled, flood_mask, (0, 0), 0)

    room_mask = cv.morphologyEx(floodfilled, cv.MORPH_OPEN, wall_kernel, iterations=2)

    contours, _ = cv.findContours(room_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * (height * width)

    filled_mask = np.zeros_like(room_mask)
    polygons = []
    metadata = []

    corner_candidates = cv.goodFeaturesToTrack(
        thick_walls,
        maxCorners=500,
        qualityLevel=corner_quality,
        minDistance=10,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04,
    )

    deduplication_threshold = 6
    unique_corners = []
    if corner_candidates is not None:
        for candidate in corner_candidates:
            x, y = candidate.ravel()
            x, y = int(x), int(y)
            matched = False
            for stored in unique_corners:
                if abs(stored[0] - x) <= deduplication_threshold and abs(stored[1] - y) <= deduplication_threshold:
                    stored[0] = int((stored[0] * stored[2] + x) / (stored[2] + 1))
                    stored[1] = int((stored[1] * stored[2] + y) / (stored[2] + 1))
                    stored[2] += 1
                    matched = True
                    break
            if not matched:
                unique_corners.append([x, y, 1])

    corner_points = [(corner[0], corner[1]) for corner in unique_corners]
    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area:
            continue

        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]

        cv.drawContours(filled_mask, [contour], contourIdx=-1, color=255, thickness=cv.FILLED)

        moments = cv.moments(contour)
        if moments["m00"] != 0:
            centroid = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        else:
            centroid = (0, 0)

        corner_ids = []
        if corner_points:
            for vertex in polygon:
                distances = [np.linalg.norm(np.array(vertex) - np.array(corner)) for corner in corner_points]
                nearest = int(np.argmin(distances))
                corner_ids.append(nearest)

        polygons.append(contour)
        metadata.append(
            {
                "room_id": len(metadata),
                "area_px": int(area),
                "polygon": polygon,
                "centroid": centroid,
                "corner_ids": corner_ids,
            }
        )

    return filled_mask, polygons, metadata, corner_points


def colourise_rooms(rgb_image: np.ndarray, room_polygons, corner_points):
    """Create an overlay and blended visualisation for segmented rooms."""

    overlay = np.zeros_like(rgb_image)
    rng = np.random.default_rng(seed=42)

    for index, contour in enumerate(room_polygons):
        colour = rng.integers(0, 255, size=3).tolist()
        cv.fillPoly(overlay, [contour], colour)

    for corner in corner_points:
        cv.circle(overlay, corner, radius=4, color=(255, 255, 255), thickness=-1)

    blended = cv.addWeighted(rgb_image, 0.6, overlay, 0.4, gamma=0)
    return overlay, blended


# ============================================================================
# Utility Functions
# ============================================================================

def _derive_record_identifier(record: Dict[str, Any], index: int) -> str:
    """Return a stable identifier for a dataset record."""
    for key in ("image_id", "plan_id", "id", "file_name", "filename"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
        return str(value)
    return f"sample_{index:05d}"


def _normalise_identifier(identifier: str) -> str:
    """Sanitise identifier for safe file naming."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("._-")
    return cleaned or "sample"


def _prepare_room_metadata(raw_metadata):
    """Convert metadata into JSON-serialisable form."""
    serialisable = []
    for room in raw_metadata:
        room_entry = {
            "room_id": int(room["room_id"]),
            "area_px": int(room["area_px"]),
            "polygon": [list(vertex) for vertex in room["polygon"]],
            "centroid": list(room["centroid"]),
            "corner_ids": [int(corner_id) for corner_id in room["corner_ids"]],
        }
        serialisable.append(room_entry)
    return serialisable


def write_json(path: Path, data: Any) -> None:
    """Persist a JSON payload to disk with pretty indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse version string like '1.0.0' into (major, minor, patch) tuple."""
    parts = version_str.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (major, minor, patch)


def _compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    Returns: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1 = _parse_version(version1)
    v2 = _parse_version(version2)
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0


def write_versioned_metadata_json(json_path: Path, payload: Dict[str, Any]) -> None:
    """
    Write metadata.json with versioning support.
    
    - Adds schema_version at the very top of the JSON
    - If the file exists with the same version, overwrites it
    - If the file exists with an older version, backs it up as metadata.v{old_version}.json
    - Keeps maximum 2 old versions (removes older ones)
    """
    # Add schema_version at the very top of the payload
    payload_with_version = {"schema_version": SCHEMA_VERSION, **payload}
    
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and read existing version if present
    existing_version = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_version = existing_data.get("schema_version")
        except (json.JSONDecodeError, IOError):
            # If we can't read it, treat as if it doesn't exist
            existing_version = None
    
    if existing_version is None:
        # No existing file or couldn't read version - just write new file
        with json_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(payload_with_version, metadata_file, indent=2, ensure_ascii=False)
            metadata_file.write("\n")
    else:
        # Compare versions
        version_comparison = _compare_versions(existing_version, SCHEMA_VERSION)
        
        if version_comparison == 0:
            # Same version - overwrite
            with json_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(payload_with_version, metadata_file, indent=2, ensure_ascii=False)
                metadata_file.write("\n")
        elif version_comparison < 0:
            # Existing version is older - backup and create new
            # Normalize version for filename (replace dots with underscores for safety)
            version_filename = existing_version.replace(".", "_")
            backup_path = json_path.parent / f"metadata.v{version_filename}.json"
            
            # If backup already exists, remove it first (shouldn't happen, but handle it)
            if backup_path.exists():
                backup_path.unlink()
            
            # Move existing file to backup
            json_path.rename(backup_path)
            
            # Write new file
            with json_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(payload_with_version, metadata_file, indent=2, ensure_ascii=False)
                metadata_file.write("\n")
            
            # Clean up old backups - keep only 2 most recent
            _cleanup_old_metadata_backups(json_path.parent)
        else:
            # Existing version is newer - this shouldn't happen, but handle gracefully
            # Create a backup with current version and warn
            version_filename = SCHEMA_VERSION.replace(".", "_")
            backup_path = json_path.parent / f"metadata.v{version_filename}.json"
            
            # If current version backup doesn't exist, create it
            if not backup_path.exists():
                with backup_path.open("w", encoding="utf-8") as backup_file:
                    json.dump(payload_with_version, backup_file, indent=2, ensure_ascii=False)
                    backup_file.write("\n")
            
            # Don't overwrite the newer version file
            print(f"Warning: Existing metadata.json has newer version {existing_version} than current {SCHEMA_VERSION}. Keeping existing file.")


def _cleanup_old_metadata_backups(directory: Path) -> None:
    """
    Keep only the 2 most recent metadata backup files.
    Removes older backup files to maintain at most 2 versions behind.
    """
    # Find all metadata backup files (metadata.v*.json)
    backup_files = list(directory.glob("metadata.v*.json"))
    
    if len(backup_files) <= 2:
        # We have 2 or fewer backups, no cleanup needed
        return
    
    # Sort by modification time (newest first)
    backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Keep the 2 most recent, delete the rest
    for old_backup in backup_files[2:]:
        try:
            old_backup.unlink()
        except OSError:
            # If we can't delete it, continue with others
            pass


# ============================================================================
# Main Workflow
# ============================================================================

def process_dataset(
    dataset_path: Optional[Path] = None,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: Optional[str] = "train",
    image_column: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force_download: bool = False,
):
    """
    Complete workflow: download (if needed), load dataset, extract masks, and save to processed.
    
    Args:
        dataset_path: Path to locally saved dataset
        dataset_name: Hugging Face dataset identifier
        split: Dataset split to load
        image_column: Override image column name
        output_dir: Directory to save processed data
        force_download: Force re-download even if dataset exists locally
    """
    if output_dir is None:
        output_dir = DEFAULT_PROCESSED_DIR / "floor_plans"
    
    # Step 1: Download dataset if needed (only if missing, not forced unless specified)
    dataset_path_to_check = dataset_path or DEFAULT_LOCAL_DATASET_PATH
    if force_download or not dataset_path_to_check.exists():
        print("=" * 50)
        print("Step 1: Downloading dataset...")
        print("=" * 50)
        download_datasets()
        print()
    
    # Step 2: Load dataset
    print("=" * 50)
    print("Step 2: Loading dataset...")
    print("=" * 50)
    dataset = load_floorplans_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
    )
    
    # Resolve image column
    try:
        resolved_image_column = image_column or resolve_column(
            dataset.column_names, PREFERRED_IMAGE_COLUMNS
        )
    except DataInspectionError:
        resolved_image_column = "image"
        print(f"Warning: Could not resolve image column, using '{resolved_image_column}'")
    
    # Resolve text column
    try:
        resolved_text_column = resolve_column(
            dataset.column_names, PREFERRED_TEXT_COLUMNS
        )
    except DataInspectionError:
        resolved_text_column = None
        print(f"Warning: Could not resolve text column, supporting text will not be included")
    
    # Step 2.5: Regenerate no_text_ids.json
    print("=" * 50)
    print("Step 2.5: Regenerating no_text_ids.json...")
    print("=" * 50)
    no_text_ids = set()
    if resolved_text_column:
        for idx, record in enumerate(dataset):
            text_value = record.get(resolved_text_column)
            is_missing = False
            
            if text_value is None:
                is_missing = True
            elif isinstance(text_value, str):
                is_missing = text_value.strip() == ""
            elif isinstance(text_value, (list, tuple, set)):
                is_missing = all((not item or str(item).strip() == "") for item in text_value)
            
            if is_missing:
                # Use 1-based ID to match floor numbering (floor001, floor002, etc.)
                floor_id = idx + 1
                no_text_ids.add(floor_id)
        
        # Save to processed/no_text_ids.json
        no_text_output_path = Path("backend") / "processed" / "no_text_ids.json"
        no_text_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(no_text_output_path, sorted(list(no_text_ids)))
        print(f"Regenerated no_text_ids.json with {len(no_text_ids)} records without text")
        print(f"Saved to: {no_text_output_path}")
    else:
        print("Skipping no_text_ids.json generation: text column not available")
    print()
    
    print(f"Loaded dataset with {len(dataset)} records")
    print(f"Using image column: '{resolved_image_column}'")
    if resolved_text_column:
        print(f"Using text column: '{resolved_text_column}'")
    print()
    
    # Step 3: Process images and create masks
    print("=" * 50)
    print("Step 3: Processing images and creating masks...")
    print("=" * 50)
    print("Note: All existing files will be overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_records = len(dataset)
    print(f"Processing {total_records} floorplan samples into {output_dir}...")
    
    for index, record in enumerate(dataset):
        image = record.get(resolved_image_column)
        if image is None:
            print(f"Skipping record {index}: no image found.")
            continue

        pil_image = image.convert("RGB")
        mask, room_polygons, metadata, corner_points = extract_room_polygons(pil_image)
        rgb_image = np.array(pil_image)
        _, blended = colourise_rooms(rgb_image, room_polygons, corner_points)

        identifier = _normalise_identifier(_derive_record_identifier(record, index))
        floor_id = index + 1  # 1-based indexing for floor plans

        sample_dir = output_dir / f"floor{floor_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        mask_path = sample_dir / "room_mask.png"
        overlay_path = sample_dir / "overlay.png"
        json_path = sample_dir / "metadata.json"

        Image.fromarray(mask.astype(np.uint8)).save(mask_path)
        Image.fromarray(blended).save(overlay_path)

        # Extract supporting text if available and this record has text
        supporting_text = None
        if resolved_text_column and floor_id not in no_text_ids:
            text_value = record.get(resolved_text_column)
            if text_value is not None:
                if isinstance(text_value, str) and text_value.strip():
                    supporting_text = text_value.strip()
                elif isinstance(text_value, (list, tuple)) and len(text_value) > 0:
                    # Handle list of strings
                    text_str = " ".join(str(item).strip() for item in text_value if item)
                    if text_str.strip():
                        supporting_text = text_str.strip()

        payload = {
            "identifier": identifier,
            "record_index": index,
            "floor_id": floor_id,
            "floor_folder": sample_dir.name,
            "image_size": {"width": int(rgb_image.shape[1]), "height": int(rgb_image.shape[0])},
            "room_count": len(metadata),
            "rooms": _prepare_room_metadata(metadata),
            "corner_points": [list(point) for point in corner_points],
            "generated_files": {
                "room_mask_png": mask_path.name,
                "overlay_png": overlay_path.name,
                "metadata_json": json_path.name,
            },
        }
        
        # Add supporting_text at the top if available
        if supporting_text is not None:
            payload = {"supporting_text": supporting_text, **payload}

        # Write metadata with versioning support
        write_versioned_metadata_json(json_path, payload)

        if (index + 1) % 25 == 0 or index == total_records - 1:
            print(f"Processed {index + 1}/{total_records} samples")
    
    print()
    print("=" * 50)
    print("Processing complete!")
    print(f"Processed data saved to: {output_dir}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format FloorPlans970 dataset: download, load, extract masks, and save to processed."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_LOCAL_DATASET_PATH,
        help="Path to a locally downloaded dataset saved via `datasets.save_to_disk`.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset identifier to download if no local copy exists.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to load when the dataset provides multiple splits.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default=None,
        help="Override image column in the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save processed floor plans (default: backend/processed/floor_plans).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if dataset exists locally.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the data formatting workflow."""
    args = parse_args()
    process_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        image_column=args.image_column,
        output_dir=args.output_dir,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

