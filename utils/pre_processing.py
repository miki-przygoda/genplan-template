"""
Utilities for inspecting and pre-processing the FloorPlans970 dataset.

This module focuses on the first stage of TODO.md — auditing the raw data to
understand text coverage, confirm the expected 512x512 image resolution, and
optionally export a handful of samples for manual colour checks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - pillow is listed in requirements
    PILImage = None  # type: ignore


DEFAULT_DATASET_NAME = "HamzaWajid1/FloorPlans970Dataset"
DEFAULT_LOCAL_DATASET_PATH = Path("dataset") / "FloorPlans970Dataset"
DEFAULT_PROCESSED_DIR = Path("processed")
DEFAULT_NO_TEXT_PATH = DEFAULT_PROCESSED_DIR / "no_text_ids.json"
EXPECTED_IMAGE_SHAPE = (512, 512)

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
    if dataset_path.exists() and any(dataset_path.iterdir()):
        dataset = load_from_disk(str(dataset_path))
    else:
        dataset = load_dataset(dataset_name, verification_mode="no_checks")

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


def detect_missing_text_records(
    dataset: Dataset,
    *,
    text_column: str,
    id_column: Optional[str] = None,
) -> List[Any]:
    """Return identifiers for records with missing or empty supporting text."""
    missing_ids: List[Any] = []
    for idx, record in enumerate(dataset):
        text_value = record.get(text_column)
        is_missing = False

        if text_value is None:
            is_missing = True
        elif isinstance(text_value, str):
            is_missing = text_value.strip() == ""
        elif isinstance(text_value, (list, tuple, set)):
            is_missing = all((not item or str(item).strip() == "") for item in text_value)
        else:
            is_missing = False

        if is_missing:
            identifier = record.get(id_column, idx) if id_column else idx
            missing_ids.append(identifier)

    return missing_ids


def _extract_image_dimensions(image: Any) -> Tuple[int, int]:
    """Return (width, height) for an image in any common HF dataset format."""
    if PILImage is not None and isinstance(image, PILImage.Image):
        return image.size

    # Datasets.Image feature yields dictionaries when decoding is disabled.
    if isinstance(image, dict) and "array" in image:
        array = image["array"]
        if hasattr(array, "shape") and len(array.shape) >= 2:
            height, width = array.shape[0], array.shape[1]
            return width, height

    if hasattr(image, "shape"):
        height, width = image.shape[0], image.shape[1]
        return width, height

    raise DataInspectionError(f"Unsupported image type for shape extraction: {type(image)}")


def validate_image_shapes(
    dataset: Dataset,
    *,
    image_column: str,
    expected_shape: Tuple[int, int] = EXPECTED_IMAGE_SHAPE,
    id_column: Optional[str] = None,
) -> List[Tuple[Any, Tuple[int, int]]]:
    """
    Check that all images in the dataset match the expected width/height.

    Returns a list of (identifier, (width, height)) for mismatched samples.
    """
    mismatches: List[Tuple[Any, Tuple[int, int]]] = []
    expected_width, expected_height = expected_shape

    for idx, record in enumerate(dataset):
        identifier = record.get(id_column, idx) if id_column else idx
        image = record.get(image_column)

        if image is None:
            mismatches.append((identifier, (0, 0)))
            continue

        width, height = _extract_image_dimensions(image)
        if width != expected_width or height != expected_height:
            mismatches.append((identifier, (width, height)))

    return mismatches


def export_sample_images(
    dataset: Dataset,
    *,
    image_column: str,
    destination: Path,
    id_column: Optional[str] = None,
    num_samples: int = 10,
    seed: int = 42,
) -> List[Path]:
    """Write a random subset of images to disk for manual inspection."""
    if num_samples <= 0:
        return []

    if PILImage is None:
        raise DataInspectionError("Pillow is required to export sample images.")

    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    written_paths: List[Path] = []
    for idx in indices[:num_samples]:
        record = dataset[idx]
        image = record.get(image_column)
        if image is None:
            continue

        # Datasets.Image returns a PIL image by default. If not, convert.
        if not isinstance(image, PILImage.Image):
            width, height = _extract_image_dimensions(image)
            if isinstance(image, dict) and "array" in image:
                pil_image = PILImage.fromarray(image["array"])
            elif hasattr(image, "numpy"):  # e.g. torch Tensor
                pil_image = PILImage.fromarray(image.numpy())
            else:
                raise DataInspectionError(
                    f"Cannot convert image of type {type(image)} to PIL.Image."
                )
        else:
            pil_image = image

        identifier = (
            record.get(id_column, f"sample_{idx:04d}") if id_column else f"sample_{idx:04d}"
        )
        safe_identifier = str(identifier).replace("/", "_")
        output_path = destination / f"{safe_identifier}.png"
        pil_image.save(output_path)
        written_paths.append(output_path)

    return written_paths


def write_json(path: Path, data: Any) -> None:
    """Persist a JSON payload to disk with pretty indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_data_audit(
    dataset_path: Optional[Path],
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: Optional[str] = "train",
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    image_column: Optional[str] = None,
    expected_shape: Tuple[int, int] = EXPECTED_IMAGE_SHAPE,
    no_text_output: Optional[Path] = DEFAULT_NO_TEXT_PATH,
    sample_output_dir: Optional[Path] = None,
    sample_count: int = 0,
) -> dict[str, Any]:
    """Run the full data inspection workflow and return a summary."""
    dataset = load_floorplans_dataset(dataset_path, dataset_name=dataset_name, split=split)

    resolved_text_column = text_column or resolve_column(
        dataset.column_names, PREFERRED_TEXT_COLUMNS
    )
    try:
        resolved_id_column = id_column or resolve_column(
            dataset.column_names, PREFERRED_ID_COLUMNS
        )
    except DataInspectionError:
        resolved_id_column = None
    resolved_image_column = image_column or resolve_column(
        dataset.column_names, PREFERRED_IMAGE_COLUMNS
    )

    missing_text_ids = detect_missing_text_records(
        dataset, text_column=resolved_text_column, id_column=resolved_id_column
    )
    mismatched_shapes = validate_image_shapes(
        dataset,
        image_column=resolved_image_column,
        expected_shape=expected_shape,
        id_column=resolved_id_column,
    )

    if no_text_output is not None:
        write_json(no_text_output, missing_text_ids)

    exported_samples: List[Path] = []
    if sample_output_dir is not None and sample_count > 0:
        exported_samples = export_sample_images(
            dataset,
            image_column=resolved_image_column,
            destination=sample_output_dir,
            id_column=resolved_id_column,
            num_samples=sample_count,
        )

    summary = {
        "dataset_name": dataset_name,
        "split": split,
        "num_records": len(dataset),
        "text_column": resolved_text_column,
        "id_column": resolved_id_column,
        "image_column": resolved_image_column,
        "missing_text_count": len(missing_text_ids),
        "missing_text_ids_path": str(no_text_output) if no_text_output else None,
        "mismatched_image_count": len(mismatched_shapes),
        "mismatched_image_details": mismatched_shapes,
        "exported_sample_paths": [str(path) for path in exported_samples],
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the FloorPlans970 dataset for missing text and image shape issues."
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
    parser.add_argument("--text-column", type=str, default=None, help="Override text column name.")
    parser.add_argument("--id-column", type=str, default=None, help="Override identifier column.")
    parser.add_argument(
        "--image-column", type=str, default=None, help="Override image column in the dataset."
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=EXPECTED_IMAGE_SHAPE[0],
        help="Expected image width for shape validation.",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=EXPECTED_IMAGE_SHAPE[1],
        help="Expected image height for shape validation.",
    )
    parser.add_argument(
        "--no-text-output",
        type=Path,
        default=DEFAULT_NO_TEXT_PATH,
        help="Where to write the list of IDs without supporting text.",
    )
    parser.add_argument(
        "--sample-output-dir",
        type=Path,
        default=None,
        help="Directory to export sample images for manual inspection.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of random images to export for manual inspection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_data_audit(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        id_column=args.id_column,
        image_column=args.image_column,
        expected_shape=(args.expected_width, args.expected_height),
        no_text_output=args.no_text_output,
        sample_output_dir=args.sample_output_dir,
        sample_count=args.sample_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
