"""
app/training/generate_training_dataset.py

Builds the training dataset for the Suitability Model:
  clean image -> SyntheticAugmentationEngine.augment() -> degraded image + injections
              -> ground_truth_generator.generate_label(injections) -> label
              -> FeatureExtractor.extract_vector(degraded image) -> 8 features

Saves one row per image: 8 feature columns + label. Labels come only
from injection metadata (independent of detector output) -- see the
golden rule documented in synthetic_augmentation.py.

Run with:
    python -m app.training.generate_training_dataset
"""

import csv
import logging
import random
from pathlib import Path

import cv2

from app.core.logging_config import setup_logging
from app.quality.feature_extractor import FeatureExtractor, FEATURE_NAMES
from app.training.synthetic_augmentation import SyntheticAugmentationEngine
from app.training.ground_truth_generator import generate_label

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_DIR = "data/raw/512x384"
OUTPUT_CSV = "data/raw/suitability_training_data.csv"
SEED = 42


def main():
    image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} source images")

    engine = SyntheticAugmentationEngine(seed=SEED)
    extractor = FeatureExtractor()

    rows = []
    failed = 0

    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            failed += 1
            continue

        degraded, injections = engine.augment(img)
        label = generate_label(injections)
        feature_vector = extractor.extract_vector(degraded)

        row = dict(zip(FEATURE_NAMES, feature_vector))
        row["label"] = label
        row["source_image"] = path.name
        rows.append(row)

        if (idx + 1) % 1000 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(image_paths)}")

    logger.info(f"Processed {len(rows)} images ({failed} failed to load)")

    suitable_count = sum(1 for r in rows if r["label"] == "Suitable")
    logger.info(f"Label distribution: Suitable={suitable_count} ({100*suitable_count/len(rows):.1f}%), "
                f"Not Suitable={len(rows)-suitable_count} ({100*(len(rows)-suitable_count)/len(rows):.1f}%)")

    fieldnames = FEATURE_NAMES + ["label", "source_image"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved training dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
