"""
app/deep_quality/evaluate.py

Compares QualityCNN against the classical heuristic detectors (Blur,
Darkness, Glare) on the KonIQ-10k test set, using Pearson correlation
against human-rated MOS as the benchmark metric (per the dataset's own
suggested benchmarking approach).

The classical detectors only produce boolean flags individually, so a
continuous "combined quality score" is built from their raw sub-scores
(normalized components, equal-weight average) for a fair comparison
against the CNN's continuous MOS prediction.

Run with:
    python -m app.deep_quality.evaluate
"""

import logging

import cv2
import pandas as pd
import torch
from scipy.stats import pearsonr

from app.core.logging_config import setup_logging
from app.deep_quality.dataset import KonIQDataset, IMAGE_SIZE
from app.deep_quality.model import QualityCNN
from app.quality.blur_detector import BlurDetector
from app.quality.darkness_detector import DarknessDetector
from app.quality.glare_detector import GlareDetector

setup_logging()
logger = logging.getLogger(__name__)

CSV_PATH = "data/raw/koniq10k_distributions_sets.csv"
IMAGE_DIR = "/home/mlengr/Downloads/archive (1)/512x384"
MODEL_PATH = "models/quality_cnn.pt"

blur_detector = BlurDetector()
dark_detector = DarknessDetector()
glare_detector = GlareDetector()


def classical_combined_score(image_path: str) -> float:
    """
    Builds a continuous 0-1 "quality score" from the three classical
    detectors' raw sub-scores, normalized and equal-weight averaged.
    Higher = better quality (matching MOS convention).
    """
    img = cv2.imread(image_path)

    blur_raw = blur_detector.compute_score(img)
    dark_raw = dark_detector.compute_score(img)
    glare_raw = glare_detector.compute_score(img)

    blur_component = min(blur_raw / 100.0, 1.0)      # higher variance = sharper = better
    dark_component = dark_raw / 255.0                 # higher brightness = less dark = better
    glare_component = 1.0 - min(glare_raw, 1.0)        # higher glare = worse, so invert

    return (blur_component + dark_component + glare_component) / 3.0


def main():
    logger.info("Loading test set...")
    test_ds = KonIQDataset(CSV_PATH, IMAGE_DIR, split="test")
    df = test_ds.df  # has image_name, MOS, etc.

    logger.info("Loading trained CNN...")
    model = QualityCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    cnn_predictions = []
    classical_scores = []
    actual_mos = []

    logger.info(f"Evaluating {len(df)} test images...")
    for idx in range(len(df)):
        row = df.iloc[idx]
        img_tensor, _ = test_ds[idx]
        img_path = f"{IMAGE_DIR}/{row['image_name']}"

        with torch.no_grad():
            cnn_pred = model(img_tensor.unsqueeze(0)).item()  # add batch dim

        classical_score = classical_combined_score(img_path)

        cnn_predictions.append(cnn_pred)
        classical_scores.append(classical_score)
        actual_mos.append(row["MOS"] / 100.0)

        if (idx + 1) % 500 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(df)}")

    cnn_corr, cnn_pvalue = pearsonr(cnn_predictions, actual_mos)
    classical_corr, classical_pvalue = pearsonr(classical_scores, actual_mos)

    logger.info("=== Results ===")
    logger.info(f"CNN vs MOS:        Pearson r = {cnn_corr:.4f} (p={cnn_pvalue:.2e})")
    logger.info(f"Classical vs MOS:  Pearson r = {classical_corr:.4f} (p={classical_pvalue:.2e})")

    results_df = pd.DataFrame({
        "image_name": df["image_name"],
        "actual_mos": actual_mos,
        "cnn_prediction": cnn_predictions,
        "classical_score": classical_scores,
    })
    results_df.to_csv("data/raw/evaluation_results.csv", index=False)
    logger.info("Saved per-image results to data/raw/evaluation_results.csv")


if __name__ == "__main__":
    main()
