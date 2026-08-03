"""
app/deep_quality/train.py

Trains QualityCNN on KonIQ-10k to predict MOS (0-1, normalized). Uses the
dataset's official train/validation/test split (from the 'set' column).

Run with:
    python -m app.deep_quality.train
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.logging_config import setup_logging
from app.deep_quality.dataset import KonIQDataset
from app.deep_quality.model import QualityCNN

setup_logging()
logger = logging.getLogger(__name__)

# ---- Config ----
CSV_PATH = "data/raw/koniq10k_distributions_sets.csv"
IMAGE_DIR = "/home/mlengr/Downloads/archive (1)/512x384"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/quality_cnn.pt"


def run_epoch(model, loader, criterion, optimizer=None) -> float:
    """
    Runs one pass over `loader`. If `optimizer` is given, trains (updates
    weights); otherwise just evaluates (used for validation/test, where
    weights must not change).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, targets in loader:
            if is_training:
                optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


def main():
    logger.info("Loading datasets...")
    train_ds = KonIQDataset(CSV_PATH, IMAGE_DIR, split="training")
    val_ds = KonIQDataset(CSV_PATH, IMAGE_DIR, split="validation")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = QualityCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info(f"Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)

        elapsed = time.time() - start
        logger.info(
            f"Epoch {epoch}/{EPOCHS} - train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  ({elapsed:.1f}s)"
        )

    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saved trained model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
