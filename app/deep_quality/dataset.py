"""
app/deep_quality/dataset.py

PyTorch Dataset for KonIQ-10k: loads an image and its human-rated MOS
(Mean Opinion Score) as a regression target.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_SIZE = 64  # small on purpose — CPU training, keep it fast


class KonIQDataset(Dataset):
    """
    Wraps the KonIQ-10k CSV + image folder. Filters rows by the CSV's own
    'set' column ('training', 'validation', or 'test') so all three splits
    come from the same file, matching the dataset's official split.
    """

    def __init__(self, csv_path: str, image_dir: str, split: str):
        df = pd.read_csv(csv_path)
        self.df = df[df["set"] == split].reset_index(drop=True)
        self.image_dir = Path(image_dir)

        logger.info(f"KonIQDataset[{split}]: {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_name"]

        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.array(img, dtype=np.float32) / 255.0  # 0-1 range

        # HWC -> CHW (PyTorch convolutions expect channels first)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # MOS is on a 0-100 scale in this dataset; normalize to 0-1 to
        # match the classical detectors' score ranges and keep the loss
        # magnitude small and stable during training.
        mos_normalized = row["MOS"] / 100.0
        mos_tensor = torch.tensor(mos_normalized, dtype=torch.float32)

        return img_tensor, mos_tensor
