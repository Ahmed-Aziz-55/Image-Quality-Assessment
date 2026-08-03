"""
app/deep_quality/model.py

A small CNN for image quality regression (predicts a single MOS score
0-1). Kept intentionally shallow — this trains on CPU, and the goal is a
baseline for comparison against classical heuristics, not state-of-the-art
accuracy.
"""

import torch
import torch.nn as nn


class QualityCNN(nn.Module):
    """
    Three conv blocks (conv -> ReLU -> maxpool) reduce a 64x64x3 image
    down to a small feature map, which is flattened and passed through
    two fully-connected layers to a single regression output.
    """

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1: 64x64x3 -> 32x32x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32x32x16 -> 16x16x32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 16x16x32 -> 8x8x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # After 3 poolings: 64 -> 32 -> 16 -> 8, so feature map is 8x8x64
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # randomly zeroes 30% of activations during
                               # training to reduce overfitting
            nn.Linear(128, 1),
            nn.Sigmoid(),  # squashes output to 0-1, matching normalized MOS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze(1)  # (batch, 1) -> (batch,) to match MOS tensor shape
