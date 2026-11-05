"""Stage 2: MCU-friendly TinyEdgeEEGNet baseline (100% CMSIS-NN / TFLite Micro compatible)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyEdgeEEGNet(nn.Module):
    """
    Lightweight Spatial-Temporal Separable CNN for Mandarin Tone / Phoneme decoding.
    Uses only Conv2d, DepthwiseConv2d, BatchNorm2d, ReLU6, AvgPool2d, AdaptiveAvgPool2d, and Linear.
    Total parameters < 2,500 (~2.2 KB in INT8).
    """

    def __init__(self, num_channels: int = 8, time_steps: int = 125, num_classes: int = 4) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16, momentum=0.3),
            nn.ReLU6(inplace=True),
        )
        self.spatial_depthwise = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, momentum=0.3),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32, momentum=0.3),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_depthwise(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
