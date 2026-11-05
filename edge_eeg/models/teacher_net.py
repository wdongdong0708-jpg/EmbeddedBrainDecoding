"""Phase 1 Oracle Upper-Bound Teacher Model: Multi-Scale Dilated TCN-Conformer for 306-ch MEG."""

from __future__ import annotations

import torch
import torch.nn as nn


class _DilatedTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=pad, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TeacherConformerTCN(nn.Module):
    """
    High-capacity GPU Teacher model for 306-channel LibriBrain MEG (Phase 1 Upper Bound).
    Combines spatial sensor projection, multi-scale dilated TCN (d=1,2,4), and Transformer self-attention.
    """

    def __init__(
        self,
        num_channels: int = 306,
        time_steps: int = 125,
        num_classes: int = 4,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        # Spatial projection from 306 MEG sensors -> d_model latent channels
        self.spatial_stem = nn.Sequential(
            nn.Conv1d(num_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # Multi-scale dilated temporal convolution stack (d = 1, 2, 4)
        self.tcn_stack = nn.Sequential(
            _DilatedTCNBlock(d_model, dilation=1),
            _DilatedTCNBlock(d_model, dilation=2),
            _DilatedTCNBlock(d_model, dilation=4),
        )
        # Lightweight self-attention layer to capture long-range phoneme context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Linear(d_model * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [Batch, 1, Channels, Time] -> squeeze to [Batch, Channels, Time]
        if x.dim() == 4:
            x = x.squeeze(1)
        feat = self.spatial_stem(x)
        feat = self.tcn_stack(feat)
        # Transformer expects [Batch, Time, d_model]
        feat_t = feat.transpose(1, 2)
        feat_t = self.attention(feat_t)
        feat = feat_t.transpose(1, 2)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(pooled)
