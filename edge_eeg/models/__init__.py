"""Model factory for Stage 1 (VAD) and Stage 2 (Rep-TCN-EEGNet, TinyEdgeEEGNet, & TeacherConformerTCN)."""

from __future__ import annotations

import torch.nn as nn

from .quantization import (
    QATConv2d,
    QATLinear,
    compute_cosine_similarity,
    compute_sqnr_db,
    evaluate_quantization_ablation,
    insert_qat_wrappers,
)
from .rep_tcn_eegnet import RepTCNEEGNet, verify_reparameterization_and_sram
from .teacher_net import TeacherConformerTCN
from .tiny_eegnet import TinyEdgeEEGNet
from .vad_net import MicroVADNet, VADSlidingGate


def build_model(
    stage: str,
    arch: str,
    num_channels: int,
    time_steps: int,
    num_classes: int,
) -> nn.Module:
    """Instantiate the target architecture for the given stage and channel configuration."""
    if arch == "teacher_conformer":
        return TeacherConformerTCN(
            num_channels=num_channels,
            time_steps=time_steps,
            num_classes=num_classes,
        )
    if stage == "vad":
        return MicroVADNet(
            num_channels=num_channels,
            time_steps=time_steps,
            num_classes=num_classes,
        )
    if arch == "rep_tcn":
        return RepTCNEEGNet(
            num_channels=num_channels,
            time_steps=time_steps,
            num_classes=num_classes,
        )
    return TinyEdgeEEGNet(
        num_channels=num_channels,
        time_steps=time_steps,
        num_classes=num_classes,
    )


__all__ = [
    "MicroVADNet",
    "VADSlidingGate",
    "TinyEdgeEEGNet",
    "RepTCNEEGNet",
    "TeacherConformerTCN",
    "QATConv2d",
    "QATLinear",
    "build_model",
    "compute_cosine_similarity",
    "compute_sqnr_db",
    "evaluate_quantization_ablation",
    "insert_qat_wrappers",
    "verify_reparameterization_and_sram",
]
