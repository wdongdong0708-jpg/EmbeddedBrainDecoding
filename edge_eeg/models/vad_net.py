"""Stage 1: Ultra-lightweight Neural VAD network (<5 KB) and sliding-window gating state machine."""

from __future__ import annotations

from collections import deque
from typing import Deque, List
import numpy as np
import torch
import torch.nn as nn


class MicroVADNet(nn.Module):
    """
    Stage 1 Neural Voice Activity Detection (Speech vs. Silence) watchdog model.
    Designed to reside permanently in MCU SRAM (<5 KB weights in INT8) using 4-8 temporal channels.
    """

    def __init__(self, num_channels: int = 8, time_steps: int = 125, num_classes: int = 2) -> None:
        super().__init__()
        # Large temporal kernel (1 x 25 @ 250Hz = 100ms) to capture delta/theta (1-8Hz) envelope phase-locking
        self.temporal_filter = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 25), stride=(1, 2), padding=(0, 12), bias=False),
            nn.BatchNorm2d(8, momentum=0.3),
            nn.ReLU6(inplace=True),
        )
        # Spatial depthwise projection across temporal electrodes
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(num_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16, momentum=0.3),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        # Depthwise-separable temporal refinement
        self.sep_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 7), padding=(0, 3), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16, momentum=0.3),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_filter(x)
        x = self.spatial_dw(x)
        x = self.sep_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VADSlidingGate:
    """
    Multi-frame smoothing gate + Pre-roll ring buffer (Thesis Sec 3.3).
    Prevents false triggers from transient ocular/motion artifacts while preserving syllable onset.
    """

    def __init__(
        self,
        window_size: int = 5,
        trigger_threshold: float = 0.65,
        preroll_frames: int = 3,
    ) -> None:
        self.window_size = window_size
        self.trigger_threshold = trigger_threshold
        self.prob_history: Deque[float] = deque(maxlen=window_size)
        self.preroll_buffer: Deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.is_awake: bool = False

    def step(self, frame_signal: np.ndarray, speech_prob: float) -> tuple[bool, List[np.ndarray]]:
        """
        Update state machine with new frame and its predicted speech probability.
        Returns:
            (is_awake, frames_to_process_in_stage2)
        """
        self.prob_history.append(float(speech_prob))
        avg_prob = float(np.mean(self.prob_history))

        if not self.is_awake:
            self.preroll_buffer.append(frame_signal)
            if len(self.prob_history) >= 2 and avg_prob >= self.trigger_threshold:
                self.is_awake = True
                flushed = list(self.preroll_buffer)
                self.preroll_buffer.clear()
                return True, flushed
            return False, []

        # Hysteresis release threshold (lower than trigger to prevent chattering)
        if avg_prob < (self.trigger_threshold - 0.25):
            self.is_awake = False
            self.preroll_buffer.append(frame_signal)
            return False, []

        return True, [frame_signal]
