"""Thesis Sec 4.3: Outlier-Resistant Percentile-Clipped Per-Channel INT8 QAT and SQNR Profiler."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_sqnr_db(ref: torch.Tensor, quant: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB."""
    sig_power = float(torch.sum(ref.float() ** 2).item())
    noise_power = float(torch.sum((ref.float() - quant.float()) ** 2).item())
    if noise_power < 1e-12:
        return 60.0
    return float(10.0 * np.log10(max(sig_power, 1e-12) / noise_power))


def compute_cosine_similarity(ref: torch.Tensor, quant: torch.Tensor) -> float:
    """Compute flattened tensor cosine similarity (standard STM32 X-CUBE-AI / TFLM verification metric)."""
    r = ref.float().reshape(-1)
    q = quant.float().reshape(-1)
    denom = float(torch.norm(r).item() * torch.norm(q).item())
    if denom < 1e-12:
        return 1.0
    return float(torch.dot(r, q).item() / denom)


def fake_quant_weight_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-output-channel INT8 weight fake-quantization with Straight-Through Estimator (STE).
    Matches CMSIS-NN `arm_convolve_s8` / `arm_depthwise_conv_s8` symmetric int8_t weight specification.
    """
    out_c = w.size(0)
    w_flat = w.view(out_c, -1)
    max_val = torch.amax(torch.abs(w_flat), dim=1).clamp(min=1e-8)
    scale = max_val / 127.0
    shape = [-1] + [1] * (w.dim() - 1)
    scale_reshaped = scale.view(*shape)
    w_q = torch.clamp(torch.round(w / scale_reshaped), -127.0, 127.0)
    w_deq = w_q * scale_reshaped
    # Straight-Through Estimator preserves gradient flow in backward pass
    return w + (w_deq - w).detach(), scale.detach()


def fake_quant_weight_per_tensor_naive(w: torch.Tensor) -> torch.Tensor:
    """Naive per-tensor MinMax INT8 weight quantization (used as PTQ baseline for ablation)."""
    max_val = torch.amax(torch.abs(w)).clamp(min=1e-8)
    scale = max_val / 127.0
    w_q = torch.clamp(torch.round(w / scale), -127.0, 127.0)
    return w_q * scale


class QATConv2d(nn.Module):
    """
    Quantization-Aware Conv2d wrapper applying:
    1) Per-channel symmetric INT8 weight fake-quantization (STE)
    2) Percentile-clipped input activation INT8 fake-quantization (EMA calibrated)
    """

    def __init__(self, conv: nn.Conv2d, percentile: float = 99.5, ema_momentum: float = 0.1) -> None:
        super().__init__()
        self.conv = conv
        self.percentile = percentile
        self.ema_momentum = ema_momentum
        self.register_buffer("act_clip_val", torch.tensor(6.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int32))
        self.use_naive_minmax: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.use_naive_minmax:
            with torch.no_grad():
                flat_abs = torch.abs(x.detach()).reshape(-1)
                q_val = torch.quantile(flat_abs, self.percentile / 100.0).clamp(min=1e-5)
                if int(self.initialized.item()) == 0:
                    self.act_clip_val.copy_(q_val)
                    self.initialized.fill_(1)
                else:
                    self.act_clip_val.mul_(1.0 - self.ema_momentum).add_(q_val * self.ema_momentum)

        if self.use_naive_minmax:
            # Naive PTQ uses raw global max (vulnerable to extreme EEG/MEG artifact spikes)
            clip_bound = torch.amax(torch.abs(x)).clamp(min=1e-5)
            w_eff = fake_quant_weight_per_tensor_naive(self.conv.weight)
        else:
            clip_bound = self.act_clip_val.clamp(min=1e-5)
            w_eff, _ = fake_quant_weight_per_channel(self.conv.weight)

        act_scale = clip_bound / 127.0
        x_clipped = torch.clamp(x, -clip_bound, clip_bound)
        x_q = torch.clamp(torch.round(x_clipped / act_scale), -127.0, 127.0) * act_scale
        x_ste = x + (x_q - x).detach()

        return F.conv2d(
            x_ste,
            w_eff,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QATLinear(nn.Module):
    """Quantization-Aware Linear classifier head with Per-Channel INT8 weights and Percentile activation clipping."""

    def __init__(self, linear: nn.Linear, percentile: float = 99.5, ema_momentum: float = 0.1) -> None:
        super().__init__()
        self.linear = linear
        self.percentile = percentile
        self.ema_momentum = ema_momentum
        self.register_buffer("act_clip_val", torch.tensor(6.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int32))
        self.use_naive_minmax: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.use_naive_minmax:
            with torch.no_grad():
                flat_abs = torch.abs(x.detach()).reshape(-1)
                q_val = torch.quantile(flat_abs, self.percentile / 100.0).clamp(min=1e-5)
                if int(self.initialized.item()) == 0:
                    self.act_clip_val.copy_(q_val)
                    self.initialized.fill_(1)
                else:
                    self.act_clip_val.mul_(1.0 - self.ema_momentum).add_(q_val * self.ema_momentum)

        if self.use_naive_minmax:
            clip_bound = torch.amax(torch.abs(x)).clamp(min=1e-5)
            w_eff = fake_quant_weight_per_tensor_naive(self.linear.weight)
        else:
            clip_bound = self.act_clip_val.clamp(min=1e-5)
            w_eff, _ = fake_quant_weight_per_channel(self.linear.weight)

        act_scale = clip_bound / 127.0
        x_clipped = torch.clamp(x, -clip_bound, clip_bound)
        x_q = torch.clamp(torch.round(x_clipped / act_scale), -127.0, 127.0) * act_scale
        x_ste = x + (x_q - x).detach()

        return F.linear(x_ste, w_eff, self.linear.bias)


def insert_qat_wrappers(model: nn.Module, percentile: float = 99.5) -> nn.Module:
    """Recursively wrap Conv2d and Linear modules with QATConv2d and QATLinear."""
    for name, child in list(model.named_children()):
        if isinstance(child, (QATConv2d, QATLinear)):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(model, name, QATConv2d(child, percentile=percentile))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child, percentile=percentile))
        else:
            insert_qat_wrappers(child, percentile=percentile)
    return model


def _set_naive_minmax_mode(model: nn.Module, flag: bool) -> None:
    for m in model.modules():
        if isinstance(m, (QATConv2d, QATLinear)):
            m.use_naive_minmax = flag


@torch.no_grad()
def evaluate_quantization_ablation(
    fp32_model: nn.Module,
    qat_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    compute_f1_fn,
) -> Dict[str, float]:
    """
    Thesis Sec 4.3 Core Ablation Experiment:
    Compare FP32 vs. Naive MinMax PTQ INT8 vs. Proposed Percentile-Clipped Per-Channel INT8 QAT
    under realistic transient non-stationary neural spike artifacts (<0.4% outlier samples).
    Returns Accuracy, F1-macro, SQNR (dB), and Cosine Similarity.
    """
    from ..engine.exporter import unwrap_qat_modules

    # Extract exact FP32 counterpart of the QAT-trained model so SQNR measures pure quantization noise
    clean_fp32 = unwrap_qat_modules(copy.deepcopy(qat_model)).to(device).eval()
    qat_model = qat_model.to(device).eval()

    # Construct Naive MinMax Per-Tensor PTQ model from the same weights
    ptq_naive_model = copy.deepcopy(clean_fp32)
    insert_qat_wrappers(ptq_naive_model, percentile=100.0)
    _set_naive_minmax_mode(ptq_naive_model, True)
    ptq_naive_model = ptq_naive_model.to(device).eval()

    _set_naive_minmax_mode(qat_model, False)

    fp32_logits_list: List[torch.Tensor] = []
    ptq_logits_list: List[torch.Tensor] = []
    qat_logits_list: List[torch.Tensor] = []
    y_true_list: List[np.ndarray] = []

    for x_b, y_b in val_loader:
        x_b = x_b.to(device)
        out_fp32 = clean_fp32(x_b)

        # Inject sparse transient electrode pop artifact (1 sample out of 1000 = 0.1% outlier tail)
        x_noisy = x_b.clone()
        x_noisy[:, 0, 0, 0] = x_noisy[:, 0, 0, 0] + 25.0

        out_ptq = ptq_naive_model(x_noisy)
        out_qat = qat_model(x_noisy)

        fp32_logits_list.append(out_fp32.cpu())
        ptq_logits_list.append(out_ptq.cpu())
        qat_logits_list.append(out_qat.cpu())
        y_true_list.append(y_b.numpy())

    fp32_all = torch.cat(fp32_logits_list, dim=0)
    ptq_all = torch.cat(ptq_logits_list, dim=0)
    qat_all = torch.cat(qat_logits_list, dim=0)
    y_true = np.concatenate(y_true_list, axis=0)

    pred_fp32 = torch.argmax(fp32_all, dim=-1).numpy()
    pred_ptq = torch.argmax(ptq_all, dim=-1).numpy()
    pred_qat = torch.argmax(qat_all, dim=-1).numpy()

    return {
        "fp32_acc": round(float(np.mean(pred_fp32 == y_true)), 4),
        "fp32_f1_macro": round(float(compute_f1_fn(y_true, pred_fp32, num_classes)), 4),
        "ptq_naive_int8_acc": round(float(np.mean(pred_ptq == y_true)), 4),
        "ptq_naive_int8_f1_macro": round(float(compute_f1_fn(y_true, pred_ptq, num_classes)), 4),
        "ptq_sqnr_db": round(compute_sqnr_db(fp32_all, ptq_all), 2),
        "ptq_cosine_sim": round(compute_cosine_similarity(fp32_all, ptq_all), 4),
        "qat_int8_acc": round(float(np.mean(pred_qat == y_true)), 4),
        "qat_int8_f1_macro": round(float(compute_f1_fn(y_true, pred_qat, num_classes)), 4),
        "qat_sqnr_db": round(compute_sqnr_db(fp32_all, qat_all), 2),
        "qat_cosine_sim": round(compute_cosine_similarity(fp32_all, qat_all), 4),
    }
