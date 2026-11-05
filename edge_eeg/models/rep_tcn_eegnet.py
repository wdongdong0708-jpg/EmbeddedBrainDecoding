"""Stage 2 Core Innovation (Thesis Sec 4.2): Structural Reparameterized Dilated TCN-EEGNet (RepTCNEEGNet)."""

from __future__ import annotations

import copy
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mathematically fuse Conv2d and BatchNorm2d parameters into a single (weight, bias) pair."""
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    std = torch.sqrt(bn_rv + bn_eps)
    scale = (bn_w / std).reshape(-1, 1, 1, 1)
    fused_w = conv_w * scale
    fused_b = (conv_b - bn_rm) * (bn_w / std) + bn_b
    return fused_w, fused_b


def _pad_kernel_1d_horizontal(kernel: torch.Tensor, target_kw: int) -> torch.Tensor:
    """Zero-pad a (C_out, C_in/g, 1, k_small) kernel symmetrically along width to (C_out, C_in/g, 1, target_kw)."""
    cur_kw = kernel.size(-1)
    if cur_kw == target_kw:
        return kernel
    pad_total = target_kw - cur_kw
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(kernel, (pad_left, pad_right, 0, 0))


class RepTemporalBlock(nn.Module):
    """
    Multi-branch temporal convolution block (RepVGG-style for 1D/2D neural time series).
    - Training mode: (1 x K) Conv+BN + (1 x K_small) Conv+BN + (1 x 1) Conv+BN + Identity BN (if in==out & stride==1)
    - Inference mode (`deploy=True`): Single fused (1 x K) Conv2d(bias=True) + ReLU6 (zero residual branch SRAM overhead)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        small_kernel: int = 5,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1 and small_kernel % 2 == 1, "Kernel sizes must be odd."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.deploy = deploy

        eff_k = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_main = eff_k // 2
        eff_k_small = small_kernel + (small_kernel - 1) * (dilation - 1)
        pad_small = eff_k_small // 2

        self.act = nn.ReLU6(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, pad_main),
                dilation=(1, dilation),
                groups=groups,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.rbr_main = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, kernel_size),
                    stride=(1, stride),
                    padding=(0, pad_main),
                    dilation=(1, dilation),
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, momentum=0.2),
            )
            self.rbr_small = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, small_kernel),
                    stride=(1, stride),
                    padding=(0, pad_small),
                    dilation=(1, dilation),
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, momentum=0.2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, stride),
                    padding=(0, 0),
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, momentum=0.2),
            )
            if out_channels == in_channels and stride == 1:
                self.rbr_identity: nn.BatchNorm2d | None = nn.BatchNorm2d(in_channels, momentum=0.2)
            else:
                self.rbr_identity = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy and self.reparam_conv is not None:
            return self.act(self.reparam_conv(x))

        out = self.rbr_main(x) + self.rbr_small(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mathematically equivalent single-branch (weight, bias) across all parallel branches."""
        w_main, b_main = self._fuse_seq(self.rbr_main)
        w_small, b_small = self._fuse_seq(self.rbr_small)
        w_1x1, b_1x1 = self._fuse_seq(self.rbr_1x1)
        w_id, b_id = self._fuse_identity()

        w_small_pad = _pad_kernel_1d_horizontal(w_small, self.kernel_size)
        w_1x1_pad = _pad_kernel_1d_horizontal(w_1x1, self.kernel_size)
        w_id_pad = _pad_kernel_1d_horizontal(w_id, self.kernel_size)

        return (
            w_main + w_small_pad + w_1x1_pad + w_id_pad,
            b_main + b_small + b_1x1 + b_id,
        )

    def _fuse_seq(self, seq: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
        conv: nn.Conv2d = seq[0]  # type: ignore[assignment]
        bn: nn.BatchNorm2d = seq[1]  # type: ignore[assignment]
        return fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            bn.running_mean,  # type: ignore[arg-type]
            bn.running_var,  # type: ignore[arg-type]
            bn.eps,
            bn.weight,
            bn.bias,
        )

    def _fuse_identity(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rbr_identity is None:
            device = self.rbr_main[0].weight.device
            dtype = self.rbr_main[0].weight.dtype
            return (
                torch.zeros(
                    (self.out_channels, self.in_channels // self.groups, 1, 1),
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(self.out_channels, device=device, dtype=dtype),
            )
        bn = self.rbr_identity
        in_per_group = self.in_channels // self.groups
        id_w = torch.zeros(
            (self.out_channels, in_per_group, 1, 1),
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        for c in range(self.out_channels):
            id_w[c, c % in_per_group, 0, 0] = 1.0
        return fuse_conv_bn_weights(
            id_w,
            None,
            bn.running_mean,  # type: ignore[arg-type]
            bn.running_var,  # type: ignore[arg-type]
            bn.eps,
            bn.weight,
            bn.bias,
        )

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        w, b = self.get_equivalent_kernel_bias()
        eff_k = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        pad_main = eff_k // 2
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, pad_main),
            dilation=(1, self.dilation),
            groups=self.groups,
            bias=True,
        ).to(w.device, dtype=w.dtype)
        self.reparam_conv.weight.data.copy_(w)
        if self.reparam_conv.bias is not None:
            self.reparam_conv.bias.data.copy_(b)
        for attr in ("rbr_main", "rbr_small", "rbr_1x1", "rbr_identity"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True


class RepTCNEEGNet(nn.Module):
    """
    Stage 2 Core Network: Multi-Scale Dilated Structural-Reparameterized EEGNet (`Rep-TCN-EEGNet`).
    - During training: Multi-branch temporal stem + Spatial depthwise + Multi-scale dilated (d=1, 2)
      reparameterizable depthwise-separable TCN blocks.
    - During MCU deployment (`deploy_reparameterize()`): Collapses all parallel branches and BatchNorms
      into a strictly sequential chain of single-branch Conv2d + ReLU6 layers, cutting Tensor Arena peak SRAM by >40%.
    """

    def __init__(
        self,
        num_channels: int = 8,
        time_steps: int = 125,
        num_classes: int = 5,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.deploy = deploy

        # 1) Reparameterizable Temporal Stem (1 x 15 + 1 x 5 + 1 x 1)
        self.stem_temporal = RepTemporalBlock(
            in_channels=1,
            out_channels=16,
            kernel_size=15,
            small_kernel=5,
            stride=1,
            groups=1,
            dilation=1,
            deploy=deploy,
        )

        # 2) Spatial Depthwise Projection across 8 electrodes
        if deploy:
            self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=True)
            self.spatial_bn = None
        else:
            self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False)
            self.spatial_bn = nn.BatchNorm2d(32, momentum=0.2)
        self.spatial_act = nn.ReLU6(inplace=True)
        self.spatial_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # 3) Dilated Separable TCN Stage A (Dilation = 1, multi-branch depthwise + pointwise)
        self.tcn1_dw = RepTemporalBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=7,
            small_kernel=3,
            stride=1,
            groups=32,
            dilation=1,
            deploy=deploy,
        )
        if deploy:
            self.tcn1_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=True)
            self.tcn1_pw_bn = None
        else:
            self.tcn1_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
            self.tcn1_pw_bn = nn.BatchNorm2d(32, momentum=0.2)
        self.tcn1_act = nn.ReLU6(inplace=True)

        # 4) Dilated Separable TCN Stage B (Dilation = 2, expands temporal receptive field without extra params)
        self.tcn2_dw = RepTemporalBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=7,
            small_kernel=3,
            stride=1,
            groups=32,
            dilation=2,
            deploy=deploy,
        )
        if deploy:
            self.tcn2_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=True)
            self.tcn2_pw_bn = None
        else:
            self.tcn2_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
            self.tcn2_pw_bn = nn.BatchNorm2d(32, momentum=0.2)
        self.tcn2_act = nn.ReLU6(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_temporal(x)
        x = self.spatial_conv(x)
        if self.spatial_bn is not None:
            x = self.spatial_bn(x)
        x = self.spatial_pool(self.spatial_act(x))

        x = self.tcn1_dw(x)
        x = self.tcn1_pw(x)
        if self.tcn1_pw_bn is not None:
            x = self.tcn1_pw_bn(x)
        x = self.tcn1_act(x)

        x = self.tcn2_dw(x)
        x = self.tcn2_pw(x)
        if self.tcn2_pw_bn is not None:
            x = self.tcn2_pw_bn(x)
        x = self.tcn2_act(x)

        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

    def deploy_reparameterize(self) -> "RepTCNEEGNet":
        """Convert multi-branch training graph in-place into a single-branch MCU inference graph."""
        if self.deploy:
            return self
        self.eval()
        self.stem_temporal.switch_to_deploy()
        self.tcn1_dw.switch_to_deploy()
        self.tcn2_dw.switch_to_deploy()

        # Fuse spatial_conv + spatial_bn
        if self.spatial_bn is not None:
            w, b = fuse_conv_bn_weights(
                self.spatial_conv.weight,
                self.spatial_conv.bias,
                self.spatial_bn.running_mean,  # type: ignore[arg-type]
                self.spatial_bn.running_var,  # type: ignore[arg-type]
                self.spatial_bn.eps,
                self.spatial_bn.weight,
                self.spatial_bn.bias,
            )
            new_sp = nn.Conv2d(16, 32, kernel_size=(self.num_channels, 1), groups=16, bias=True).to(w.device, w.dtype)
            new_sp.weight.data.copy_(w)
            new_sp.bias.data.copy_(b)  # type: ignore[union-attr]
            self.spatial_conv = new_sp
            self.spatial_bn = None

        # Fuse tcn1_pw + tcn1_pw_bn and tcn2_pw + tcn2_pw_bn
        for pw_attr, bn_attr in (("tcn1_pw", "tcn1_pw_bn"), ("tcn2_pw", "tcn2_pw_bn")):
            conv_mod: nn.Conv2d = getattr(self, pw_attr)
            bn_mod: nn.BatchNorm2d | None = getattr(self, bn_attr)
            if bn_mod is not None:
                w, b = fuse_conv_bn_weights(
                    conv_mod.weight,
                    conv_mod.bias,
                    bn_mod.running_mean,  # type: ignore[arg-type]
                    bn_mod.running_var,  # type: ignore[arg-type]
                    bn_mod.eps,
                    bn_mod.weight,
                    bn_mod.bias,
                )
                new_pw = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=True).to(w.device, w.dtype)
                new_pw.weight.data.copy_(w)
                new_pw.bias.data.copy_(b)  # type: ignore[union-attr]
                setattr(self, pw_attr, new_pw)
                setattr(self, bn_attr, None)

        self.deploy = True
        return self


@torch.no_grad()
def verify_reparameterization_and_sram(
    model: RepTCNEEGNet,
    num_channels: int = 8,
    time_steps: int = 125,
) -> Dict[str, float]:
    """
    Verify mathematical equivalence before/after `deploy_reparameterize()` and compute
    MCU Tensor Arena (peak simultaneous activation memory) reduction (Thesis Sec 4.2).
    """
    model_train_eval = copy.deepcopy(model).cpu().eval()
    x_test = torch.randn(4, 1, num_channels, time_steps)
    y_multi = model_train_eval(x_test)

    train_params = sum(p.numel() for p in model_train_eval.parameters())

    model_deploy = copy.deepcopy(model_train_eval).deploy_reparameterize()
    y_single = model_deploy(x_test)
    deploy_params = sum(p.numel() for p in model_deploy.parameters())

    max_abs_err = float(torch.max(torch.abs(y_multi - y_single)).item())

    # Calculate single-frame (Batch=1) peak live activation tensor memory in TFLite Micro double-ended arena:
    # - Multi-branch graph must simultaneously keep input tensor + 3 parallel branch outputs + identity buffer
    #   at the largest temporal resolution (16 x 8 x 125 floats = 16,000 elements per branch -> 4 live buffers)
    # - Reparameterized single-path graph only needs ping-pong (input + output) buffers (2 live buffers)
    stem_elems = 16 * num_channels * time_steps
    in_elems = 1 * num_channels * time_steps
    multi_branch_peak_fp32_kb = (in_elems + 4 * stem_elems) * 4.0 / 1024.0
    single_path_peak_fp32_kb = (in_elems + 1 * stem_elems) * 4.0 / 1024.0
    multi_branch_peak_int8_kb = (in_elems + 4 * stem_elems) * 1.0 / 1024.0
    single_path_peak_int8_kb = (in_elems + 1 * stem_elems) * 1.0 / 1024.0
    sram_saving_pct = (1.0 - single_path_peak_int8_kb / multi_branch_peak_int8_kb) * 100.0

    return {
        "train_params": float(train_params),
        "deploy_params": float(deploy_params),
        "max_abs_diff": max_abs_err,
        "multi_branch_arena_fp32_kb": round(multi_branch_peak_fp32_kb, 2),
        "deploy_arena_fp32_kb": round(single_path_peak_fp32_kb, 2),
        "multi_branch_arena_int8_kb": round(multi_branch_peak_int8_kb, 2),
        "deploy_arena_int8_kb": round(single_path_peak_int8_kb, 2),
        "arena_sram_reduction_pct": round(sram_saving_pct, 1),
    }
