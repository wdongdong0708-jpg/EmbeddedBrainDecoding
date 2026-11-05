"""Training loop with Cross-Channel Knowledge Distillation, Structural Reparameterization, and INT8 QAT."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..models.quantization import evaluate_quantization_ablation, insert_qat_wrappers
from ..models.rep_tcn_eegnet import RepTCNEEGNet, verify_reparameterization_and_sram
from .exporter import export_model_to_c_header


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute unweighted macro-F1 score (official PNPL competition metric)."""
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        x_b, y_b = batch[0].to(device), batch[1].to(device)
        logits = model(x_b)
        loss = criterion(logits, y_b)
        bs = x_b.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_b.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros(1, dtype=np.int64)
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros(1, dtype=np.int64)
    acc = float(np.mean(y_pred == y_true))
    f1_macro = compute_f1_macro(y_true, y_pred, num_classes)
    return {
        "val_loss": total_loss / max(total_count, 1),
        "val_acc": acc,
        "val_f1_macro": f1_macro,
    }


def _attach_teacher_logits_to_loader(
    train_loader: DataLoader,
    teacher_logits: torch.Tensor,
) -> DataLoader:
    """Augment a training DataLoader(x, y) into DataLoader(x, y, teacher_logits)."""
    ds: TensorDataset = train_loader.dataset  # type: ignore[assignment]
    x_train, y_train = ds.tensors[0], ds.tensors[1]
    aug_ds = TensorDataset(x_train, y_train, teacher_logits.cpu())
    return DataLoader(aug_ds, batch_size=train_loader.batch_size or 32, shuffle=True)


def run_training_session(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    run_dir: Path,
    subject_ids: List[str],
    epochs: int = 10,
    lr: float = 2e-3,
    resume: str = "none",
    use_gpu: bool = True,
    qat: bool = False,
    distill_teacher_logits: torch.Tensor | None = None,
    distill_temp: float = 2.5,
    distill_alpha: float = 0.5,
) -> Dict[str, object]:
    """
    Execute training loop supporting:
    - Cross-channel Knowledge Distillation (306ch Teacher -> 8ch Student)
    - Multi-branch Structural Reparameterization (`RepTCNEEGNet.deploy_reparameterize()`)
    - Percentile-Clipped Per-Channel INT8 QAT fine-tuning & SQNR ablation
    - Automatic C99 static weight header export (`model_weights.h`)
    """
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    if distill_teacher_logits is not None:
        train_loader = _attach_teacher_logits_to_loader(train_loader, distill_teacher_logits)
        print(
            f"[distill] Enabled 306-ch Teacher -> {model.num_channels if hasattr(model, 'num_channels') else 8}-ch "
            f"Student Knowledge Distillation (T={distill_temp}, alpha={distill_alpha})"
        )

    total_params = sum(p.numel() for p in model.parameters())
    fp32_kb = total_params * 4.0 / 1024.0
    int8_kb = total_params * 1.0 / 1024.0

    (run_dir / "subjects.json").write_text(
        json.dumps({"subjects": subject_ids, "num_subjects": len(subject_ids)}, indent=2),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    all_train_y = torch.cat([b[1] for b in train_loader], dim=0)
    counts = torch.bincount(all_train_y, minlength=num_classes).float()
    valid_mask = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if valid_mask.any():
        inv_freq = 1.0 / torch.sqrt(counts[valid_mask])
        weights[valid_mask] = inv_freq / inv_freq.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    start_epoch = 1
    best_f1 = -1.0
    history: List[Dict[str, float]] = []

    last_ckpt = run_dir / "last.ckpt"
    best_ckpt = run_dir / "best.ckpt"

    if resume == "auto" and last_ckpt.exists() and not qat:
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        history = list(ckpt.get("history", []))
        print(f"[trainer] Resumed from {last_ckpt} at epoch {start_epoch} (best_f1={best_f1:.4f})")

    print(
        f"[trainer] Device: {device} | Arch: {type(model).__name__} | Params: {total_params:,} "
        f"(FP32: {fp32_kb:.2f} KB -> INT8: {int8_kb:.2f} KB Flash)"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    fp32_epochs = max(int(round(epochs * 0.6)), 1) if qat else epochs
    qat_epochs = max(epochs - fp32_epochs, 1) if qat else 0

    for epoch in range(start_epoch, fp32_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x_b = batch[0].to(device)
            y_b = batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_b)
            ce_loss = criterion(logits, y_b)

            if len(batch) >= 3:
                t_logits = batch[2].to(device)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / distill_temp, dim=-1),
                    F.softmax(t_logits / distill_temp, dim=-1),
                    reduction="batchmean",
                ) * (distill_temp ** 2)
                loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            bs = x_b.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_correct += int((torch.argmax(logits, dim=-1) == y_b).sum().item())
            train_total += bs

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)

        row = {"epoch": float(epoch), "phase": "fp32", "train_loss": train_loss, "train_acc": train_acc, **val_metrics}
        history.append(row)
        is_best = val_metrics["val_f1_macro"] >= best_f1
        if is_best:
            best_f1 = val_metrics["val_f1_macro"]

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_f1": best_f1,
            "history": history,
            "total_params": total_params,
        }
        torch.save(state, last_ckpt)
        if is_best:
            torch.save(state, best_ckpt)

        print(
            f"  [FP32] Epoch [{epoch:02d}/{fp32_epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% | "
            f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']*100:.1f}% "
            f"val_F1={val_metrics['val_f1_macro']*100:.1f}%"
            + (" [BEST]" if is_best else "")
        )

    reparam_report: Dict[str, float] = {}
    quant_report: Dict[str, float] = {}
    export_report: Dict[str, object] = {}

    # Verify structural reparameterization equivalence & Tensor Arena SRAM reduction
    if isinstance(model, RepTCNEEGNet) and not model.deploy:
        reparam_report = verify_reparameterization_and_sram(
            model,
            num_channels=model.num_channels,
            time_steps=model.time_steps,
        )
        print(
            f"[reparam] Multi-Branch -> Single-Path Fusion Verified! "
            f"Max Abs Diff: {reparam_report['max_abs_diff']:.2e} | "
            f"Params: {int(reparam_report['train_params']):,} -> {int(reparam_report['deploy_params']):,} | "
            f"Peak Tensor Arena SRAM (INT8): {reparam_report['multi_branch_arena_int8_kb']:.2f} KB -> "
            f"{reparam_report['deploy_arena_int8_kb']:.2f} KB (-{reparam_report['arena_sram_reduction_pct']:.1f}%)"
        )

    if qat:
        fp32_ref_model = copy.deepcopy(model).cpu().eval()
        if hasattr(fp32_ref_model, "deploy_reparameterize"):
            fp32_ref_model.deploy_reparameterize()

        # Reparameterize into single-branch graph before inserting INT8 QAT wrappers
        if hasattr(model, "deploy_reparameterize"):
            model.deploy_reparameterize()
        model = insert_qat_wrappers(model, percentile=99.5).to(device)
        qat_optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=1e-5)

        print(f"[qat] Starting Percentile-Clipped (99.5%) Per-Channel INT8 QAT for {qat_epochs} epoch(s)...")
        for q_ep in range(1, qat_epochs + 1):
            model.train()
            q_loss_sum = 0.0
            q_correct = 0
            q_total = 0
            for batch in train_loader:
                x_b = batch[0].to(device)
                y_b = batch[1].to(device)
                qat_optimizer.zero_grad(set_to_none=True)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                qat_optimizer.step()
                bs = x_b.size(0)
                q_loss_sum += float(loss.item()) * bs
                q_correct += int((torch.argmax(logits, dim=-1) == y_b).sum().item())
                q_total += bs

            val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
            if val_metrics["val_f1_macro"] >= best_f1:
                best_f1 = val_metrics["val_f1_macro"]
            print(
                f"  [QAT ] Epoch [{q_ep:02d}/{qat_epochs:02d}] "
                f"train_loss={q_loss_sum/max(q_total,1):.4f} train_acc={q_correct/max(q_total,1)*100:.1f}% | "
                f"val_acc={val_metrics['val_acc']*100:.1f}% val_F1={val_metrics['val_f1_macro']*100:.1f}%"
            )

        quant_report = evaluate_quantization_ablation(
            fp32_model=fp32_ref_model,
            qat_model=model,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            compute_f1_fn=compute_f1_macro,
        )
        print(
            f"[qat-ablation] FP32 F1={quant_report['fp32_f1_macro']*100:.1f}% | "
            f"Naive MinMax PTQ INT8 F1={quant_report['ptq_naive_int8_f1_macro']*100:.1f}% "
            f"(SQNR={quant_report['ptq_sqnr_db']:.1f} dB, CosSim={quant_report['ptq_cosine_sim']:.4f}) | "
            f"Proposed Percentile QAT INT8 F1={quant_report['qat_int8_f1_macro']*100:.1f}% "
            f"(SQNR={quant_report['qat_sqnr_db']:.1f} dB, CosSim={quant_report['qat_cosine_sim']:.4f})"
        )
        export_report = export_model_to_c_header(model, run_dir)

    elapsed = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    summary = {
        "device": str(device),
        "arch": type(model).__name__,
        "total_params": total_params,
        "fp32_weight_kb": round(fp32_kb, 2),
        "int8_weight_kb": round(int8_kb, 2),
        "best_val_f1_macro": round(best_f1, 4),
        "elapsed_sec": round(elapsed, 2),
        "peak_cuda_mb": round(peak_vram_mb, 2),
        "reparameterization": reparam_report,
        "quantization_ablation": quant_report,
        "c_export": export_report,
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[trainer] Completed in {elapsed:.2f}s | Best Val F1-macro: {best_f1*100:.1f}% | "
        f"Peak VRAM: {peak_vram_mb:.1f} MB | Saved -> {run_dir}"
    )
    return summary
