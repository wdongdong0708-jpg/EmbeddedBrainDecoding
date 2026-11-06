"""Real PNPL LibriBrain dataset loader with spatial channel masking and NVMe caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
from pnpl.datasets.libribrain2025.constants import (
    RUN_KEYS,
    TEST_RUN_KEYS,
    VALIDATION_RUN_KEYS,
)

from ..config import WorkspacePaths, select_channel_indices

CACHE_SCHEMA_VERSION = "v2"
VAD_CACHE_SCHEMA_VERSION = "v3"


def _expected_cache_schema(eff_stage: str) -> str:
    return VAD_CACHE_SCHEMA_VERSION if eff_stage == "vad" else CACHE_SCHEMA_VERSION

# Mapping from 39 ARPAbet phonemes in LibriBrainPhoneme to 5 Neuro-Acoustic Articulatory Clusters (C1..C5 -> 0..4)
ARPABET_TO_PLACE5: Dict[str, int] = {
    # C1 (0): Labial / Labiodental / Labial-velar & rounded back
    "b": 0, "p": 0, "m": 0, "f": 0, "v": 0, "w": 0, "uw": 0, "ow": 0, "aw": 0, "ao": 0,
    # C2 (1): Alveolar / Dental coronal stops, nasals, laterals & central vowels
    "d": 1, "t": 1, "n": 1, "l": 1, "th": 1, "dh": 1, "ah": 1, "uh": 1,
    # C3 (2): Sibilant / Palato-alveolar / Retroflex fricatives, affricates & rhotics
    "s": 2, "z": 2, "sh": 2, "zh": 2, "ch": 2, "jh": 2, "r": 2, "er": 2,
    # C4 (3): Palatal & High/Mid Front vowels/diphthongs (steep F2 formant transition)
    "y": 3, "iy": 3, "ih": 3, "ey": 3, "eh": 3, "ae": 3, "ay": 3, "oy": 3,
    # C5 (4): Velar / Glottal & Open dorsal vowels
    "g": 4, "k": 4, "ng": 4, "hh": 4, "aa": 4,
}

# High-SNR canonical phoneme prototypes for each of the 5 articulatory places (C1..C5)
PROTO_PLACE5: Dict[str, int] = {
    "v": 0, "w": 0, "ow": 0, "m": 0,
    "t": 1, "l": 1, "ah": 1,
    "z": 2, "s": 2,
    "ih": 3, "eh": 3, "ey": 3,
    "k": 4, "aa": 4,
}


def get_train_run_keys(num_runs: int = 1) -> List[tuple[str, str, str, str]]:
    """Select the first `num_runs` training runs from LibriBrain RUN_KEYS (excluding val/test)."""
    excluded = set(tuple(k) for k in (list(VALIDATION_RUN_KEYS) + list(TEST_RUN_KEYS)))
    train_keys = [tuple(k) for k in RUN_KEYS if tuple(k) not in excluded]
    if num_runs <= 0 or num_runs >= len(train_keys):
        return train_keys
    return train_keys[:num_runs]


def _cache_file_path(
    cache_dir: Path,
    modality: str,
    stage: str,
    label_mode: str,
    num_runs: int,
) -> Path:
    eff_stage = "phoneme" if stage == "cascade" else stage
    suffix = label_mode if eff_stage == "phoneme" else "binary"
    return cache_dir / f"{modality}_{eff_stage}_{suffix}_runs{num_runs:02d}.pt"


def _extract_clipped_window(raw_ds, idx: int, time_steps: int) -> torch.Tensor:
    item = raw_ds[idx]
    data_tensor: torch.Tensor = item[0].detach().to(dtype=torch.float32)
    if data_tensor.shape[-1] >= time_steps:
        data_tensor = data_tensor[..., :time_steps]
    else:
        pad = time_steps - data_tensor.shape[-1]
        data_tensor = torch.nn.functional.pad(data_tensor, (0, pad))
    return data_tensor.unsqueeze(0)  # [1, 306, T]


def _build_balanced_grouped_trials(
    x_s: torch.Tensor,
    y_s: torch.Tensor,
    num_classes: int,
    group_size: int,
    n_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Follow official PNPL GroupedDataset protocol strictly WITHIN a single chronological split
    (Train or Val) so random neighboring phonemes in the 500ms window cancel out with zero cross-split leakage.
    """
    g = torch.Generator().manual_seed(seed)
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for c in range(num_classes):
        idx = (y_s == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        for _ in range(n_per_class):
            perm = idx[torch.randint(0, len(idx), (group_size,), generator=g)]
            xs.append(x_s[perm].mean(dim=0))
            ys.append(c)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def prepare_subject_cache(
    paths: WorkspacePaths,
    modality: str,
    stage: str,
    subject_ids: List[str],
    num_runs: int = 1,
    label_mode: str = "place5",
    time_steps: int = 125,
    max_samples: int | None = None,
    force: bool = False,
) -> Path:
    """
    Load real LibriBrain MEG trials via official `pnpl` library and cache contiguous tensors to disk.
    Downloads raw .h5 and _events.tsv files into `paths.studies_dir / 'libribrain'` if not yet present.
    """
    eff_stage = "phoneme" if stage == "cascade" else stage
    expected_schema = _expected_cache_schema(eff_stage)
    target_pt = _cache_file_path(paths.cache_dir, modality, eff_stage, label_mode, num_runs)
    manifest = paths.cache_dir / f"manifest_{modality}_{eff_stage}.json"

    if target_pt.exists() and max_samples is None and not force:
        try:
            existing_meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
            if existing_meta.get("cache_schema") == expected_schema and existing_meta.get("num_runs") == num_runs:
                print(f"[cache] Reusing verified {expected_schema} LibriBrain cache shard: {target_pt}")
                return target_pt
        except Exception:
            pass

    ds_root = paths.studies_dir / "libribrain"
    ds_root.mkdir(parents=True, exist_ok=True)
    run_keys = get_train_run_keys(num_runs=num_runs)
    tmax = round(time_steps / 250.0, 4)

    print(
        f"[pnpl] Loading real LibriBrain ({eff_stage}, label_mode={label_mode}) "
        f"for {len(run_keys)} run(s): {run_keys[:3]} from {ds_root} ..."
    )

    if eff_stage == "vad":
        raw_ds = LibriBrainSpeech(
            data_path=str(ds_root),
            include_run_keys=run_keys,
            tmin=0.0,
            tmax=tmax,
            oversample_silence_jitter=5,
            preload_h5=True,
            download=True,
        )
        num_classes = 2
        tr_per_run = 1000 if len(run_keys) == 1 else max(400, 3200 // len(run_keys))
        va_per_run = 250 if len(run_keys) == 1 else max(100, 800 // len(run_keys))
        if max_samples is not None:
            tr_per_run = max(max_samples // (4 * max(len(run_keys), 1)), 16)
            va_per_run = max(tr_per_run // 4, 8)

        x_tr_list: List[torch.Tensor] = []
        y_tr_list: List[torch.Tensor] = []
        x_va_list: List[torch.Tensor] = []
        y_va_list: List[torch.Tensor] = []

        for r_idx, rk in enumerate(run_keys):
            rk_indices = [
                i for i, s in enumerate(raw_ds.samples) if (s[0], s[1], s[2], s[3]) == rk
            ]
            if not rk_indices:
                continue
            sp_onsets = [
                float(raw_ds.samples[i][4])
                for i in rk_indices
                if len(raw_ds.samples[i][5]) == time_steps and raw_ds.samples[i][5].sum() > 0
            ]
            first_sp_onset = min(sp_onsets) if sp_onsets else 28.5

            # Exclude 0..4.0s IIR startup transient and speech-onset boundary transition windows per run
            sil_rk = [
                i
                for i in rk_indices
                if len(raw_ds.samples[i][5]) == time_steps
                and raw_ds.samples[i][5].sum() == 0
                and 4.0 <= float(raw_ds.samples[i][4]) < first_sp_onset - 1.0
            ]
            if len(sil_rk) < 32:
                sil_rk = [
                    i
                    for i in rk_indices
                    if len(raw_ds.samples[i][5]) == time_steps
                    and raw_ds.samples[i][5].sum() == 0
                    and float(raw_ds.samples[i][4]) >= 4.0
                ]
            sp_rk = [
                i
                for i in rk_indices
                if len(raw_ds.samples[i][5]) == time_steps
                and raw_ds.samples[i][5].sum() == time_steps
                and float(raw_ds.samples[i][4]) >= max(first_sp_onset + 1.0, 4.0)
            ]
            if len(sil_rk) < 10 or len(sp_rk) < 10:
                continue

            # Chronological 75% Train / 5% Embargo Gap (zero window overlap) / 20% Val strictly WITHIN each run
            n_sil_tr, n_sil_va0 = int(len(sil_rk) * 0.75), int(len(sil_rk) * 0.80)
            n_sp_tr, n_sp_va0 = int(len(sp_rk) * 0.75), int(len(sp_rk) * 0.80)
            tr_indices = sil_rk[:n_sil_tr] + sp_rk[:n_sp_tr]
            va_indices = sil_rk[n_sil_va0:] + sp_rk[n_sp_va0:]

            x_tr_rk = torch.stack([_extract_clipped_window(raw_ds, i, time_steps) for i in tr_indices], dim=0)
            y_tr_rk = torch.tensor([0] * n_sil_tr + [1] * n_sp_tr, dtype=torch.long)
            x_va_rk = torch.stack([_extract_clipped_window(raw_ds, i, time_steps) for i in va_indices], dim=0)
            y_va_rk = torch.tensor(
                [0] * (len(sil_rk) - n_sil_va0) + [1] * (len(sp_rk) - n_sp_va0),
                dtype=torch.long,
            )

            # Remove per-window DC offset and apply balanced per-run Train-split std normalization
            x_tr_rk = x_tr_rk - x_tr_rk.mean(dim=-1, keepdim=True)
            x_va_rk = x_va_rk - x_va_rk.mean(dim=-1, keepdim=True)
            sd_0 = x_tr_rk[y_tr_rk == 0].std(dim=(0, 3), keepdim=True)
            sd_1 = x_tr_rk[y_tr_rk == 1].std(dim=(0, 3), keepdim=True)
            sd_rk = (0.5 * (sd_0 + sd_1)).clamp_min(1e-6)
            x_tr_rk = x_tr_rk / sd_rk
            x_va_rk = x_va_rk / sd_rk

            # 3-frame within-split grouping (matching VADSlidingGate 3-frame integration in Sec 3.3) with 1:1 balance
            x_tr_g, y_tr_g = _build_balanced_grouped_trials(
                x_tr_rk, y_tr_rk, num_classes=2, group_size=3, n_per_class=tr_per_run, seed=42 + r_idx
            )
            x_va_g, y_va_g = _build_balanced_grouped_trials(
                x_va_rk, y_va_rk, num_classes=2, group_size=3, n_per_class=va_per_run, seed=99 + r_idx
            )
            x_tr_list.append(x_tr_g)
            y_tr_list.append(y_tr_g)
            x_va_list.append(x_va_g)
            y_va_list.append(y_va_g)

        x_tr = torch.cat(x_tr_list, dim=0)
        y_tr = torch.cat(y_tr_list, dim=0)
        x_va = torch.cat(x_va_list, dim=0)
        y_va = torch.cat(y_va_list, dim=0)
        split_index = int(x_tr.shape[0])
        x_tensor = torch.cat([x_tr, x_va], dim=0)
        y_tensor = torch.cat([y_tr, y_va], dim=0)

    else:
        raw_ds = LibriBrainPhoneme(
            data_path=str(ds_root),
            include_run_keys=run_keys,
            tmin=0.0,
            tmax=tmax,
            preload_h5=True,
            download=True,
        )
        phonemes_sorted = list(raw_ds.labels_sorted)
        num_classes = 5 if label_mode == "place5" else len(phonemes_sorted)

        if label_mode == "place5":
            proto_indices: List[int] = []
            proto_labels: List[int] = []
            for idx, sample in enumerate(raw_ds.samples):
                ph_base = str(sample[5]).split("_")[0]
                if ph_base in PROTO_PLACE5:
                    proto_indices.append(idx)
                    proto_labels.append(PROTO_PLACE5[ph_base])

            if max_samples is not None and max_samples < len(proto_indices):
                sel = np.linspace(0, len(proto_indices) - 1, num=max_samples, dtype=np.int64).tolist()
                proto_indices = [proto_indices[i] for i in sel]
                proto_labels = [proto_labels[i] for i in sel]

            x_raw = torch.stack([_extract_clipped_window(raw_ds, i, time_steps) for i in proto_indices], dim=0)
            y_raw = torch.tensor(proto_labels, dtype=torch.long)

            # Strict chronological split: 75% Train pool, 5% Embargo Gap, 20% Val pool
            n_tr_end = int(len(x_raw) * 0.75)
            n_va_start = int(len(x_raw) * 0.80)
            x_tr_raw, y_tr_raw = x_raw[:n_tr_end], y_raw[:n_tr_end]
            x_va_raw, y_va_raw = x_raw[n_va_start:], y_raw[n_va_start:]

            # Subtract common-mode auditory onset ERP estimated strictly on Train pool
            grand_tr = x_tr_raw.mean(dim=0, keepdim=True)
            x_tr_raw = x_tr_raw - grand_tr
            x_va_raw = x_va_raw - grand_tr

            scale_factor = max(min(len(run_keys), 4), 1)
            tr_per_class = 400 * scale_factor
            va_per_class = 100 * scale_factor
            x_tr, y_tr = _build_balanced_grouped_trials(
                x_tr_raw, y_tr_raw, num_classes=5, group_size=16, n_per_class=tr_per_class, seed=42
            )
            x_va, y_va = _build_balanced_grouped_trials(
                x_va_raw, y_va_raw, num_classes=5, group_size=16, n_per_class=va_per_class, seed=99
            )
            split_index = int(x_tr.shape[0])
            x_tensor = torch.cat([x_tr, x_va], dim=0)
            y_tensor = torch.cat([y_tr, y_va], dim=0)
        else:
            total_len = len(raw_ds)
            if max_samples is not None and max_samples < total_len:
                indices = np.linspace(0, total_len - 1, num=max_samples, dtype=np.int64).tolist()
            else:
                indices = list(range(total_len))
            xs = [_extract_clipped_window(raw_ds, i, time_steps) for i in indices]
            ys = [int(raw_ds[i][1].item()) for i in indices]
            x_tensor = torch.stack(xs, dim=0)
            y_tensor = torch.tensor(ys, dtype=torch.long)
            split_index = max(int(len(x_tensor) * 0.8), 1)

    if max_samples is None:
        torch.save(
            {
                "x": x_tensor,
                "y": y_tensor,
                "num_classes": num_classes,
                "split_index": split_index,
                "cache_schema": expected_schema,
                "source": "pnpl_libribrain_real",
                "run_keys": run_keys,
                "label_mode": label_mode,
            },
            target_pt,
        )
        manifest.write_text(
            json.dumps(
                {
                    "modality": modality,
                    "stage": eff_stage,
                    "label_mode": label_mode,
                    "num_runs": num_runs,
                    "num_trials": int(x_tensor.shape[0]),
                    "split_index": split_index,
                    "num_classes": num_classes,
                    "cache_schema": expected_schema,
                    "source": "pnpl_libribrain_real",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[cache] Saved real LibriBrain shard ({expected_schema}): {x_tensor.shape[0]} trials "
            f"(train={split_index}, val={x_tensor.shape[0]-split_index}), "
            f"shape={tuple(x_tensor.shape)}, classes={num_classes} -> {target_pt}"
        )
    return target_pt


def build_dataloaders(
    paths: WorkspacePaths,
    modality: str,
    stage: str,
    subject_ids: List[str],
    target_channels: int = 8,
    time_steps: int = 125,
    batch_size: int = 32,
    num_runs: int = 1,
    label_mode: str = "place5",
    require_cache: bool = False,
    smoke_mode: bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build train and validation DataLoaders strictly from REAL LibriBrain MEG signals.
    In `smoke_mode`, uses a stratified subsample of real trials from the cached shard
    so smoke tests execute in ~1-2 seconds on 100% genuine human neural recordings.
    """
    eff_stage = "phoneme" if stage == "cascade" else stage
    expected_schema = _expected_cache_schema(eff_stage)
    shard_path = _cache_file_path(paths.cache_dir, modality, eff_stage, label_mode, num_runs)

    if not shard_path.exists():
        if require_cache:
            raise FileNotFoundError(
                f"Missing precomputed real cache shard: {shard_path}. "
                "Run `python -m edge_eeg cache ...` or unpack the archive first!"
            )
        prepare_subject_cache(
            paths=paths,
            modality=modality,
            stage=eff_stage,
            subject_ids=subject_ids,
            num_runs=num_runs,
            label_mode=label_mode,
            time_steps=time_steps,
        )

    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if payload.get("cache_schema") != expected_schema and (paths.studies_dir / "libribrain").exists():
        print(f"[cache] Upgrading legacy cache shard {shard_path.name} to schema {expected_schema}...")
        prepare_subject_cache(
            paths=paths,
            modality=modality,
            stage=eff_stage,
            subject_ids=subject_ids,
            num_runs=num_runs,
            label_mode=label_mode,
            time_steps=time_steps,
            force=True,
        )
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)

    x_all: torch.Tensor = payload["x"]
    y_all: torch.Tensor = payload["y"]
    num_classes = int(payload["num_classes"])
    n_total = x_all.shape[0]
    n_train = int(payload.get("split_index", max(int(n_total * 0.8), 1)))

    total_ch = x_all.shape[2]
    if target_channels < total_ch:
        ch_idx = select_channel_indices(total_ch, target_channels)
        x_all = x_all[:, :, ch_idx, :]

    x_tr, y_tr = x_all[:n_train], y_all[:n_train]
    x_va, y_va = x_all[n_train:], y_all[n_train:]

    if smoke_mode:
        if len(x_tr) > 800:
            idx_tr = torch.linspace(0, len(x_tr) - 1, steps=800).long()
            x_tr, y_tr = x_tr[idx_tr], y_tr[idx_tr]
        if len(x_va) > 224:
            idx_va = torch.linspace(0, len(x_va) - 1, steps=224).long()
            x_va, y_va = x_va[idx_va], y_va[idx_va]

    train_ds = TensorDataset(x_tr, y_tr)
    val_ds = TensorDataset(x_va, y_va)

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, num_classes
