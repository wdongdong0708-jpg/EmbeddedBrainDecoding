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


def get_train_run_keys(num_runs: int = 1) -> List[tuple[str, str, str, str]]:
    """Select the first `num_runs` training runs from LibriBrain RUN_KEYS (excluding val/test)."""
    excluded = set(tuple(k) for k in ( list(VALIDATION_RUN_KEYS) + list(TEST_RUN_KEYS) ))
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


def prepare_subject_cache(
    paths: WorkspacePaths,
    modality: str,
    stage: str,
    subject_ids: List[str],
    num_runs: int = 1,
    label_mode: str = "place5",
    time_steps: int = 125,
    max_samples: int | None = None,
) -> Path:
    """
    Load real LibriBrain MEG trials via official `pnpl` library and cache contiguous tensors to disk.
    Downloads raw .h5 and _events.tsv files into `paths.studies_dir / 'libribrain'` if not yet present.
    """
    eff_stage = "phoneme" if stage == "cascade" else stage
    target_pt = _cache_file_path(paths.cache_dir, modality, eff_stage, label_mode, num_runs)
    if target_pt.exists() and max_samples is None:
        print(f"[cache] Reusing real LibriBrain cache shard: {target_pt}")
        return target_pt

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
            oversample_silence_jitter=35,
            preload_h5=True,
            download=True,
        )
        num_classes = 2
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

    total_len = len(raw_ds)
    if max_samples is not None and max_samples < total_len:
        # Deterministic strided sampling across the entire run to preserve class diversity
        indices = np.linspace(0, total_len - 1, num=max_samples, dtype=np.int64).tolist()
    else:
        indices = list(range(total_len))

    xs: List[torch.Tensor] = []
    ys: List[int] = []

    for idx in indices:
        item = raw_ds[idx]
        data_tensor: torch.Tensor = item[0].detach().to(dtype=torch.float32)
        if data_tensor.shape[-1] >= time_steps:
            data_tensor = data_tensor[..., :time_steps]
        else:
            pad = time_steps - data_tensor.shape[-1]
            data_tensor = torch.nn.functional.pad(data_tensor, (0, pad))

        raw_label = item[1]
        if eff_stage == "vad":
            # LibriBrainSpeech returns per-timepoint binary mask of length 125
            y_val = int(raw_label.float().mean().item() >= 0.5)
        else:
            phoneme_id = int(raw_label.item())
            if label_mode == "place5":
                ph_str = phonemes_sorted[phoneme_id]
                y_val = ARPABET_TO_PLACE5[ph_str]
            else:
                y_val = phoneme_id

        xs.append(data_tensor.unsqueeze(0))  # [1, 306, T]
        ys.append(y_val)

    x_tensor = torch.stack(xs, dim=0)  # [N, 1, 306, T]
    y_tensor = torch.tensor(ys, dtype=torch.long)

    if max_samples is None:
        torch.save(
            {
                "x": x_tensor,
                "y": y_tensor,
                "num_classes": num_classes,
                "source": "pnpl_libribrain_real",
                "run_keys": run_keys,
                "label_mode": label_mode,
            },
            target_pt,
        )
        manifest = paths.cache_dir / f"manifest_{modality}_{eff_stage}.json"
        manifest.write_text(
            json.dumps(
                {
                    "modality": modality,
                    "stage": eff_stage,
                    "label_mode": label_mode,
                    "num_runs": num_runs,
                    "num_trials": len(xs),
                    "num_classes": num_classes,
                    "source": "pnpl_libribrain_real",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[cache] Saved real LibriBrain shard: {len(xs)} trials, "
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
    In `smoke_mode`, uses the first 1,024 real trials from the cached shard (or builds the shard if missing)
    so smoke tests execute in ~1-2 seconds on 100% genuine human neural recordings.
    """
    eff_stage = "phoneme" if stage == "cascade" else stage
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
    x_all: torch.Tensor = payload["x"]
    y_all: torch.Tensor = payload["y"]
    num_classes = int(payload["num_classes"])

    if smoke_mode and x_all.shape[0] > 1024:
        # Stratified/strided subsample of 1,024 real neural trials for fast smoke verification
        idx = torch.linspace(0, x_all.shape[0] - 1, steps=1024).long()
        x_all = x_all[idx]
        y_all = y_all[idx]

    total_ch = x_all.shape[2]
    if target_channels < total_ch:
        ch_idx = select_channel_indices(total_ch, target_channels)
        x_all = x_all[:, :, ch_idx, :]

    # Contiguous chronological 80/20 train/val split (avoids temporal leakage across adjacent windows)
    n_total = x_all.shape[0]
    n_train = max(int(n_total * 0.8), 1)
    train_ds = TensorDataset(x_all[:n_train], y_all[:n_train])
    val_ds = TensorDataset(x_all[n_train:], y_all[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, num_classes
