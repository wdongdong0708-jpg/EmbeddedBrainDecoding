"""Environment variable contracts, path resolution, and sensor channel mappings for edge_eeg."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

# ============================================================================
# 1. Environment Variable Contract (Local Dev <-> Colab Cloud)
# ============================================================================
ENV_STUDIES = "EDGEEEG_STUDIES"
ENV_CACHE = "EDGEEEG_CACHE"
ENV_RESULTS = "EDGEEEG_RESULTS"


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved directories governed by EDGEEEG_* environment variables."""

    studies_dir: Path
    cache_dir: Path
    results_dir: Path

    @classmethod
    def from_env(cls, ensure_dirs: bool = True) -> "WorkspacePaths":
        studies = Path(os.environ.get(ENV_STUDIES, "./workspace/data")).resolve()
        cache = Path(os.environ.get(ENV_CACHE, "./workspace/cache")).resolve()
        results = Path(os.environ.get(ENV_RESULTS, "./workspace/runs")).resolve()
        if ensure_dirs:
            studies.mkdir(parents=True, exist_ok=True)
            cache.mkdir(parents=True, exist_ok=True)
            results.mkdir(parents=True, exist_ok=True)
        return cls(studies_dir=studies, cache_dir=cache, results_dir=results)

    def get_run_dir(
        self,
        stage: str,
        modality: str,
        arch: str,
        channels: int,
        subset_tag: str,
    ) -> Path:
        """Route experiment outputs to isolated subdirectories so subsets never overwrite each other."""
        folder_name = f"{stage}_{modality}_{arch}_{channels}ch_{subset_tag}"
        run_dir = self.results_dir / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


# ============================================================================
# 2. Neuro-Anatomical 8-Channel Bilateral Temporal Configurations
# ============================================================================
# International 10-20 system 8 golden channels covering bilateral auditory cortex
# (Wernicke's / Superior Temporal Gyrus) & fronto-central speech areas
EEG_8CH_NAMES: List[str] = ["T7", "T8", "TP7", "TP8", "C3", "C4", "F7", "F8"]

# Representative bilateral temporal magnetometer/gradiometer indices in 306-channel Elekta Neuromag MEG
# (4 left-temporal + 4 right-temporal auditory cortex channels)
MEG_TEMPORAL_8CH_INDICES: List[int] = [15, 18, 21, 24, 135, 138, 141, 144]


def select_channel_indices(total_channels: int, target_channels: int) -> List[int]:
    """Return deterministic sensor indices when pruning from total_channels to target_channels."""
    if target_channels >= total_channels:
        return list(range(total_channels))
    if target_channels == 8 and total_channels == 306:
        return list(MEG_TEMPORAL_8CH_INDICES)
    # Evenly spaced deterministic subset for other channel counts (e.g. 64 -> 8 or 16)
    step = total_channels / float(target_channels)
    return [int(round(i * step)) % total_channels for i in range(target_channels)]


def normalize_subject_list(
    subject: str | None = None,
    subjects: str | None = None,
    num_subjects: int | None = None,
) -> tuple[List[str], str]:
    """Parse CLI subject arguments into a sorted list of subject IDs and a canonical subset_tag."""
    if subjects:
        items = [s.strip().zfill(3) for s in subjects.split(",") if s.strip()]
        items = sorted(dict.fromkeys(items))
        tag = f"subs-{'_'.join(items[:4])}" + (f"_plus{len(items)-4}" if len(items) > 4 else "")
        return items, tag

    if num_subjects is not None and num_subjects > 0:
        items = [f"{i:03d}" for i in range(1, num_subjects + 1)]
        return items, f"num{num_subjects:02d}"

    subj = (subject or "001").strip()
    if subj.lower() == "all":
        # LibriBrain100 has 33 subjects; default single-subject benchmark is 001 (Sherlock1)
        items = [f"{i:03d}" for i in range(1, 34)]
        return items, "all"

    subj_id = subj.zfill(3) if subj.isdigit() else subj
    return [subj_id], f"sub{subj_id}"
