"""Unified CLI entrypoint: `python -m edge_eeg {smoke,cache,train,eval,export}`."""

from __future__ import annotations

import argparse
import sys
import torch
from torch.utils.data import DataLoader

from .config import WorkspacePaths, normalize_subject_list
from .data import build_dataloaders, prepare_subject_cache
from .engine import export_model_to_c_header, run_training_session
from .lexicon import (
    CODEBOOK_12,
    POLARIZED_SUBSET_4_INDICES,
    compute_hamming_report,
    simulate_stage3_cascade_boost,
)
from .models import build_model


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--modality", choices=["meg", "eeg"], default="meg", help="Signal modality")
    p.add_argument(
        "--stage",
        choices=["vad", "phoneme", "cascade"],
        default="phoneme",
        help="Stage 1 (vad), Stage 2 (phoneme/tone), or Stage 3 (cascade)",
    )
    p.add_argument(
        "--arch",
        choices=["rep_tcn", "student_tiny", "teacher_conformer"],
        default="rep_tcn",
        help="Model architecture (rep_tcn=RepTCNEEGNet, student_tiny=TinyEdgeEEGNet/MicroVADNet, teacher_conformer=306ch Teacher)",
    )
    p.add_argument("--subject", type=str, default="001", help="Single subject ID (e.g. 001) or 'all'")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs (e.g. 001,002)")
    p.add_argument("--num-subjects", type=int, default=None, help="First N subjects")
    p.add_argument("--num-runs", type=int, default=1, help="Number of LibriBrain runs to load (1..88)")
    p.add_argument(
        "--label-mode",
        choices=["place5", "phoneme39"],
        default="place5",
        help="Stage 2 label granularity: 'place5' (5 articulatory places C1..C5) or 'phoneme39' (39 ARPAbet phonemes)",
    )
    p.add_argument("--channels", type=int, default=8, help="Number of input sensor channels (8 or 306)")
    p.add_argument("--time-steps", type=int, default=125, help="Window length in samples (125 = 500ms @ 250Hz)")
    p.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    p.add_argument("--devices", type=int, default=1, help="Number of accelerator devices")
    p.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU thread limit")
    p.add_argument("--require-cache", action="store_true", help="Fail immediately if NVMe cache is missing")
    p.add_argument("--resume", choices=["none", "auto"], default="none", help="Auto-resume from last.ckpt")
    p.add_argument("--qat", action="store_true", help="Enable Reparameterization + Percentile Per-Channel INT8 QAT")
    p.add_argument("--distill", action="store_true", help="Enable 306-ch Teacher -> 8-ch Student Knowledge Distillation")
    p.add_argument("--verify-codebook", action="store_true", help="Print 12-sentence Hamming distance report")


def _print_codebook_and_viterbi_report() -> None:
    print("=" * 78)
    print("  [Thesis Sec 2.1] 12-Sentence Mandarin Command Codebook & Hamming Analysis")
    print("=" * 78)
    for e in CODEBOOK_12:
        t_str = "-".join(str(x) for x in e.tones_1based)
        c_str = "-".join(f"C{x}" for x in e.consonants_1based)
        note_str = f"  ({e.note})" if e.note else ""
        print(f"  {e.code_id} [{e.domain}] {e.text} ({e.pinyin:22s}) | T:{t_str} | C:{c_str}{note_str}")

    rep12 = compute_hamming_report(CODEBOOK_12)
    rep4 = compute_hamming_report([CODEBOOK_12[i] for i in POLARIZED_SUBSET_4_INDICES])
    print("-" * 78)
    print(
        f"  12-Command Full Set ({rep12['num_pairs']} pairs): "
        f"Tone d_avg={rep12['tone_d_avg']:.2f}, Cons d_avg={rep12['cons_d_avg']:.2f} | "
        f"Joint 8D d_min={rep12['joint_d_min']}, d_avg={rep12['joint_d_avg']:.2f} / 8.00 "
        f"(Hard correction radius t={rep12['hard_error_correction_radius']})"
    )
    print(
        f"  4-Command Polarized EEG Set ({rep4['num_pairs']} pairs: S01,S04,S08,S12): "
        f"Joint 8D d_min={rep4['joint_d_min']}, d_avg={rep4['joint_d_avg']:.2f} / 8.00 "
        f"(Hard correction radius t={rep4['hard_error_correction_radius']})"
    )

    sim12 = simulate_stage3_cascade_boost(single_tone_acc=0.55, single_cons_acc=0.48, use_polarized_4=False)
    sim4 = simulate_stage3_cascade_boost(single_tone_acc=0.55, single_cons_acc=0.48, use_polarized_4=True)
    print("-" * 78)
    print("  [Thesis Sec 5.1] Stage 3 Soft-Decision Viterbi Lexicon Decoding Boost:")
    print(
        f"  Single-Syllable Tone Acc={sim12['single_tone_acc']*100:.1f}%, "
        f"Consonant Place Acc={sim12['single_cons_acc']*100:.1f}%"
    )
    print(
        f"  -> [12-Sentence Set] Raw Unconstrained Exact-Match : {sim12['raw_joint_8feat_acc']*100:5.1f}%\n"
        f"  -> [12-Sentence Set] Stage 3 Viterbi (Tone-Only)   : {sim12['viterbi_tone_only_acc']*100:5.1f}%\n"
        f"  -> [12-Sentence Set] Stage 3 Viterbi (8D FP32)     : {sim12['viterbi_joint_fp32_acc']*100:5.1f}%\n"
        f"  -> [12-Sentence Set] Stage 3 Viterbi (8D MCU INT8) : {sim12['viterbi_joint_int8_acc']*100:5.1f}%\n"
        f"  -> [4-Sentence EEG ] Stage 3 Viterbi (8D MCU INT8) : {sim4['viterbi_joint_int8_acc']*100:5.1f}%"
    )
    print("=" * 78)


def _compute_teacher_distillation_logits(
    paths: WorkspacePaths,
    args: argparse.Namespace,
    subject_ids: list[str],
    subset_tag: str,
    smoke_mode: bool,
) -> torch.Tensor:
    """Load or train the 306-channel TeacherConformerTCN and extract soft logits over the training split."""
    full_ch = 306 if args.modality == "meg" else 64
    t_train_loader, t_val_loader, num_classes = build_dataloaders(
        paths=paths,
        modality=args.modality,
        stage=args.stage,
        subject_ids=subject_ids,
        target_channels=full_ch,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        label_mode=args.label_mode,
        require_cache=args.require_cache,
        smoke_mode=smoke_mode,
    )
    teacher = build_model(
        stage=args.stage,
        arch="teacher_conformer",
        num_channels=full_ch,
        time_steps=args.time_steps,
        num_classes=num_classes,
    )
    teacher_dir = paths.get_run_dir(
        stage=args.stage,
        modality=args.modality,
        arch="teacher_conformer",
        channels=full_ch,
        subset_tag="smoke" if smoke_mode else subset_tag,
    )
    best_ckpt = teacher_dir / "best.ckpt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        try:
            teacher.load_state_dict(ckpt["model_state"])
            print(f"[distill] Loaded pre-trained 306-ch Teacher checkpoint from {best_ckpt}")
        except RuntimeError:
            best_ckpt = teacher_dir / "nonexistent.ckpt"

    if not best_ckpt.exists():
        print(f"[distill] Training 306-ch TeacherConformerTCN first to generate soft targets...")
        run_training_session(
            model=teacher,
            train_loader=t_train_loader,
            val_loader=t_val_loader,
            num_classes=num_classes,
            run_dir=teacher_dir,
            subject_ids=subject_ids,
            epochs=max(args.epochs // 2, 2),
            lr=args.lr * 0.5,
            resume="none",
            use_gpu=(args.devices > 0),
            qat=False,
        )

    device = torch.device("cuda" if (args.devices > 0 and torch.cuda.is_available()) else "cpu")
    teacher = teacher.to(device).eval()
    seq_loader = DataLoader(t_train_loader.dataset, batch_size=args.batch_size, shuffle=False)
    logits_list = []
    with torch.no_grad():
        for x_b, _ in seq_loader:
            logits_list.append(teacher(x_b.to(device)).cpu())
    return torch.cat(logits_list, dim=0)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="edge_eeg", description="Edge EEG/MEG Hierarchical Speech Decoding CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "cache", "train", "eval", "export"):
        sp = subparsers.add_parser(name)
        _add_common_args(sp)

    args = parser.parse_args()
    torch.set_num_threads(max(args.cpu_threads, 1))
    paths = WorkspacePaths.from_env(ensure_dirs=True)
    subject_ids, subset_tag = normalize_subject_list(args.subject, args.subjects, args.num_subjects)

    if args.cmd == "eval" or args.verify_codebook or args.stage == "cascade":
        _print_codebook_and_viterbi_report()
        if args.cmd == "eval":
            return

    if args.cmd == "cache":
        prepare_subject_cache(
            paths=paths,
            modality=args.modality,
            stage=args.stage,
            subject_ids=subject_ids,
            num_runs=args.num_runs,
            label_mode=args.label_mode,
            time_steps=args.time_steps,
        )
        return

    smoke_mode = args.cmd == "smoke"
    train_loader, val_loader, num_classes = build_dataloaders(
        paths=paths,
        modality=args.modality,
        stage=args.stage,
        subject_ids=subject_ids,
        target_channels=args.channels,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        label_mode=args.label_mode,
        require_cache=args.require_cache,
        smoke_mode=smoke_mode,
    )

    model = build_model(
        stage=args.stage,
        arch=args.arch,
        num_channels=args.channels,
        time_steps=args.time_steps,
        num_classes=num_classes,
    )

    run_dir = paths.get_run_dir(
        stage=args.stage,
        modality=args.modality,
        arch=args.arch,
        channels=args.channels,
        subset_tag="smoke" if smoke_mode else subset_tag,
    )

    if args.cmd == "export":
        best_ckpt = run_dir / "best.ckpt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"], strict=False)
        export_model_to_c_header(model, run_dir, model_prefix=f"edge_eeg_{args.stage}")
        return

    teacher_logits = None
    if args.distill and args.arch != "teacher_conformer":
        teacher_logits = _compute_teacher_distillation_logits(
            paths=paths,
            args=args,
            subject_ids=subject_ids,
            subset_tag=subset_tag,
            smoke_mode=smoke_mode,
        )

    run_training_session(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        run_dir=run_dir,
        subject_ids=subject_ids,
        epochs=args.epochs,
        lr=args.lr,
        resume=args.resume,
        use_gpu=(args.devices > 0),
        qat=args.qat,
        distill_teacher_logits=teacher_logits,
    )

    if smoke_mode:
        _print_codebook_and_viterbi_report()


if __name__ == "__main__":
    main()
