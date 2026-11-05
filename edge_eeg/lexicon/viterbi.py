"""Stage 3 Soft-Decision Log-Likelihood Viterbi / Lexicon Decoder (Thesis Sec 5.1)."""

from __future__ import annotations

from typing import Dict, Sequence
import numpy as np

from .codebook import (
    CODEBOOK_12,
    CODEBOOK_CONSONANTS,
    CODEBOOK_TONES,
    POLARIZED_SUBSET_4_INDICES,
)


class LexiconViterbiDecoder:
    """
    Closed-vocabulary soft-decision sequence decoder over 4-syllable Mandarin commands.
    Supports both floating-point log-probabilities and INT8 quantized log-likelihood scores
    (mirroring the pure C implementation on MCU).
    """

    def __init__(
        self,
        subset_indices: Sequence[int] | None = None,
        tone_weight: float = 1.0,
        consonant_weight: float = 1.0,
    ) -> None:
        if subset_indices is not None:
            self.active_indices = np.array(list(subset_indices), dtype=np.int32)
        else:
            self.active_indices = np.arange(len(CODEBOOK_12), dtype=np.int32)

        self.tones = CODEBOOK_TONES[self.active_indices]          # shape: (M, 4)
        self.consonants = CODEBOOK_CONSONANTS[self.active_indices]  # shape: (M, 4)
        self.entries = [CODEBOOK_12[i] for i in self.active_indices]
        self.tone_weight = float(tone_weight)
        self.consonant_weight = float(consonant_weight)

    def decode_soft_fp32(
        self,
        tone_log_probs: np.ndarray,
        cons_log_probs: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """
        Compute cumulative log-likelihood path scores for all M legal sentences.
        Args:
            tone_log_probs: shape (4, 4) - 4 syllable positions x 4 Mandarin tones
            cons_log_probs: shape (4, 5) - 4 syllable positions x 5 articulatory places (optional)
        Returns:
            (best_local_idx, path_scores) where best_local_idx is in 0..M-1.
        """
        m = len(self.active_indices)
        scores = np.zeros(m, dtype=np.float32)
        for s_idx in range(m):
            for pos in range(4):
                t_cls = self.tones[s_idx, pos]
                scores[s_idx] += self.tone_weight * float(tone_log_probs[pos, t_cls])
                if cons_log_probs is not None:
                    c_cls = self.consonants[s_idx, pos]
                    scores[s_idx] += self.consonant_weight * float(cons_log_probs[pos, c_cls])
        best_idx = int(np.argmax(scores))
        return best_idx, scores

    def decode_soft_int8(
        self,
        tone_q_scores: np.ndarray,
        cons_q_scores: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """
        Fixed-point INT8 score accumulation (exact match to MCU C99 int16_t accumulator).
        """
        m = len(self.active_indices)
        acc = np.zeros(m, dtype=np.int32)
        for s_idx in range(m):
            for pos in range(4):
                acc[s_idx] += int(tone_q_scores[pos, self.tones[s_idx, pos]])
                if cons_q_scores is not None:
                    acc[s_idx] += int(cons_q_scores[pos, self.consonants[s_idx, pos]])
        best_idx = int(np.argmax(acc))
        return best_idx, acc


def _find_snr_bias_for_target_acc(num_classes: int, target_acc: float, rng: np.random.Generator) -> float:
    """Calibrate additive Gaussian signal bias so single-trial top-1 accuracy matches target_acc."""
    lo, hi = 0.0, 5.0
    noise = rng.normal(0.0, 1.0, size=(20000, num_classes))
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        logits = noise.copy()
        logits[:, 0] += mid
        acc = float(np.mean(np.argmax(logits, axis=1) == 0))
        if acc < target_acc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_stage3_cascade_boost(
    single_tone_acc: float = 0.55,
    single_cons_acc: float = 0.48,
    num_trials: int = 4000,
    seed: int = 42,
    use_polarized_4: bool = False,
) -> Dict[str, float]:
    """
    Monte-Carlo validation (Thesis Sec 5.1):
    Compare unconstrained per-syllable argmax vs. Stage 3 soft-decision Viterbi lexicon decoding
    across FP32 and INT8 quantized log-likelihoods.
    """
    rng = np.random.default_rng(seed)
    subset = POLARIZED_SUBSET_4_INDICES if use_polarized_4 else None
    decoder = LexiconViterbiDecoder(subset_indices=subset)
    m = len(decoder.active_indices)

    tone_bias = _find_snr_bias_for_target_acc(4, single_tone_acc, rng)
    cons_bias = _find_snr_bias_for_target_acc(5, single_cons_acc, rng)

    raw_tone_only_exact = 0
    raw_joint_exact = 0
    viterbi_tone_only_hits = 0
    viterbi_fp32_hits = 0
    viterbi_int8_hits = 0
    realized_tone_correct = 0
    realized_cons_correct = 0

    for _ in range(num_trials):
        true_local_idx = int(rng.integers(0, m))
        true_tones = decoder.tones[true_local_idx]
        true_cons = decoder.consonants[true_local_idx]

        tone_logits = rng.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
        cons_logits = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float32)
        for pos in range(4):
            tone_logits[pos, true_tones[pos]] += tone_bias
            cons_logits[pos, true_cons[pos]] += cons_bias

        # Log-softmax normalization
        tone_log_probs = tone_logits - np.log(np.sum(np.exp(tone_logits), axis=1, keepdims=True))
        cons_log_probs = cons_logits - np.log(np.sum(np.exp(cons_logits), axis=1, keepdims=True))

        pred_tones_raw = np.argmax(tone_log_probs, axis=1)
        pred_cons_raw = np.argmax(cons_log_probs, axis=1)
        realized_tone_correct += int(np.sum(pred_tones_raw == true_tones))
        realized_cons_correct += int(np.sum(pred_cons_raw == true_cons))

        if np.array_equal(pred_tones_raw, true_tones):
            raw_tone_only_exact += 1
            if np.array_equal(pred_cons_raw, true_cons):
                raw_joint_exact += 1

        # Stage 3 Viterbi (tone-only vs. 8D joint FP32 vs. 8D joint INT8)
        pred_t_only, _ = decoder.decode_soft_fp32(tone_log_probs, None)
        if pred_t_only == true_local_idx:
            viterbi_tone_only_hits += 1

        pred_fp32, _ = decoder.decode_soft_fp32(tone_log_probs, cons_log_probs)
        if pred_fp32 == true_local_idx:
            viterbi_fp32_hits += 1

        # Quantize log-probs to symmetric INT8 [-127, 127]
        scale = 32.0
        tone_q = np.clip(np.round(tone_log_probs * scale), -128, 127).astype(np.int8)
        cons_q = np.clip(np.round(cons_log_probs * scale), -128, 127).astype(np.int8)
        pred_int8, _ = decoder.decode_soft_int8(tone_q, cons_q)
        if pred_int8 == true_local_idx:
            viterbi_int8_hits += 1

    return {
        "num_commands": float(m),
        "single_tone_acc": realized_tone_correct / float(num_trials * 4),
        "single_cons_acc": realized_cons_correct / float(num_trials * 4),
        "raw_tone_4char_acc": raw_tone_only_exact / float(num_trials),
        "raw_joint_8feat_acc": raw_joint_exact / float(num_trials),
        "viterbi_tone_only_acc": viterbi_tone_only_hits / float(num_trials),
        "viterbi_joint_fp32_acc": viterbi_fp32_hits / float(num_trials),
        "viterbi_joint_int8_acc": viterbi_int8_hits / float(num_trials),
    }
