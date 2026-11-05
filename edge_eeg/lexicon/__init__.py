"""Lexicon codebook and soft-decision Viterbi decoding for Stage 3 closed-vocabulary command recognition."""

from .codebook import (
    CODEBOOK_12,
    CODEBOOK_CONSONANTS,
    CODEBOOK_TONES,
    POLARIZED_SUBSET_4_INDICES,
    CommandEntry,
    compute_hamming_report,
)
from .viterbi import LexiconViterbiDecoder, simulate_stage3_cascade_boost

__all__ = [
    "CODEBOOK_12",
    "CODEBOOK_CONSONANTS",
    "CODEBOOK_TONES",
    "POLARIZED_SUBSET_4_INDICES",
    "CommandEntry",
    "compute_hamming_report",
    "LexiconViterbiDecoder",
    "simulate_stage3_cascade_boost",
]
