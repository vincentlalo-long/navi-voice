"""Phase 3 enrichment utilities for Japanese ASR normalization, schema v2, and QC."""

from .asr import HardenedWhisperTranscriber
from .quality_control import evaluate_asr_gates, evaluate_slice_qc
from .schema import (
    SCHEMA_VERSION,
    compute_file_sha256,
    default_human_review,
    finalize_record_v2,
    merge_cached_record,
    resolve_effective_emotion,
)
from .text_norm import (
    DEFAULT_BOH_SUBSTRINGS,
    canonicalize_japanese_text,
    compute_speaking_rates,
    count_japanese_phonetic_units,
    detect_boh_hallucination,
    detect_repetition_anomaly,
)

__all__ = [
    "DEFAULT_BOH_SUBSTRINGS",
    "SCHEMA_VERSION",
    "HardenedWhisperTranscriber",
    "canonicalize_japanese_text",
    "compute_file_sha256",
    "compute_speaking_rates",
    "count_japanese_phonetic_units",
    "default_human_review",
    "detect_boh_hallucination",
    "detect_repetition_anomaly",
    "evaluate_asr_gates",
    "evaluate_slice_qc",
    "finalize_record_v2",
    "merge_cached_record",
    "resolve_effective_emotion",
]
