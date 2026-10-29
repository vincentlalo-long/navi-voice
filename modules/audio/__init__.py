"""
Audio processing, normalization, and acoustic DSP modules.
"""

from .utils import (
    apply_raised_cosine_fade,
    compute_audio_stats,
    compute_edge_metrics,
    evaluate_boundary_quality,
    normalize_loudness_ebur128,
    normalize_peak,
)

__all__ = [
    "normalize_peak",
    "normalize_loudness_ebur128",
    "apply_raised_cosine_fade",
    "compute_audio_stats",
    "compute_edge_metrics",
    "evaluate_boundary_quality",
]
