"""
Backward compatibility layer for modules.audio.
Preserves the legacy function signatures and return shapes while delegating
to the canonical implementations in modules.audio.utils.
"""

from typing import Any, Dict

import numpy as np

from modules.audio.utils import (
    apply_raised_cosine_fade,
    compute_audio_stats,
    normalize_loudness_ebur128,
    normalize_peak,
)
from modules.audio.utils import compute_edge_metrics as _compute_edge_metrics


def evaluate_boundary_quality(
    audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0
) -> Dict[str, Any]:
    """Legacy dict-returning boundary evaluation wrapper."""
    return _compute_edge_metrics(audio, sampling_rate, edge_ms=edge_ms)


def compute_edge_metrics(
    audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0
) -> Dict[str, Any]:
    """Legacy compact edge-metrics wrapper."""
    metrics = _compute_edge_metrics(audio, sampling_rate, edge_ms=edge_ms)
    return {
        "eer_start": metrics["eer_start"],
        "step_jump": metrics["step_jump"],
        "clean_boundary": metrics["clean_boundary"],
        "boundary_status": metrics["status"],
    }


__all__ = [
    "normalize_peak",
    "normalize_loudness_ebur128",
    "apply_raised_cosine_fade",
    "compute_audio_stats",
    "compute_edge_metrics",
    "evaluate_boundary_quality",
]
