"""
Audio Utilities for DSP, Normalization & Quality Metrics.
"""

from typing import Any, Dict
import numpy as np


def normalize_peak(
    audio: np.ndarray, target_peak_db: float = -1.0, max_gain_db: float = 18.0
) -> np.ndarray:
    """
    Normalize peak amplitude to target dBFS with a maximum gain ceiling.
    Prevents blowing up the noise floor on quiet/whisper audio slices.
    """
    peak = np.max(np.abs(audio))
    if peak <= 1e-6:
        return audio

    target_peak = 10 ** (target_peak_db / 20.0)
    desired_gain = target_peak / peak
    max_gain = 10 ** (max_gain_db / 20.0)
    effective_gain = min(desired_gain, max_gain)

    return audio * effective_gain


def apply_raised_cosine_fade(
    audio: np.ndarray, sampling_rate: int, fade_ms: float = 5.0
) -> np.ndarray:
    """
    Apply a gentle raised-cosine (Hann-like) fade-in and fade-out to slice boundaries.
    Eraskes Gibbs transient clicks O(1/f) by ensuring amplitude smoothly transitions to 0.
    """
    fade_samples = int(sampling_rate * fade_ms / 1000.0)
    if len(audio) < fade_samples * 2 or fade_samples < 2:
        return audio

    out = audio.copy()
    # Fade-in: 0.5 * (1 - cos(pi * n / N)) from 0 to 1
    t_in = np.linspace(0, np.pi, fade_samples, endpoint=True)
    fade_in_curve = 0.5 * (1.0 - np.cos(t_in))
    out[:fade_samples] *= fade_in_curve

    # Fade-out: 0.5 * (1 + cos(pi * n / N)) from 1 to 0
    t_out = np.linspace(0, np.pi, fade_samples, endpoint=True)
    fade_out_curve = 0.5 * (1.0 + np.cos(t_out))
    out[-fade_samples:] *= fade_out_curve

    return out


def compute_audio_stats(audio: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Compute basic audio stats (duration, RMS level in dB, peak level in dB, clipping)."""
    duration = len(audio) / sampling_rate
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    rms_db = 20 * np.log10(rms)
    peak = np.max(np.abs(audio)) + 1e-9
    peak_db = 20 * np.log10(peak)

    # Clipping detection (samples near full-scale saturation >= 0.999)
    clipped_samples = int(np.sum(np.abs(audio) >= 0.999))
    clipping_rate = float(clipped_samples / (len(audio) + 1e-9))

    return {
        "duration_sec": round(duration, 3),
        "rms_db": round(float(rms_db), 2),
        "peak_db": round(float(peak_db), 2),
        "clipping_rate": round(clipping_rate, 5),
    }


def evaluate_boundary_quality(
    audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0
) -> Dict[str, Any]:
    """
    Canonical, unified evaluation of boundary edge metrics across the entire codebase.

    Criteria:
      - CLEAN: Both edges <= -35 dB AND EER <= 0.20 AND step_jump <= 0.02
      - SAFE:  Both edges <= -27 dB AND EER <= 0.45 AND step_jump <= 0.05
      - WARN:  Borderline or mid-phoneme cut
    """
    edge_samples = int(sampling_rate * edge_ms / 1000.0)
    if len(audio) < edge_samples * 2:
        return {
            "duration_sec": round(len(audio) / sampling_rate, 3),
            "start_rms_db": -99.0,
            "end_rms_db": -99.0,
            "core_rms_db": -99.0,
            "eer_start": 0.0,
            "eer_end": 0.0,
            "step_jump": 0.0,
            "status": "TOO_SHORT",
            "clean_boundary": True,
        }

    start_region = audio[:edge_samples]
    end_region = audio[-edge_samples:]
    core_region = audio[edge_samples:-edge_samples]

    rms_start = np.sqrt(np.mean(start_region**2)) + 1e-9
    rms_end = np.sqrt(np.mean(end_region**2)) + 1e-9
    rms_core = np.sqrt(np.mean(core_region**2)) + 1e-9

    start_db = float(20 * np.log10(rms_start))
    end_db = float(20 * np.log10(rms_end))
    core_db = float(20 * np.log10(rms_core))

    eer_start = float(rms_start / rms_core)
    eer_end = float(rms_end / rms_core)
    step_jump = float(max(abs(audio[0]), abs(audio[-1])))

    if (
        start_db <= -35.0
        and end_db <= -35.0
        and eer_start <= 0.20
        and eer_end <= 0.20
        and step_jump <= 0.02
    ):
        status = "CLEAN"
    elif (
        start_db <= -27.0
        and end_db <= -27.0
        and eer_start <= 0.45
        and eer_end <= 0.45
        and step_jump <= 0.05
    ):
        status = "SAFE"
    else:
        status = "WARN"

    clean_boundary = bool(status in ("CLEAN", "SAFE"))

    return {
        "duration_sec": round(len(audio) / sampling_rate, 3),
        "start_rms_db": round(start_db, 1),
        "end_rms_db": round(end_db, 1),
        "core_rms_db": round(core_db, 1),
        "eer_start": round(eer_start, 3),
        "eer_end": round(eer_end, 3),
        "step_jump": round(step_jump, 4),
        "status": status,
        "clean_boundary": clean_boundary,
    }


def compute_edge_metrics(
    audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0
) -> Dict[str, Any]:
    """Backward-compatible wrapper mapping to evaluate_boundary_quality."""
    metrics = evaluate_boundary_quality(audio, sampling_rate, edge_ms)
    return {
        "eer_start": metrics["eer_start"],
        "step_jump": metrics["step_jump"],
        "clean_boundary": metrics["clean_boundary"],
        "boundary_status": metrics["status"],
    }
