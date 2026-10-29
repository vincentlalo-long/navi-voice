"""
Audio Processing & Acoustic Digital Signal Processing (DSP) Utilities.
Provides peak normalization, EBU R128 integrated loudness normalization,
raised-cosine boundary fading, and acoustic boundary quality metrics.
"""

from typing import Any, Dict, Tuple
import numpy as np


def normalize_peak(
    audio: np.ndarray, target_peak_db: float = -1.0, max_gain_db: float = 18.0
) -> np.ndarray:
    """
    Scale audio so that peak amplitude reaches target_peak_db (FS).
    Clamps maximum amplification to max_gain_db to prevent noise explosion.
    """
    if len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak <= 1e-6:
        return audio

    target_peak = 10 ** (target_peak_db / 20.0)
    desired_gain = target_peak / peak
    max_gain = 10 ** (max_gain_db / 20.0)
    effective_gain = min(desired_gain, max_gain)

    return audio * effective_gain


def normalize_loudness_ebur128(
    audio: np.ndarray,
    sampling_rate: int,
    target_lufs: float = -20.0,
    max_gain_db: float = 18.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    """
    Normalize perceived loudness to target LUFS using ITU-R BS.1770-4 / EBU R128.
    Applies max gain ceiling and peak limiter to prevent noise explosion and digital clipping.

    Note:
      Audio chunks shorter than 400ms (EBU R128 minimum gating block) cannot be reliably
      measured by pyloudnorm and will safely fall back to peak normalization.
    """
    if len(audio) == 0 or np.max(np.abs(audio)) <= 1e-6:
        return audio

    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sampling_rate)
        loudness = meter.integrated_loudness(audio)
        if np.isneginf(loudness) or np.isnan(loudness):
            return normalize_peak(audio, target_peak_db=peak_ceiling_db, max_gain_db=max_gain_db)

        gain_db = min(target_lufs - loudness, max_gain_db)
        normalized = audio * (10.0 ** (gain_db / 20.0))
        peak = np.max(np.abs(normalized))
        target_peak = 10.0 ** (peak_ceiling_db / 20.0)
        if peak > target_peak:
            normalized = normalized * (target_peak / peak)
        return normalized
    except ImportError:
        print("[!] Warning: 'pyloudnorm' is not installed. Falling back to peak normalization.")
        return normalize_peak(audio, target_peak_db=peak_ceiling_db, max_gain_db=max_gain_db)
    except (ValueError, ZeroDivisionError):
        # EBU R128 gating requires at least ~400ms; short slices fall back to peak.
        return normalize_peak(audio, target_peak_db=peak_ceiling_db, max_gain_db=max_gain_db)


def apply_raised_cosine_fade(
    audio: np.ndarray, sampling_rate: int, fade_ms: float = 5.0
) -> np.ndarray:
    """
    Apply a gentle raised-cosine (Hann-like) fade-in and fade-out to slice boundaries.
    Erases Gibbs transient clicks O(1/f) by ensuring amplitude smoothly transitions to 0.
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
    """Calculate core acoustic telemetry metrics."""
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "peak_db": -99.0,
            "rms_db": -99.0,
            "clipping_rate": 0.0,
        }

    duration = len(audio) / sampling_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    peak_db = float(20 * np.log10(peak)) if peak > 1e-6 else -99.0
    rms_db = float(20 * np.log10(rms)) if rms > 1e-6 else -99.0

    # Clipping detection threshold at -0.05 dBFS
    clip_thresh = 10 ** (-0.05 / 20.0)
    clipping_rate = float(np.mean(np.abs(audio) >= clip_thresh))

    return {
        "duration_sec": round(duration, 3),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "clipping_rate": round(clipping_rate, 6),
    }


def compute_edge_metrics(
    audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0
) -> Dict[str, Any]:
    """
    Measure energy discontinuity and step jumps at slice boundaries.
    """
    edge_samples = int(sampling_rate * edge_ms / 1000.0)
    if len(audio) < edge_samples * 2:
        return {
            "duration_sec": round(len(audio) / sampling_rate, 3) if len(audio) > 0 else 0.0,
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


def evaluate_boundary_quality(audio: np.ndarray, sampling_rate: int) -> Tuple[bool, str]:
    """Convenience wrapper for boundary validation."""
    metrics = compute_edge_metrics(audio, sampling_rate)
    return metrics["clean_boundary"], metrics["status"]
