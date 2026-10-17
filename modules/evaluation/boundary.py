"""
Boundary Quality & Acoustic Edge Metric Evaluator.
"""

from typing import Any, Dict, List
import numpy as np


def evaluate_slice_boundary(
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


def aggregate_boundary_metrics(slice_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate boundary statistics across a set of slice evaluations:
    CLEAN%, SAFE%, WARN%, mean/p95 edge RMS, and mean/p95 EER.
    """
    total = len(slice_evaluations)
    if total == 0:
        return {
            "total_slices": 0,
            "clean_pct": 0.0,
            "safe_pct": 0.0,
            "warn_pct": 0.0,
            "cbr_pct": 0.0,  # Clean Boundary Rate (CLEAN + SAFE)
            "mean_edge_rms_db": 0.0,
            "p95_edge_rms_db": 0.0,
            "mean_eer": 0.0,
            "p95_eer": 0.0,
            "max_step_jump": 0.0,
        }

    clean_count = sum(1 for m in slice_evaluations if m.get("status") == "CLEAN")
    safe_count = sum(1 for m in slice_evaluations if m.get("status") == "SAFE")
    warn_count = sum(1 for m in slice_evaluations if m.get("status") == "WARN")

    edge_rms_values = []
    eer_values = []
    step_jumps = []

    for m in slice_evaluations:
        edge_rms_values.extend([m.get("start_rms_db", -99.0), m.get("end_rms_db", -99.0)])
        eer_values.extend([m.get("eer_start", 0.0), m.get("eer_end", 0.0)])
        step_jumps.append(m.get("step_jump", 0.0))

    clean_pct = round((clean_count / total) * 100.0, 1)
    safe_pct = round((safe_count / total) * 100.0, 1)
    warn_pct = round((warn_count / total) * 100.0, 1)
    cbr_pct = round(clean_pct + safe_pct, 1)

    return {
        "total_slices": total,
        "clean_pct": clean_pct,
        "safe_pct": safe_pct,
        "warn_pct": warn_pct,
        "cbr_pct": cbr_pct,
        "mean_edge_rms_db": round(float(np.mean(edge_rms_values)), 1),
        "p95_edge_rms_db": round(float(np.percentile(edge_rms_values, 95)), 1),
        "mean_eer": round(float(np.mean(eer_values)), 3),
        "p95_eer": round(float(np.percentile(eer_values, 95)), 3),
        "max_step_jump": round(float(np.max(step_jumps)), 4),
    }
