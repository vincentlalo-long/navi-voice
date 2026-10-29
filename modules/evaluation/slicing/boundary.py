"""
Phase 1: Audio Segmentation & Slicing Boundary Quality Evaluation.
Measures acoustic boundary cleanliness, zero-crossing quality, energy discontinuity (EER),
and academic boundary tolerance metrics (Precision, Recall, F1@tolerance, R-value).
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
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "start_rms_db": -99.0,
            "end_rms_db": -99.0,
            "core_rms_db": -99.0,
            "eer_start": 0.0,
            "eer_end": 0.0,
            "step_jump": 0.0,
            "status": "WARN",
            "clean_boundary": False,
        }

    edge_samples = int(sampling_rate * edge_ms / 1000.0)
    dur = len(audio) / sampling_rate
    step_jump = float(max(abs(audio[0]), abs(audio[-1])))

    if len(audio) < edge_samples * 2:
        return {
            "duration_sec": round(dur, 3),
            "start_rms_db": -99.0,
            "end_rms_db": -99.0,
            "core_rms_db": -99.0,
            "eer_start": 0.0,
            "eer_end": 0.0,
            "step_jump": round(step_jump, 4),
            "status": "TOO_SHORT",
            "clean_boundary": True,
        }

    start_reg = audio[:edge_samples]
    end_reg = audio[-edge_samples:]
    core_reg = audio[edge_samples:-edge_samples]

    rms_start = np.sqrt(np.mean(start_reg**2)) + 1e-9
    rms_end = np.sqrt(np.mean(end_reg**2)) + 1e-9
    rms_core = np.sqrt(np.mean(core_reg**2)) + 1e-9

    start_db = float(20 * np.log10(rms_start))
    end_db = float(20 * np.log10(rms_end))
    core_db = float(20 * np.log10(rms_core))

    eer_start = float(rms_start / rms_core)
    eer_end = float(rms_end / rms_core)

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

    return {
        "duration_sec": round(dur, 3),
        "start_rms_db": round(start_db, 1),
        "end_rms_db": round(end_db, 1),
        "core_rms_db": round(core_db, 1),
        "eer_start": round(eer_start, 3),
        "eer_end": round(eer_end, 3),
        "step_jump": round(step_jump, 4),
        "status": status,
        "clean_boundary": bool(status in ("CLEAN", "SAFE")),
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
            "cbr_pct": 0.0,
            "clean_boundary_rate": 0.0,
            "clean_edge_pct": 0.0,
            "warn_edge_pct": 0.0,
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
        "clean_boundary_rate": cbr_pct,
        "clean_edge_pct": clean_pct,
        "warn_edge_pct": warn_pct,
        "mean_edge_rms_db": round(float(np.mean(edge_rms_values)), 1),
        "p95_edge_rms_db": round(float(np.percentile(edge_rms_values, 95)), 1),
        "mean_eer": round(float(np.mean(eer_values)), 3),
        "p95_eer": round(float(np.percentile(eer_values, 95)), 3),
        "max_step_jump": round(float(np.max(step_jumps)), 4),
    }


def calculate_boundary_tolerance_metrics(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
    tolerance_sec: float = 0.025,
) -> Dict[str, float]:
    """
    Standard academic boundary metrics (Precision, Recall, F1@tolerance, R-value, Over-segmentation).
    Based on Räsänen et al. (Interspeech 2009) and Buckeye/TIMIT evaluation standards.

    R-value formula (Räsänen et al., 2009):
        r1 = sqrt((1 - Recall)^2 + OS^2)
        r2 = |Recall - OS - 1| / sqrt(2)
        R-val = 1 - (r1 + r2) / 2
    """
    if len(gt_boundaries) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "r_value": 0.0,
            "os_rate": 0.0,
            "tolerance_ms": tolerance_sec * 1000.0,
        }

    pred_sorted = np.sort(pred_boundaries)
    gt_sorted = np.sort(gt_boundaries)
    tp = 0

    # Maximum-cardinality non-crossing matching. For 1-D boundaries with an
    # absolute tolerance, an optimal matching can always be chosen non-crossing,
    # so dynamic programming yields the true maximum (not a nearest-neighbour
    # greedy approximation).
    n_pred = len(pred_sorted)
    n_gt = len(gt_sorted)
    if n_pred > 0 and n_gt > 0:
        prev = np.zeros(n_gt + 1, dtype=np.int32)
        for i in range(1, n_pred + 1):
            cur = np.zeros(n_gt + 1, dtype=np.int32)
            p = pred_sorted[i - 1]
            for j in range(1, n_gt + 1):
                best = max(int(prev[j]), int(cur[j - 1]))
                if abs(p - gt_sorted[j - 1]) <= tolerance_sec:
                    best = max(best, int(prev[j - 1]) + 1)
                cur[j] = best
            prev = cur
        tp = int(prev[n_gt])

    precision = tp / len(pred_sorted) if len(pred_sorted) > 0 else 0.0
    recall = tp / len(gt_sorted) if len(gt_sorted) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    os_rate = (len(pred_sorted) / len(gt_sorted)) - 1.0

    r1 = np.sqrt((1.0 - recall) ** 2 + os_rate ** 2)
    r2 = abs(recall - os_rate - 1.0) / np.sqrt(2.0)
    r_val = 1.0 - (r1 + r2) / 2.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "r_value": round(float(r_val), 4),
        "os_rate": round(float(os_rate), 4),
        "tolerance_ms": round(tolerance_sec * 1000.0, 1),
    }
