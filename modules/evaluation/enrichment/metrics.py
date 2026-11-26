"""Aggregate Phase 3 enrichment, QC waterfall, prosody, and expression metrics."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from modules.enrichment.quality_control import DEFAULT_SILVER_QC_CFG
from modules.enrichment.schema import resolve_effective_emotion


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Return value as a finite float, or default if missing/non-numeric/non-finite."""
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _extract_canonical_text(record: Dict[str, Any]) -> str:
    """Return canonical_text from asr_diagnostics with fallback to top-level text."""
    asr_diag = record.get("asr_diagnostics")
    if isinstance(asr_diag, dict) and asr_diag.get("canonical_text") is not None:
        return str(asr_diag["canonical_text"])
    return str(record.get("text") or "")


def _summarize_numeric(values: List[float], decimals: int = 4) -> Dict[str, Any]:
    """Compute distribution summary stats safely for empty or singleton inputs."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p20": 0.0,
            "p80": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    p20, p80 = np.percentile(arr, [20.0, 80.0])
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), decimals),
        "median": round(float(np.median(arr)), decimals),
        "std": round(float(np.std(arr)), decimals),
        "p20": round(float(p20), decimals),
        "p80": round(float(p80), decimals),
        "min": round(float(np.min(arr)), decimals),
        "max": round(float(np.max(arr)), decimals),
    }


def _is_acoustic_safe(record: Dict[str, Any]) -> bool:
    """Check Phase 2 acoustic safety aligned with SILVER QC limits."""
    status = str(record.get("status") or "")
    dur = _safe_float(record.get("duration_sec"), 0.0) or 0.0
    clip = _safe_float(record.get("clipping_rate"), 0.0) or 0.0
    disallowed = set(DEFAULT_SILVER_QC_CFG["disallowed_statuses"])
    return (
        status not in disallowed
        and clip < float(DEFAULT_SILVER_QC_CFG["max_clipping_rate_exclusive"])
        and float(DEFAULT_SILVER_QC_CFG["min_duration_sec"])
        <= dur
        <= float(DEFAULT_SILVER_QC_CFG["max_duration_sec"])
    )


def _compute_category_breakdown(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate slice counts, QC tier counts, and durations per effective category."""
    total_count = len(records)
    buckets: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cat, _, _ = resolve_effective_emotion(rec)
        dur = _safe_float(rec.get("duration_sec"), 0.0) or 0.0
        qc = rec.get("qc") if isinstance(rec.get("qc"), dict) else {}
        tier = str(qc.get("tier") or "")

        entry = buckets.setdefault(
            cat,
            {
                "count": 0,
                "gold_count": 0,
                "silver_count": 0,
                "reject_count": 0,
                "duration_sec": 0.0,
                "share_pct": 0.0,
            },
        )
        entry["count"] += 1
        if tier == "GOLD":
            entry["gold_count"] += 1
        elif tier == "SILVER":
            entry["silver_count"] += 1
        elif tier == "REJECT":
            entry["reject_count"] += 1
        entry["duration_sec"] += dur

    sorted_items = sorted(
        buckets.items(),
        key=lambda item: (-item[1]["count"], item[0]),
    )
    result: Dict[str, Dict[str, Any]] = {}
    for cat, stats in sorted_items:
        stats["duration_sec"] = round(stats["duration_sec"], 3)
        stats["share_pct"] = (
            round((stats["count"] / total_count) * 100.0, 2) if total_count > 0 else 0.0
        )
        result[cat] = stats
    return result


def compute_enrichment_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute waterfall, rejection, category, delivery, abstention, and prosody metrics."""
    total_slices = len(records)
    acoustic_safe = 0
    non_empty_asr = 0
    anti_hallucination_pass = 0
    silver_count = 0
    gold_count = 0
    reject_count = 0
    pending_asr_count = 0
    usable_for_training_count = 0
    reference_eligible_count = 0
    human_reviewed_count = 0

    total_duration_sec = 0.0
    usable_duration_sec = 0.0
    gold_duration_sec = 0.0

    rejection_reasons: Dict[str, int] = {}
    silver_downgrade_reasons: Dict[str, int] = {}
    has_any_qc = False

    for rec in records:
        dur = _safe_float(rec.get("duration_sec"), 0.0) or 0.0
        total_duration_sec += dur

        if _is_acoustic_safe(rec):
            acoustic_safe += 1
            text = _extract_canonical_text(rec).strip()
            if text:
                non_empty_asr += 1
                asr_diag = rec.get("asr_diagnostics")
                flags = (
                    list(asr_diag.get("hallucination_flags") or [])
                    if isinstance(asr_diag, dict)
                    else []
                )
                if not flags:
                    anti_hallucination_pass += 1

        qc = rec.get("qc")
        if isinstance(qc, dict):
            has_any_qc = True
            tier = qc.get("tier")
            if tier == "GOLD":
                gold_count += 1
                gold_duration_sec += dur
            elif tier == "SILVER":
                silver_count += 1
                for reason in qc.get("reasons") or []:
                    key = str(reason)
                    silver_downgrade_reasons[key] = silver_downgrade_reasons.get(key, 0) + 1
            elif tier == "REJECT":
                reject_count += 1
                for reason in qc.get("reasons") or []:
                    key = str(reason)
                    rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
            elif tier == "PENDING_ASR":
                pending_asr_count += 1

            if qc.get("usable_for_training") is True:
                usable_for_training_count += 1
                usable_duration_sec += dur
            if qc.get("usable_for_reference") is True:
                reference_eligible_count += 1

        human_review = rec.get("human_review")
        if isinstance(human_review, dict) and human_review.get("reviewed") is True:
            human_reviewed_count += 1

    retention_pct = (
        round((usable_for_training_count / total_slices) * 100.0, 2)
        if total_slices > 0
        else 0.0
    )
    gold_pct = (
        round((gold_count / total_slices) * 100.0, 2) if total_slices > 0 else 0.0
    )

    waterfall = {
        "total_slices": total_slices,
        "acoustic_safe": acoustic_safe,
        "non_empty_asr": non_empty_asr,
        "anti_hallucination_pass": anti_hallucination_pass,
        "usable_for_training": usable_for_training_count,
        "silver_count": silver_count,
        "gold_count": gold_count,
        "reject_count": reject_count,
        "pending_asr_count": pending_asr_count,
        "reference_eligible_count": reference_eligible_count,
        "human_reviewed_count": human_reviewed_count,
        "total_duration_sec": round(total_duration_sec, 3),
        "usable_duration_sec": round(usable_duration_sec, 3),
        "gold_duration_sec": round(gold_duration_sec, 3),
        "retention_pct": retention_pct,
        "gold_pct": gold_pct,
    }

    sorted_rejections = dict(
        sorted(rejection_reasons.items(), key=lambda item: (-item[1], item[0]))
    )
    sorted_silver_reasons = dict(
        sorted(silver_downgrade_reasons.items(), key=lambda item: (-item[1], item[0]))
    )

    usable_records = [
        rec
        for rec in records
        if isinstance(rec.get("qc"), dict) and rec["qc"].get("usable_for_training") is True
    ]
    analysis_records = usable_records if has_any_qc else list(records)

    categories = _compute_category_breakdown(analysis_records)
    categories_all = _compute_category_breakdown(records)

    delivery_buckets: Dict[str, Dict[str, Any]] = {}
    for rec in analysis_records:
        _, delivery_mode, _ = resolve_effective_emotion(rec)
        dur = _safe_float(rec.get("duration_sec"), 0.0) or 0.0
        entry = delivery_buckets.setdefault(
            delivery_mode,
            {"count": 0, "duration_sec": 0.0, "share_pct": 0.0},
        )
        entry["count"] += 1
        entry["duration_sec"] += dur

    analysis_total = len(analysis_records)
    delivery_modes: Dict[str, Dict[str, Any]] = {}
    for mode, stats in sorted(
        delivery_buckets.items(), key=lambda item: (-item[1]["count"], item[0])
    ):
        stats["duration_sec"] = round(stats["duration_sec"], 3)
        stats["share_pct"] = (
            round((stats["count"] / analysis_total) * 100.0, 2)
            if analysis_total > 0
            else 0.0
        )
        delivery_modes[mode] = stats

    def _abstention_stats(subset: List[Dict[str, Any]]) -> tuple[int, int, float]:
        tagged = 0
        abstained = 0
        for rec in subset:
            vocal = rec.get("vocal_expression")
            auto_tags = vocal.get("auto_tags") if isinstance(vocal, dict) else None
            if isinstance(auto_tags, dict):
                tagged += 1
                if (
                    auto_tags.get("status") == "abstained_uncertain"
                    or auto_tags.get("primary") == "Uncertain"
                ):
                    abstained += 1
        rate = round(abstained / tagged, 4) if tagged > 0 else 0.0
        return tagged, abstained, rate

    tagged_count, abstained_count, abstention_rate = _abstention_stats(analysis_records)
    all_tagged_count, all_abstained_count, all_abstention_rate = _abstention_stats(records)

    abstention = {
        "tagged_count": tagged_count,
        "abstained_count": abstained_count,
        "abstention_rate": abstention_rate,
        "all_tagged_count": all_tagged_count,
        "all_abstained_count": all_abstained_count,
        "all_abstention_rate": all_abstention_rate,
    }

    voiced_f0: List[float] = []
    voiced_f0_iqr: List[float] = []
    unvoiced_count = 0
    voiced_count = 0
    feature_lists: Dict[str, List[float]] = {
        "voiced_fraction": [],
        "spectral_centroid_hz": [],
        "spectral_flatness": [],
        "energy_std_db": [],
        "crest_factor_db": [],
        "raw_rms_db": [],
        "duration_sec": [],
        "effective_cps": [],
        "effective_mora_rate": [],
        "avg_logprob": [],
        "no_speech_prob": [],
        "compression_ratio": [],
    }

    for rec in analysis_records:
        dur = _safe_float(rec.get("duration_sec"))
        if dur is not None:
            feature_lists["duration_sec"].append(dur)
        rms = _safe_float(rec.get("raw_rms_db"))
        if rms is not None:
            feature_lists["raw_rms_db"].append(rms)

        prosody = rec.get("prosody")
        if isinstance(prosody, dict):
            vf = _safe_float(prosody.get("voiced_fraction"))
            f0 = _safe_float(prosody.get("f0_median_hz"))
            iqr = _safe_float(prosody.get("f0_iqr_hz"))
            if vf is not None and vf >= 0.15 and f0 is not None and f0 > 0.0:
                voiced_count += 1
                voiced_f0.append(f0)
                if iqr is not None:
                    voiced_f0_iqr.append(iqr)
            else:
                unvoiced_count += 1

            for key in (
                "voiced_fraction",
                "spectral_centroid_hz",
                "spectral_flatness",
                "energy_std_db",
                "crest_factor_db",
            ):
                val = _safe_float(prosody.get(key))
                if val is not None:
                    feature_lists[key].append(val)

        asr_diag = rec.get("asr_diagnostics")
        if isinstance(asr_diag, dict):
            for key in (
                "effective_cps",
                "effective_mora_rate",
                "avg_logprob",
                "no_speech_prob",
                "compression_ratio",
            ):
                val = _safe_float(asr_diag.get(key))
                if val is not None:
                    feature_lists[key].append(val)

    prosody_summary: Dict[str, Any] = {
        "scope": "usable_for_training" if has_any_qc else "all_records",
        "voiced_count": voiced_count,
        "unvoiced_count": unvoiced_count,
        "f0_median_hz": _summarize_numeric(voiced_f0, decimals=2),
        "f0_iqr_hz": _summarize_numeric(voiced_f0_iqr, decimals=2),
    }
    for key, vals in feature_lists.items():
        prosody_summary[key] = _summarize_numeric(vals, decimals=4)

    return {
        "waterfall": waterfall,
        "rejection_reasons": sorted_rejections,
        "silver_downgrade_reasons": sorted_silver_reasons,
        "categories": categories,
        "categories_all": categories_all,
        "delivery_modes": delivery_modes,
        "abstention": abstention,
        "prosody_summary": prosody_summary,
    }
