"""4-Gate ASR hallucination checks and 3-Tier (GOLD/SILVER/REJECT) QC engine."""

import math
from typing import Any, Dict, List, Optional

from .schema import resolve_effective_emotion
from .text_norm import (
    DEFAULT_BOH_SUBSTRINGS,
    count_japanese_phonetic_units,
    detect_boh_hallucination,
    detect_repetition_anomaly,
)


def _safe_float(value: Any, default: float) -> float:
    """Convert value to a finite float, returning default on None/invalid/non-finite."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default

DEFAULT_ASR_GATE_CFG: Dict[str, Any] = {
    "max_compression_ratio": 2.4,
    "min_avg_logprob": -1.0,
    "max_no_speech_prob": 0.6,
    "no_speech_min_avg_logprob": -0.8,
    "min_mean_word_prob": 0.45,
    "under_rate_min_duration_sec": 3.0,
    "min_effective_cps": 1.2,
    "max_effective_cps": 18.0,
    "max_effective_mora_rate": 25.0,
    "max_consecutive_repeats": 4,
}

DEFAULT_GOLD_QC_CFG: Dict[str, Any] = {
    "allowed_statuses": ["CLEAN", "SAFE"],
    "max_clipping_rate": 0.0,
    "min_duration_sec": 2.0,
    "max_duration_sec": 10.0,
    "min_avg_logprob": -0.60,
    "max_no_speech_prob": 0.40,
    "max_compression_ratio": 2.0,
    "min_effective_cps": 2.0,
    "max_effective_cps": 15.0,
    "reference_status": "CLEAN",
    "reference_min_duration_sec": 3.5,
    "reference_max_duration_sec": 8.0,
}

DEFAULT_SILVER_QC_CFG: Dict[str, Any] = {
    "disallowed_statuses": ["TOO_SHORT"],
    "max_clipping_rate_exclusive": 0.001,
    "min_duration_sec": 1.2,
    "max_duration_sec": 12.0,
    "min_avg_logprob": -1.00,
    "max_no_speech_prob": 0.60,
    "max_compression_ratio": 2.4,
    "min_effective_cps": 1.2,
    "max_effective_cps": 18.0,
}


def _extract_section_cfg(
    cfg: Optional[Dict[str, Any]],
    section: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge defaults with a nested or flat QC gate config section."""
    resolved = dict(defaults)
    if not isinstance(cfg, dict):
        return resolved

    candidate: Any = cfg
    if isinstance(candidate.get("enrichment"), dict):
        candidate = candidate["enrichment"]
    if isinstance(candidate.get("qc_gates"), dict):
        candidate = candidate["qc_gates"]
    if isinstance(candidate.get(section), dict):
        candidate = candidate[section]
    elif any(k in candidate for k in ("asr", "gold", "silver")):
        # Caller passed a qc_gates container without this specific section key.
        return resolved

    if isinstance(candidate, dict):
        for key, value in candidate.items():
            if value is not None:
                resolved[key] = value
    return resolved


def evaluate_asr_gates(
    asr_diag: Dict[str, Any],
    duration_sec: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Evaluate the 4 anti-hallucination ASR gates and return triggered flag codes."""
    gate_cfg = _extract_section_cfg(cfg, "asr", DEFAULT_ASR_GATE_CFG)
    diag = asr_diag if isinstance(asr_diag, dict) else {}

    raw_text = str(diag.get("raw_text") or "")
    canonical_text = str(diag.get("canonical_text") or "")
    if not raw_text and canonical_text:
        raw_text = canonical_text

    clean_char_count = diag.get("clean_char_count")
    if clean_char_count is None:
        clean_char_count = count_japanese_phonetic_units(canonical_text)[0]
    else:
        clean_char_count = int(clean_char_count)

    avg_logprob = _safe_float(diag.get("avg_logprob"), -99.0)
    no_speech_prob = _safe_float(diag.get("no_speech_prob"), 1.0)
    compression_ratio = _safe_float(diag.get("compression_ratio"), 0.0)
    mean_word_prob = _safe_float(diag.get("mean_word_prob"), 0.0)
    effective_cps = _safe_float(diag.get("effective_cps"), 0.0)
    effective_mora_rate = _safe_float(diag.get("effective_mora_rate"), 0.0)
    dur_val = _safe_float(duration_sec, 0.0)

    flags: List[str] = []

    # Gate 1: Confidence & Compression
    if clean_char_count == 0:
        flags.append("ASR_EMPTY_TEXT")
    if compression_ratio > float(gate_cfg["max_compression_ratio"]):
        flags.append("ASR_HIGH_COMPRESSION")
    if avg_logprob < float(gate_cfg["min_avg_logprob"]):
        flags.append("ASR_LOW_LOGPROB")
    if (
        no_speech_prob > float(gate_cfg["max_no_speech_prob"])
        and avg_logprob < float(gate_cfg["no_speech_min_avg_logprob"])
    ):
        flags.append("ASR_HIGH_NOSPEECH")
    if mean_word_prob < float(gate_cfg["min_mean_word_prob"]):
        flags.append("ASR_LOW_WORD_PROB")

    # Gate 2: Bag-of-Hallucinations
    blacklist = gate_cfg.get("boh_blacklist", DEFAULT_BOH_SUBSTRINGS)
    if detect_boh_hallucination(raw_text, blacklist=blacklist) is not None:
        flags.append("ASR_BOH_BLACKLIST")

    # Gate 3: Repetition Entropy
    max_repeats = int(gate_cfg.get("max_consecutive_repeats", 4))
    if detect_repetition_anomaly(canonical_text, max_consecutive_repeats=max_repeats):
        flags.append("ASR_NGRAM_REPETITION")

    # Gate 4: Speaking Rate
    if (
        dur_val > float(gate_cfg["under_rate_min_duration_sec"])
        and effective_cps < float(gate_cfg["min_effective_cps"])
    ):
        flags.append("ASR_UNDER_RATE")
    if (
        effective_cps > float(gate_cfg["max_effective_cps"])
        or effective_mora_rate > float(gate_cfg["max_effective_mora_rate"])
    ):
        flags.append("ASR_OVER_RATE")

    return flags


def evaluate_slice_qc(
    record: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a slice into GOLD, SILVER, or REJECT with training/reference flags."""
    gold_cfg = _extract_section_cfg(cfg, "gold", DEFAULT_GOLD_QC_CFG)
    silver_cfg = _extract_section_cfg(cfg, "silver", DEFAULT_SILVER_QC_CFG)

    status = str(record.get("status") or "")
    duration_sec = _safe_float(record.get("duration_sec"), 0.0)
    clipping_rate = _safe_float(record.get("clipping_rate"), 0.0)

    asr_diag = record.get("asr_diagnostics")
    diag = asr_diag if isinstance(asr_diag, dict) else {}
    canonical_text = str(diag.get("canonical_text") or "")
    hallucination_flags = list(diag.get("hallucination_flags") or [])
    avg_logprob = _safe_float(diag.get("avg_logprob"), -99.0)
    no_speech_prob = _safe_float(diag.get("no_speech_prob"), 1.0)
    compression_ratio = _safe_float(diag.get("compression_ratio"), 0.0)
    effective_cps = _safe_float(diag.get("effective_cps"), 0.0)

    reject_reasons: List[str] = []
    for flag in hallucination_flags:
        if flag not in reject_reasons:
            reject_reasons.append(str(flag))

    if canonical_text == "":
        reject_reasons.append("empty_canonical_text")
    if status in set(silver_cfg["disallowed_statuses"]):
        reject_reasons.append(f"status_disallowed:{status}")
    if clipping_rate >= float(silver_cfg["max_clipping_rate_exclusive"]):
        reject_reasons.append("clipping_rate_exceeded")
    if not (float(silver_cfg["min_duration_sec"]) <= duration_sec <= float(silver_cfg["max_duration_sec"])):
        reject_reasons.append("duration_outside_silver_range")
    if avg_logprob < float(silver_cfg["min_avg_logprob"]):
        reject_reasons.append("avg_logprob_below_silver")
    if no_speech_prob > float(silver_cfg["max_no_speech_prob"]):
        reject_reasons.append("no_speech_prob_above_silver")
    if compression_ratio > float(silver_cfg["max_compression_ratio"]):
        reject_reasons.append("compression_ratio_above_silver")
    if not (float(silver_cfg["min_effective_cps"]) <= effective_cps <= float(silver_cfg["max_effective_cps"])):
        reject_reasons.append("effective_cps_outside_silver_range")

    if reject_reasons:
        deduped_reasons = list(dict.fromkeys(reject_reasons))
        return {
            "tier": "REJECT",
            "reasons": deduped_reasons,
            "usable_for_training": False,
            "usable_for_reference": False,
        }

    silver_reasons: List[str] = []
    if status not in set(gold_cfg["allowed_statuses"]):
        silver_reasons.append(f"status_not_gold:{status}")
    if clipping_rate > float(gold_cfg["max_clipping_rate"]):
        silver_reasons.append("clipping_rate_nonzero")
    if not (float(gold_cfg["min_duration_sec"]) <= duration_sec <= float(gold_cfg["max_duration_sec"])):
        silver_reasons.append("duration_outside_gold_range")
    if avg_logprob < float(gold_cfg["min_avg_logprob"]):
        silver_reasons.append("avg_logprob_below_gold")
    if no_speech_prob > float(gold_cfg["max_no_speech_prob"]):
        silver_reasons.append("no_speech_prob_above_gold")
    if compression_ratio > float(gold_cfg["max_compression_ratio"]):
        silver_reasons.append("compression_ratio_above_gold")
    if not (float(gold_cfg["min_effective_cps"]) <= effective_cps <= float(gold_cfg["max_effective_cps"])):
        silver_reasons.append("effective_cps_outside_gold_range")

    if silver_reasons:
        return {
            "tier": "SILVER",
            "reasons": silver_reasons,
            "usable_for_training": True,
            "usable_for_reference": False,
        }

    effective_category, _, _ = resolve_effective_emotion(record)
    usable_for_reference = (
        status == str(gold_cfg["reference_status"])
        and float(gold_cfg["reference_min_duration_sec"])
        <= duration_sec
        <= float(gold_cfg["reference_max_duration_sec"])
        and effective_category != "Uncertain"
    )
    return {
        "tier": "GOLD",
        "reasons": [],
        "usable_for_training": True,
        "usable_for_reference": usable_for_reference,
    }
