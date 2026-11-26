"""Schema v2.0 helpers, file hashing, provenance cache merge, and record finalization."""

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION: str = "2.0"
_HASH_CHUNK_BYTES = 65536


def compute_file_sha256(path: Path) -> str:
    """Compute a chunked 64 KB SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_human_review() -> Dict[str, Any]:
    """Return the default unreviewed human_review block for Schema v2.0."""
    return {
        "reviewed": False,
        "verified_primary": None,
        "verified_delivery": None,
        "notes": "",
    }


def _normalize_human_review(raw_review: Any) -> Dict[str, Any]:
    """Normalize a cached human_review dict while preserving defaults."""
    normalized = default_human_review()
    if isinstance(raw_review, dict):
        normalized.update(copy.deepcopy(raw_review))
        normalized["reviewed"] = bool(normalized.get("reviewed", False))
        if normalized.get("notes") is None:
            normalized["notes"] = ""
    return normalized


def merge_cached_record(
    fresh_base: Dict[str, Any],
    cached: Optional[Dict[str, Any]],
    audio_sha256: str,
    expected_asr_model: str,
) -> Tuple[Dict[str, Any], bool]:
    """Merge safe cached fields onto a fresh Phase 2 base record.

    Preserves Phase 2 telemetry in ``fresh_base`` without mutating caller inputs.
    Reuses ``human_review`` only when ``audio_sha256`` matches, and reuses
    ``asr_diagnostics`` only when both ``audio_sha256`` and ``asr_model`` match.
    """
    merged = copy.deepcopy(fresh_base)
    prov = merged.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        merged["provenance"] = prov
    prov["schema_version"] = SCHEMA_VERSION
    prov["audio_sha256"] = audio_sha256

    if not isinstance(cached, dict):
        merged["human_review"] = default_human_review()
        return merged, False

    cached_prov = cached.get("provenance")
    if not isinstance(cached_prov, dict) or cached_prov.get("audio_sha256") != audio_sha256:
        merged["human_review"] = default_human_review()
        return merged, False

    merged["human_review"] = _normalize_human_review(cached.get("human_review"))

    cached_asr = cached.get("asr_diagnostics")
    can_reuse_asr = (
        cached_prov.get("asr_model") == expected_asr_model
        and isinstance(cached_asr, dict)
        and cached_asr.get("canonical_text") is not None
    )
    if can_reuse_asr:
        merged["asr_diagnostics"] = copy.deepcopy(cached_asr)
        prov["asr_model"] = expected_asr_model

    return merged, can_reuse_asr


def resolve_effective_emotion(record: Dict[str, Any]) -> Tuple[str, str, bool]:
    """Resolve effective (category, delivery_mode, is_human_verified) for a record."""
    vocal = record.get("vocal_expression")
    vocal_dict = vocal if isinstance(vocal, dict) else {}
    auto_tags = vocal_dict.get("auto_tags")
    auto_dict = auto_tags if isinstance(auto_tags, dict) else {}

    auto_primary = auto_dict.get("primary") or record.get("category") or "Neutral/Dialogue"
    auto_delivery = vocal_dict.get("delivery_mode") or "normal"

    human_review = record.get("human_review")
    if isinstance(human_review, dict) and human_review.get("reviewed") is True:
        verified_primary = human_review.get("verified_primary") or auto_primary
        verified_delivery = human_review.get("verified_delivery") or auto_delivery
        return str(verified_primary), str(verified_delivery), True

    return str(auto_primary), str(auto_delivery), False


def finalize_record_v2(record: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronize legacy flat fields and Schema v2.0 provenance/review defaults."""
    asr_diag = record.get("asr_diagnostics")
    if isinstance(asr_diag, dict):
        canonical_text = asr_diag.get("canonical_text")
        if canonical_text is not None:
            record["text"] = str(canonical_text)
        else:
            record.setdefault("text", "")

        eff_cps = asr_diag.get("effective_cps")
        if eff_cps is not None:
            try:
                eff_cps_val = float(eff_cps)
                record["chars_per_sec"] = (
                    round(eff_cps_val, 2) if math.isfinite(eff_cps_val) else 0.0
                )
            except (TypeError, ValueError):
                record.setdefault("chars_per_sec", 0.0)
        else:
            record.setdefault("chars_per_sec", 0.0)
    else:
        record.setdefault("text", "")
        record.setdefault("chars_per_sec", 0.0)

    effective_category, _, _ = resolve_effective_emotion(record)
    record["category"] = effective_category

    prov = record.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        record["provenance"] = prov
    prov["schema_version"] = SCHEMA_VERSION

    defaults = default_human_review()
    human_review = record.get("human_review")
    if not isinstance(human_review, dict):
        record["human_review"] = defaults
    else:
        for key, value in defaults.items():
            human_review.setdefault(key, value)

    return record
