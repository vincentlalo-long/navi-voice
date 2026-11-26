"""Leakage-safe dataset packager and two-stage reference prompt curation."""

import copy
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .quality_control import evaluate_slice_qc
from .schema import default_human_review, finalize_record_v2, resolve_effective_emotion

DEFAULT_PACKAGER_CFG: Dict[str, Any] = {
    "val_ratio": 0.05,
    "block_size": 20,
    "seed": 42,
    "top_candidates_per_cat": 8,
    "active_per_cat": 3,
}

_LANG_LABELS: Dict[str, str] = {
    "JA": "Japanese",
    "JP": "Japanese",
    "EN": "English",
    "ZH": "Chinese",
    "KO": "Korean",
}

_CSV_FIELDNAMES: List[str] = [
    "id",
    "filename",
    "relative_path",
    "absolute_path",
    "source_file",
    "start_time_sec",
    "end_time_sec",
    "slicer_method",
    "duration_sec",
    "status",
    "clean_boundary",
    "raw_rms_db",
    "raw_peak_db",
    "applied_gain_db",
    "text",
    "chars_per_sec",
    "category",
    "qc_tier",
    "usable_for_training",
    "usable_for_reference",
    "delivery_mode",
    "effective_category",
    "auto_confidence",
    "human_reviewed",
    "verified_primary",
    "verified_delivery",
    "f0_median_hz",
    "f0_iqr_hz",
    "voiced_fraction",
    "spectral_centroid_hz",
    "energy_std_db",
    "pitch_bin",
    "speed_bin",
    "energy_bin",
    "effective_mora_rate",
    "avg_logprob",
    "no_speech_prob",
    "compression_ratio",
    "mean_word_prob",
    "audio_sha256",
    "asr_model",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to a finite float, returning default on invalid/non-finite input."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int, returning default on None or invalid input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _record_sort_key(record: Dict[str, Any]) -> Tuple[str, float, int]:
    """Deterministic temporal sort key: (source_file, start_time_sec, id)."""
    return (
        str(record.get("source_file") or ""),
        _safe_float(record.get("start_time_sec"), 0.0),
        _safe_int(record.get("id"), 0),
    )


def _extract_canonical_text(record: Dict[str, Any]) -> str:
    """Return canonical_text from asr_diagnostics with fallback to top-level text."""
    asr_diag = record.get("asr_diagnostics")
    if isinstance(asr_diag, dict) and asr_diag.get("canonical_text") is not None:
        return str(asr_diag["canonical_text"])
    return str(record.get("text") or "")


def _copy_human_review(raw_review: Any) -> Dict[str, Any]:
    """Return a clean copy of human_review populated with Schema v2.0 defaults."""
    copied = default_human_review()
    if isinstance(raw_review, dict):
        copied.update(copy.deepcopy(raw_review))
        copied["reviewed"] = bool(copied.get("reviewed", False))
        if copied.get("notes") is None:
            copied["notes"] = ""
    return copied


def _extract_packager_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve packager config from cfg['enrichment']['packager'], cfg['packager'], or flat cfg."""
    resolved = dict(DEFAULT_PACKAGER_CFG)
    if not isinstance(cfg, dict):
        return resolved

    candidate: Any = cfg
    if isinstance(candidate.get("enrichment"), dict):
        candidate = candidate["enrichment"]
    if isinstance(candidate.get("packager"), dict):
        candidate = candidate["packager"]

    if isinstance(candidate, dict):
        for key in DEFAULT_PACKAGER_CFG:
            if key in candidate and candidate[key] is not None:
                resolved[key] = candidate[key]
    return resolved


def sanitize_list_field(text: str) -> str:
    """Strip pipe delimiters and control whitespace so .list lines keep 4 columns."""
    replaced = re.sub(r"[|\r\n\t]", " ", str(text or ""))
    return re.sub(r"\s+", " ", replaced).strip()


class _UnionFind:
    """Disjoint-set structure for merging temporally adjacent blocks with shared text."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, idx: int) -> int:
        while self.parent[idx] != idx:
            self.parent[idx] = self.parent[self.parent[idx]]
            idx = self.parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def build_leakage_safe_splits(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.05,
    block_size: int = 20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split usable training records into leakage-safe (train, val) lists.

    Records are partitioned into contiguous same-source blocks of ``block_size``,
    merged across blocks whenever they share an identical non-empty
    ``canonical_text`` (or ``text``), and assigned to validation deterministically
    while preferring emotion category coverage and keeping train non-empty.
    """
    usable = [
        rec
        for rec in records
        if isinstance(rec.get("qc"), dict) and rec["qc"].get("usable_for_training") is True
    ]
    if not usable:
        return [], []

    sorted_usable = sorted(usable, key=_record_sort_key)
    effective_block_size = max(1, int(block_size))

    # Partition into contiguous blocks without ever mixing different source_files.
    blocks: List[List[int]] = []
    current_block: List[int] = []
    current_source: Optional[str] = None
    for pos, rec in enumerate(sorted_usable):
        source = str(rec.get("source_file") or "")
        if (
            current_block
            and (source != current_source or len(current_block) >= effective_block_size)
        ):
            blocks.append(current_block)
            current_block = []
        current_source = source
        current_block.append(pos)
    if current_block:
        blocks.append(current_block)

    # Union-find merge blocks that share identical non-empty canonical_text.
    uf = _UnionFind(len(blocks))
    text_to_first_block: Dict[str, int] = {}
    for block_idx, positions in enumerate(blocks):
        for pos in positions:
            text_key = _extract_canonical_text(sorted_usable[pos]).strip()
            if not text_key:
                continue
            first_block = text_to_first_block.get(text_key)
            if first_block is None:
                text_to_first_block[text_key] = block_idx
            else:
                uf.union(first_block, block_idx)

    comp_map: Dict[int, List[int]] = {}
    for block_idx, positions in enumerate(blocks):
        root = uf.find(block_idx)
        comp_map.setdefault(root, []).extend(positions)

    components: List[List[int]] = [sorted(positions) for _, positions in sorted(comp_map.items())]
    if len(components) < 2 or val_ratio <= 0.0:
        return sorted_usable, []

    comp_cats: List[set[str]] = [
        {resolve_effective_emotion(sorted_usable[pos])[0] for pos in comp}
        for comp in components
    ]
    comp_sizes: List[int] = [len(comp) for comp in components]
    total_count = len(sorted_usable)
    target_val_count = max(1, round(total_count * float(val_ratio)))

    rng = random.Random(seed)
    order = list(range(len(components)))
    rng.shuffle(order)

    val_indices: set[int] = set()
    val_count = 0
    covered_cats: set[str] = set()

    def _can_add(comp_idx: int) -> bool:
        return (total_count - (val_count + comp_sizes[comp_idx])) >= 1

    # Pass 1: prioritize covering distinct emotion categories without exhausting train.
    while val_count < target_val_count or not val_indices:
        best_idx: Optional[int] = None
        best_gain = 0
        for idx in order:
            if idx in val_indices or not _can_add(idx):
                continue
            gain = len(comp_cats[idx] - covered_cats)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        val_indices.add(best_idx)
        val_count += comp_sizes[best_idx]
        covered_cats.update(comp_cats[best_idx])

    # Pass 2: approach target_val_count with remaining shuffled components.
    for idx in order:
        if idx in val_indices or not _can_add(idx):
            continue
        new_count = val_count + comp_sizes[idx]
        if val_count == 0 or abs(new_count - target_val_count) <= abs(val_count - target_val_count):
            val_indices.add(idx)
            val_count = new_count
            if val_count >= target_val_count:
                break

    # Fallback: ensure at least 1 val component when len(components) >= 2 and val_ratio > 0.
    if not val_indices:
        smallest_idx = min(
            (idx for idx in order if _can_add(idx)),
            key=lambda idx: comp_sizes[idx],
            default=None,
        )
        if smallest_idx is not None:
            val_indices.add(smallest_idx)

    val_pos_set = {pos for idx in val_indices for pos in components[idx]}
    train_recs = [rec for pos, rec in enumerate(sorted_usable) if pos not in val_pos_set]
    val_recs = [rec for pos, rec in enumerate(sorted_usable) if pos in val_pos_set]
    return train_recs, val_recs


def select_reference_prompts(
    records: List[Dict[str, Any]],
    top_candidates_per_cat: int = 8,
    active_per_cat: int = 3,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Build two-stage reference candidates and active reference prompts per category."""
    sorted_records = sorted(records, key=_record_sort_key)

    neighbors: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    for idx, rec in enumerate(sorted_records):
        src = str(rec.get("source_file") or "")
        prev_fn: Optional[str] = None
        next_fn: Optional[str] = None
        if idx > 0 and str(sorted_records[idx - 1].get("source_file") or "") == src:
            prev_fn = sorted_records[idx - 1].get("filename")
        if idx + 1 < len(sorted_records) and str(sorted_records[idx + 1].get("source_file") or "") == src:
            next_fn = sorted_records[idx + 1].get("filename")
        neighbors[id(rec)] = (prev_fn, next_fn)

    grouped: Dict[str, List[Tuple[Tuple[int, float, int], Dict[str, Any]]]] = {}
    for rec in sorted_records:
        qc = rec.get("qc")
        if not isinstance(qc, dict) or qc.get("usable_for_reference") is not True:
            continue

        eff_cat, eff_delivery, human_verified = resolve_effective_emotion(rec)
        if eff_cat == "Uncertain":
            continue

        asr_diag = rec.get("asr_diagnostics") if isinstance(rec.get("asr_diagnostics"), dict) else {}
        vocal = rec.get("vocal_expression") if isinstance(rec.get("vocal_expression"), dict) else {}
        auto_tags = vocal.get("auto_tags") if isinstance(vocal.get("auto_tags"), dict) else {}
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}

        avg_logprob = _safe_float(asr_diag.get("avg_logprob"), -99.0)
        confidence = _safe_float(auto_tags.get("confidence"), 0.0)
        duration_sec = _safe_float(rec.get("duration_sec"), 0.0)
        numeric_score = round(avg_logprob, 4) + 0.5 * confidence - 0.08 * abs(duration_sec - 5.2)

        prev_fn, next_fn = neighbors.get(id(rec), (None, None))
        candidate_obj: Dict[str, Any] = {
            "filename": rec.get("filename", ""),
            "audio_sha256": str(prov.get("audio_sha256") or ""),
            "relative_path": rec.get("relative_path", ""),
            "absolute_path": rec.get("absolute_path", ""),
            "source_file": rec.get("source_file", ""),
            "start_time_sec": rec.get("start_time_sec"),
            "end_time_sec": rec.get("end_time_sec"),
            "duration_sec": rec.get("duration_sec"),
            "canonical_text": _extract_canonical_text(rec),
            "effective_category": eff_cat,
            "effective_delivery": eff_delivery,
            "human_verified": human_verified,
            "prev_slice_filename": prev_fn,
            "next_slice_filename": next_fn,
            "score": round(numeric_score, 6),
            "human_review": _copy_human_review(rec.get("human_review")),
        }

        sort_key = (1 if human_verified else 0, numeric_score, -_safe_int(rec.get("id"), 0))
        grouped.setdefault(eff_cat, []).append((sort_key, candidate_obj))

    candidates: Dict[str, List[Dict[str, Any]]] = {}
    active: Dict[str, List[Dict[str, Any]]] = {}
    for cat in sorted(grouped.keys()):
        ranked = [item[1] for item in sorted(grouped[cat], key=lambda pair: pair[0], reverse=True)]
        candidates[cat] = ranked[: max(0, int(top_candidates_per_cat))]
        active[cat] = copy.deepcopy(ranked[: max(0, int(active_per_cat))])

    return candidates, active


def _iter_curation_entries(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield candidate dicts from either {category: [candidate, ...]} or [candidate, ...]."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for group in payload.values():
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict):
                        yield item


def apply_human_curation_file(
    records: List[Dict[str, Any]],
    curation_path: Path,
) -> int:
    """Apply human-reviewed curation entries back onto matching records in-place."""
    payload = json.loads(Path(curation_path).read_text(encoding="utf-8"))
    by_filename: Dict[str, Dict[str, Any]] = {
        str(rec["filename"]): rec for rec in records if rec.get("filename")
    }

    updated_filenames: set[str] = set()
    for entry in _iter_curation_entries(payload):
        human_review = entry.get("human_review")
        if not isinstance(human_review, dict) or human_review.get("reviewed") is not True:
            continue

        filename = str(entry.get("filename") or "")
        record = by_filename.get(filename)
        if record is None:
            continue

        entry_sha = entry.get("audio_sha256")
        prov = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        record_sha = prov.get("audio_sha256")
        if entry_sha and entry_sha != record_sha:
            continue

        record["human_review"] = _copy_human_review(human_review)
        record["qc"] = evaluate_slice_qc(record)
        finalize_record_v2(record)
        updated_filenames.add(filename)

    return len(updated_filenames)


def _scalarize_csv_value(value: Any) -> Any:
    """Convert a value to a CSV-safe scalar without nested JSON structures."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return value


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Schema v2.0 record into scalar-only CSV columns."""
    qc = record.get("qc") if isinstance(record.get("qc"), dict) else {}
    vocal = record.get("vocal_expression") if isinstance(record.get("vocal_expression"), dict) else {}
    auto_tags = vocal.get("auto_tags") if isinstance(vocal.get("auto_tags"), dict) else {}
    human_review = record.get("human_review") if isinstance(record.get("human_review"), dict) else {}
    prosody = record.get("prosody") if isinstance(record.get("prosody"), dict) else {}
    bins = prosody.get("bins") if isinstance(prosody.get("bins"), dict) else {}
    asr_diag = record.get("asr_diagnostics") if isinstance(record.get("asr_diagnostics"), dict) else {}
    prov = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}

    eff_cat, eff_delivery, human_verified = resolve_effective_emotion(record)

    flat: Dict[str, Any] = {
        "id": record.get("id"),
        "filename": record.get("filename"),
        "relative_path": record.get("relative_path"),
        "absolute_path": record.get("absolute_path"),
        "source_file": record.get("source_file"),
        "start_time_sec": record.get("start_time_sec"),
        "end_time_sec": record.get("end_time_sec"),
        "slicer_method": record.get("slicer_method"),
        "duration_sec": record.get("duration_sec"),
        "status": record.get("status"),
        "clean_boundary": record.get("clean_boundary"),
        "raw_rms_db": record.get("raw_rms_db"),
        "raw_peak_db": record.get("raw_peak_db"),
        "applied_gain_db": record.get("applied_gain_db"),
        "text": record.get("text"),
        "chars_per_sec": record.get("chars_per_sec"),
        "category": record.get("category"),
        "qc_tier": qc.get("tier"),
        "usable_for_training": qc.get("usable_for_training"),
        "usable_for_reference": qc.get("usable_for_reference"),
        "delivery_mode": eff_delivery,
        "effective_category": eff_cat,
        "auto_confidence": auto_tags.get("confidence"),
        "human_reviewed": human_verified,
        "verified_primary": human_review.get("verified_primary"),
        "verified_delivery": human_review.get("verified_delivery"),
        "f0_median_hz": prosody.get("f0_median_hz"),
        "f0_iqr_hz": prosody.get("f0_iqr_hz"),
        "voiced_fraction": prosody.get("voiced_fraction"),
        "spectral_centroid_hz": prosody.get("spectral_centroid_hz"),
        "energy_std_db": prosody.get("energy_std_db"),
        "pitch_bin": bins.get("pitch"),
        "speed_bin": bins.get("speed"),
        "energy_bin": bins.get("energy"),
        "effective_mora_rate": asr_diag.get("effective_mora_rate"),
        "avg_logprob": asr_diag.get("avg_logprob"),
        "no_speech_prob": asr_diag.get("no_speech_prob"),
        "compression_ratio": asr_diag.get("compression_ratio"),
        "mean_word_prob": asr_diag.get("mean_word_prob"),
        "audio_sha256": prov.get("audio_sha256"),
        "asr_model": prov.get("asr_model"),
    }
    return {col: _scalarize_csv_value(flat.get(col)) for col in _CSV_FIELDNAMES}


def _build_style_prompt(record: Dict[str, Any], speaker_lang: str) -> str:
    """Build a natural-language style prompt describing speaker, emotion, delivery, and prosody."""
    lang_code = str(speaker_lang or "").strip()
    lang_label = _LANG_LABELS.get(lang_code.upper(), lang_code or "Unknown")
    eff_cat, eff_delivery, _ = resolve_effective_emotion(record)
    prosody = record.get("prosody") if isinstance(record.get("prosody"), dict) else {}
    bins = prosody.get("bins") if isinstance(prosody.get("bins"), dict) else {}
    pitch = str(bins.get("pitch") or "normal")
    speed = str(bins.get("speed") or "normal")
    energy = str(bins.get("energy") or "normal")
    return (
        f"{lang_label} speaker, {eff_cat} tone, {eff_delivery} delivery, "
        f"{pitch} pitch, {speed} speed, {energy} energy."
    )


def _write_list_file(
    path: Path,
    records: Iterable[Dict[str, Any]],
    speaker_name: str,
    speaker_lang: str,
) -> None:
    """Write a 4-column pipe-delimited dataset .list file."""
    clean_speaker = sanitize_list_field(speaker_name)
    clean_lang = sanitize_list_field(speaker_lang)
    lines: List[str] = []
    for rec in records:
        clean_text = sanitize_list_field(_extract_canonical_text(rec))
        if not clean_text:
            continue
        raw_path = rec.get("absolute_path") or rec.get("relative_path") or rec.get("filename") or ""
        clean_path = sanitize_list_field(str(raw_path))
        lines.append(f"{clean_path}|{clean_speaker}|{clean_lang}|{clean_text}")

    content = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(content, encoding="utf-8")


def export_enriched_dataset(
    records: List[Dict[str, Any]],
    output_dir: Path,
    speaker_name: str,
    speaker_lang: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Export Schema v2.0 metadata, splits, manifest.jsonl, and reference prompt files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg_cfg = _extract_packager_cfg(cfg)

    for rec in records:
        finalize_record_v2(rec)

    metadata_json_path = out_dir / "metadata.json"
    metadata_json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata_csv_path = out_dir / "metadata.csv"
    with metadata_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for rec in records:
            writer.writerow(_flatten_record_for_csv(rec))

    usable_records = [
        rec
        for rec in records
        if isinstance(rec.get("qc"), dict) and rec["qc"].get("usable_for_training") is True
    ]
    gold_records = [
        rec for rec in usable_records if rec["qc"].get("tier") == "GOLD"
    ]

    dataset_list_path = out_dir / "dataset.list"
    _write_list_file(dataset_list_path, usable_records, speaker_name, speaker_lang)

    dataset_gold_list_path = out_dir / "dataset_gold.list"
    _write_list_file(dataset_gold_list_path, gold_records, speaker_name, speaker_lang)

    train_recs, val_recs = build_leakage_safe_splits(
        records,
        val_ratio=float(pkg_cfg["val_ratio"]),
        block_size=int(pkg_cfg["block_size"]),
        seed=int(pkg_cfg["seed"]),
    )
    train_list_path = out_dir / "train.list"
    val_list_path = out_dir / "val.list"
    _write_list_file(train_list_path, train_recs, speaker_name, speaker_lang)
    _write_list_file(val_list_path, val_recs, speaker_name, speaker_lang)

    manifest_jsonl_path = out_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in usable_records:
            eff_cat, eff_delivery, _ = resolve_effective_emotion(rec)
            entry = {
                **rec,
                "speaker": speaker_name,
                "language": speaker_lang,
                "effective_category": eff_cat,
                "effective_delivery": eff_delivery,
                "style_prompt": _build_style_prompt(rec, speaker_lang),
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    candidates, active = select_reference_prompts(
        records,
        top_candidates_per_cat=int(pkg_cfg["top_candidates_per_cat"]),
        active_per_cat=int(pkg_cfg["active_per_cat"]),
    )
    ref_candidates_path = out_dir / "reference_candidates.json"
    ref_prompts_path = out_dir / "reference_prompts.json"
    ref_candidates_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ref_prompts_path.write_text(
        json.dumps(active, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "metadata_json": metadata_json_path,
        "metadata_csv": metadata_csv_path,
        "dataset_list": dataset_list_path,
        "dataset_gold_list": dataset_gold_list_path,
        "train_list": train_list_path,
        "val_list": val_list_path,
        "manifest_jsonl": manifest_jsonl_path,
        "reference_candidates": ref_candidates_path,
        "reference_prompts": ref_prompts_path,
    }
