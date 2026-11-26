"""Phase 3 enrichment audit report generator (JSON summary, Markdown, PNG plots)."""

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.enrichment.schema import SCHEMA_VERSION
from modules.tagging.classifier import TAGGER_VERSION
from modules.tagging.prosody import PROSODY_EXTRACTOR_VERSION

from .metrics import compute_enrichment_metrics
from .visualize import (
    plot_category_distribution_png,
    plot_prosody_acoustic_space_png,
    plot_qc_waterfall_png,
)


def _package_version(name: str) -> str:
    """Return installed distribution version without failing report generation."""
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not-installed"


def _build_provenance_meta(run_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build deterministic provenance metadata block for audit artifacts."""
    extra = dict(run_meta) if isinstance(run_meta, dict) else {}
    processed_dir = str(extra.pop("processed_dir", "unknown"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed_dir": processed_dir,
        "schema_version": SCHEMA_VERSION,
        "prosody_extractor_version": PROSODY_EXTRACTOR_VERSION,
        "tagger_version": TAGGER_VERSION,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "soundfile": _package_version("soundfile"),
            "matplotlib": _package_version("matplotlib"),
            "faster-whisper": _package_version("faster-whisper"),
            "torch": _package_version("torch"),
        },
        **extra,
    }


def _render_markdown_report(
    metrics: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    """Format enrichment metrics and provenance metadata into Markdown tables."""
    wf = metrics["waterfall"]
    total = max(1, int(wf["total_slices"]))

    lines: List[str] = [
        "# Phase 3 Enrichment & QC Audit Report",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| `generated_at_utc` | `{meta['generated_at_utc']}` |",
        f"| `processed_dir` | `{meta['processed_dir']}` |",
        f"| `schema_version` | `{meta['schema_version']}` |",
        f"| `prosody_extractor_version` | `{meta['prosody_extractor_version']}` |",
        f"| `tagger_version` | `{meta['tagger_version']}` |",
        f"| `python` | `{meta['python']}` |",
    ]

    pkg_str = ", ".join(f"{k}={v}" for k, v in meta["versions"].items())
    lines.append(f"| `packages` | `{pkg_str}` |")
    for opt_key in ("speaker", "language", "profile", "curation_file", "curation_updated_count"):
        if opt_key in meta and meta[opt_key] is not None:
            lines.append(f"| `{opt_key}` | `{meta[opt_key]}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 1. QC Funnel Waterfall",
        "",
        f"- **Total Duration:** {wf['total_duration_sec']:.2f}s | **Usable Training Duration:** {wf['usable_duration_sec']:.2f}s | **GOLD Duration:** {wf['gold_duration_sec']:.2f}s",
        f"- **Retention Rate:** {wf['retention_pct']:.2f}% usable | {wf['gold_pct']:.2f}% GOLD",
        "",
        "| Stage | Count | % of Total |",
        "| :--- | ---: | ---: |",
    ])

    stage_rows = [
        ("Total Slices", wf["total_slices"]),
        ("Acoustic Safe", wf["acoustic_safe"]),
        ("Non-Empty ASR", wf["non_empty_asr"]),
        ("Anti-Hallucination Gates Pass", wf["anti_hallucination_pass"]),
        ("Usable for Training (GOLD + SILVER)", wf["usable_for_training"]),
        ("GOLD Tier", wf["gold_count"]),
        ("SILVER Tier", wf["silver_count"]),
        ("REJECT Tier", wf["reject_count"]),
        ("Pending ASR", wf["pending_asr_count"]),
        ("Reference Eligible", wf["reference_eligible_count"]),
        ("Human Reviewed", wf["human_reviewed_count"]),
    ]
    for label, count in stage_rows:
        pct = (count / total) * 100.0 if wf["total_slices"] > 0 else 0.0
        lines.append(f"| {label} | {count:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "![QC Waterfall](qc_waterfall.png)",
        "",
        "---",
        "",
        "## 2. Rejection Breakdown",
        "",
    ])

    rejections = metrics["rejection_reasons"]
    if rejections:
        rej_denom = max(1, int(wf["reject_count"]))
        lines.extend([
            "| Reason Code | Count | % of Rejected Slices |",
            "| :--- | ---: | ---: |",
        ])
        for reason, cnt in rejections.items():
            lines.append(f"| `{reason}` | {cnt:,} | {(cnt / rej_denom) * 100.0:.2f}% |")
    else:
        lines.append("No slices were rejected.")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Emotion Category & Duration Distribution",
        "",
    ])

    categories = metrics["categories"]
    if categories:
        lines.extend([
            "| Category | Total | GOLD | SILVER | Duration (s) | Share % |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for cat, stats in categories.items():
            lines.append(
                f"| `{cat}` | {stats['count']:,} | {stats['gold_count']:,} | "
                f"{stats['silver_count']:,} | {stats['duration_sec']:.2f} | {stats['share_pct']:.2f}% |"
            )
    else:
        lines.append("No categorized usable slices available.")

    lines.extend([
        "",
        "![Category Distribution](category_distribution.png)",
        "",
        "---",
        "",
        "## 4. Delivery Modes & Tagger Abstention",
        "",
    ])

    ab = metrics["abstention"]
    lines.append(
        f"- **Abstention Rate (Analysis Scope):** {ab['abstention_rate'] * 100.0:.2f}% "
        f"({ab['abstained_count']} / {ab['tagged_count']} tagged slices)"
    )
    lines.append(
        f"- **Abstention Rate (All Tagged Slices):** {ab['all_abstention_rate'] * 100.0:.2f}% "
        f"({ab['all_abstained_count']} / {ab['all_tagged_count']} tagged slices)"
    )
    lines.append("")

    delivery_modes = metrics["delivery_modes"]
    if delivery_modes:
        lines.extend([
            "| Delivery Mode | Count | Duration (s) | Share % |",
            "| :--- | ---: | ---: | ---: |",
        ])
        for mode, stats in delivery_modes.items():
            lines.append(
                f"| `{mode}` | {stats['count']:,} | {stats['duration_sec']:.2f} | {stats['share_pct']:.2f}% |"
            )
    else:
        lines.append("No delivery mode entries available.")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Prosody & ASR Acoustic Telemetry",
        "",
    ])

    prosody = metrics["prosody_summary"]
    lines.append(
        f"- **Scope:** `{prosody['scope']}` | **Voiced Slices ($\\ge 0.15$):** {prosody['voiced_count']} | "
        f"**Unvoiced Slices:** {prosody['unvoiced_count']}"
    )
    lines.extend([
        "",
        "| Feature | Count | Mean | Median | P20 | P80 | Min | Max |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for feature_key in (
        "f0_median_hz",
        "f0_iqr_hz",
        "voiced_fraction",
        "spectral_centroid_hz",
        "spectral_flatness",
        "energy_std_db",
        "crest_factor_db",
        "raw_rms_db",
        "duration_sec",
        "effective_cps",
        "effective_mora_rate",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
    ):
        stat = prosody.get(feature_key, {})
        lines.append(
            f"| `{feature_key}` | {stat.get('count', 0)} | {stat.get('mean', 0.0):.3f} | "
            f"{stat.get('median', 0.0):.3f} | {stat.get('p20', 0.0):.3f} | "
            f"{stat.get('p80', 0.0):.3f} | {stat.get('min', 0.0):.3f} | {stat.get('max', 0.0):.3f} |"
        )

    lines.extend([
        "",
        "![Prosody Acoustic Space](prosody_acoustic_space.png)",
        "",
    ])
    return "\n".join(lines)


def generate_enrichment_report(
    records: List[Dict[str, Any]],
    output_dir: Path,
    run_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Compute enrichment metrics, render PNG figures, and write JSON/Markdown reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_enrichment_metrics(records)
    meta = _build_provenance_meta(run_meta)

    waterfall_png = out_dir / "qc_waterfall.png"
    prosody_png = out_dir / "prosody_acoustic_space.png"
    category_png = out_dir / "category_distribution.png"

    plot_qc_waterfall_png(metrics, waterfall_png)
    plot_prosody_acoustic_space_png(records, prosody_png)
    plot_category_distribution_png(metrics, category_png)

    summary_json = out_dir / "enrichment_summary.json"
    summary_payload = {"_metadata": meta, **metrics}
    summary_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_md = out_dir / "enrichment_report.md"
    report_md.write_text(
        _render_markdown_report(metrics, meta),
        encoding="utf-8",
    )

    return {
        "enrichment_summary_json": summary_json,
        "enrichment_report_md": report_md,
        "qc_waterfall_png": waterfall_png,
        "prosody_acoustic_space_png": prosody_png,
        "category_distribution_png": category_png,
    }
