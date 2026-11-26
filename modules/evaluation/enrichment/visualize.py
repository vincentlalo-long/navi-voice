"""Matplotlib (Agg) diagnostic visualizations for Phase 3 enrichment audits."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.enrichment.schema import resolve_effective_emotion


def _get_pyplot():
    """Import matplotlib pyplot using the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _render_empty_placeholder(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="#64748b",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def plot_qc_waterfall_png(
    metrics: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """Render the QC waterfall funnel and top rejection reasons to a PNG figure."""
    plt = _get_pyplot()
    import numpy as np

    wf = metrics.get("waterfall", {})
    rejections = metrics.get("rejection_reasons", {})

    stages: List[Tuple[str, int, str]] = [
        ("Total Slices", int(wf.get("total_slices", 0)), "#475569"),
        ("Acoustic Safe", int(wf.get("acoustic_safe", 0)), "#3b82f6"),
        ("Non-Empty ASR", int(wf.get("non_empty_asr", 0)), "#0ea5e9"),
        ("ASR Gates Pass", int(wf.get("anti_hallucination_pass", 0)), "#06b6d4"),
        ("Usable (GOLD+SILVER)", int(wf.get("usable_for_training", 0)), "#10b981"),
        ("GOLD Tier", int(wf.get("gold_count", 0)), "#16a34a"),
        ("Reference Eligible", int(wf.get("reference_eligible_count", 0)), "#8b5cf6"),
        ("Human Reviewed", int(wf.get("human_reviewed_count", 0)), "#f59e0b"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Phase 3 QC Funnel & Rejection Diagnostics", fontsize=14, fontweight="bold", y=0.98)

    try:
        ax1, ax2 = axes
        total = max(1, int(wf.get("total_slices", 0)))
        if int(wf.get("total_slices", 0)) == 0:
            ax1.set_title("QC Pipeline Waterfall", fontsize=11, fontweight="bold")
            _render_empty_placeholder(ax1, "No slices available")
        else:
            labels = [s[0] for s in stages][::-1]
            counts = [s[1] for s in stages][::-1]
            colors = [s[2] for s in stages][::-1]
            y_pos = np.arange(len(labels))

            bars = ax1.barh(y_pos, counts, color=colors, alpha=0.88, height=0.62)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=9, fontweight="bold")
            ax1.set_xlim(0, max(counts + [1]) * 1.24)
            ax1.set_xlabel("Utterance Count")
            ax1.set_title("QC Pipeline Waterfall", fontsize=11, fontweight="bold")
            ax1.grid(axis="x", linestyle="--", alpha=0.4)

            for bar, cnt in zip(bars, counts):
                pct = (cnt / total) * 100.0
                ax1.annotate(
                    f"{cnt:,} ({pct:.1f}%)",
                    xy=(cnt, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax2.set_title("Rejection Reason Breakdown", fontsize=11, fontweight="bold")
        if not rejections:
            _render_empty_placeholder(ax2, "No rejected slices")
        else:
            top_items = list(rejections.items())[:10][::-1]
            r_labels = [k for k, _ in top_items]
            r_counts = [int(v) for _, v in top_items]
            ry = np.arange(len(r_labels))

            rbars = ax2.barh(ry, r_counts, color="#ef4444", alpha=0.85, height=0.6)
            ax2.set_yticks(ry)
            ax2.set_yticklabels(r_labels, fontsize=8.5)
            ax2.set_xlim(0, max(r_counts + [1]) * 1.22)
            ax2.set_xlabel("Rejected Slice Count")
            ax2.grid(axis="x", linestyle="--", alpha=0.4)

            for bar, cnt in zip(rbars, r_counts):
                ax2.annotate(
                    f"{cnt:,}",
                    xy=(cnt, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#991b1b",
                )

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_png_path), dpi=180)
    finally:
        plt.close(fig)


def plot_prosody_acoustic_space_png(
    records: List[Dict[str, Any]],
    output_png_path: Path,
) -> None:
    """Render a 4-panel acoustic, prosody, speaking-rate, and ASR-gate diagnostic figure."""
    plt = _get_pyplot()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Prosody Acoustic Space & ASR Gate Telemetry",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    try:
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]

        # Collect voiced records for F0 panels
        f0_by_cat: Dict[str, List[float]] = {}
        voiced_points_by_delivery: Dict[str, Tuple[List[float], List[float]]] = {}
        cps_values: List[float] = []
        asr_points_by_tier: Dict[str, Tuple[List[float], List[float]]] = {}

        for rec in records:
            eff_cat, delivery_mode, _ = resolve_effective_emotion(rec)
            prosody = rec.get("prosody") if isinstance(rec.get("prosody"), dict) else {}
            asr_diag = (
                rec.get("asr_diagnostics")
                if isinstance(rec.get("asr_diagnostics"), dict)
                else {}
            )
            qc = rec.get("qc") if isinstance(rec.get("qc"), dict) else {}
            tier = str(qc.get("tier") or "UNSET")

            vf = _safe_float(prosody.get("voiced_fraction"))
            f0 = _safe_float(prosody.get("f0_median_hz"))
            raw_rms = _safe_float(rec.get("raw_rms_db"))
            if raw_rms is None:
                raw_rms = _safe_float(rec.get("rms_db"))

            if vf is not None and vf >= 0.15 and f0 is not None and f0 > 0.0:
                f0_by_cat.setdefault(eff_cat, []).append(f0)
                if raw_rms is not None:
                    xs, ys = voiced_points_by_delivery.setdefault(delivery_mode, ([], []))
                    xs.append(raw_rms)
                    ys.append(f0)

            cps = _safe_float(asr_diag.get("effective_cps"))
            if cps is not None:
                cps_values.append(cps)

            logprob = _safe_float(asr_diag.get("avg_logprob"))
            comp = _safe_float(asr_diag.get("compression_ratio"))
            if logprob is not None and comp is not None and logprob > -50.0:
                xs, ys = asr_points_by_tier.setdefault(tier, ([], []))
                xs.append(logprob)
                ys.append(comp)

        # Panel 1: F0 by category
        ax1.set_title("1. Voiced Median F0 (Hz) by Emotion Category", fontsize=11, fontweight="bold")
        ax1.set_ylabel("F0 Median (Hz)")
        if f0_by_cat:
            sorted_cats = sorted(f0_by_cat.keys(), key=lambda c: (-len(f0_by_cat[c]), c))[:10]
            box_data = [f0_by_cat[c] for c in sorted_cats]
            bp = ax1.boxplot(box_data, tick_labels=sorted_cats, patch_artist=True)
            for box in bp["boxes"]:
                box.set(facecolor="#bfdbfe", color="#1d4ed8", linewidth=1.3)
            for median in bp["medians"]:
                median.set(color="#b91c1c", linewidth=2.0)
            ax1.tick_params(axis="x", labelrotation=20, labelsize=8.5)
            ax1.grid(axis="y", linestyle=":", alpha=0.5)
        else:
            _render_empty_placeholder(ax1, "No voiced F0 data (voiced_fraction >= 0.15)")

        # Panel 2: F0 vs raw_rms_db colored by delivery_mode
        ax2.set_title("2. F0 Median (Hz) vs Pre-Norm Level (raw_rms_db)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("raw_rms_db (dBFS)")
        ax2.set_ylabel("F0 Median (Hz)")
        delivery_colors = {
            "normal": "#2563eb",
            "whisper_breathy": "#8b5cf6",
            "loud_strained": "#ef4444",
        }
        if voiced_points_by_delivery:
            for mode, (xs, ys) in sorted(voiced_points_by_delivery.items()):
                ax2.scatter(
                    xs,
                    ys,
                    label=f"{mode} ({len(xs)})",
                    color=delivery_colors.get(mode, "#64748b"),
                    alpha=0.72,
                    s=26,
                    edgecolors="white",
                    linewidths=0.4,
                )
            ax2.axvline(-32.0, color="#8b5cf6", linestyle="--", linewidth=1.1, alpha=0.7, label="Quiet Ref (-32 dB)")
            ax2.axvline(-14.0, color="#ef4444", linestyle="--", linewidth=1.1, alpha=0.7, label="Loud Ref (-14 dB)")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="best", fontsize=8, framealpha=0.9)
        else:
            _render_empty_placeholder(ax2, "No voiced F0 vs RMS pairs available")

        # Panel 3: effective_cps histogram with [1.2, 18.0] gate lines
        ax3.set_title("3. Effective Speaking Rate (effective_cps) vs QC Gates", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Effective Characters / Sec (speech duration)")
        ax3.set_ylabel("Utterance Count")
        ax3.axvline(1.2, color="#dc2626", linestyle="--", linewidth=1.5, label="Min Silver Gate (1.2 cps)")
        ax3.axvline(18.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Max Silver Gate (18.0 cps)")
        ax3.axvspan(1.2, 18.0, color="#dcfce7", alpha=0.45, label="Valid Rate Band [1.2, 18.0]")
        if cps_values:
            upper_bound = max(20.0, max(cps_values) * 1.1)
            ax3.hist(
                cps_values,
                bins=24,
                range=(0.0, upper_bound),
                color="#0ea5e9",
                alpha=0.8,
                edgecolor="white",
            )
            ax3.grid(axis="y", linestyle=":", alpha=0.5)
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
        else:
            ax3.set_xlim(0.0, 20.0)
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
            _render_empty_placeholder(ax3, "No ASR speaking rate data")

        # Panel 4: avg_logprob vs compression_ratio colored by QC tier
        ax4.set_title("4. ASR Confidence (avg_logprob) vs Compression Ratio", fontsize=11, fontweight="bold")
        ax4.set_xlabel("avg_logprob (higher is better)")
        ax4.set_ylabel("compression_ratio (lower is less repetitive)")
        ax4.axvline(-1.0, color="#ef4444", linestyle="--", linewidth=1.3, label="Silver Min Logprob (-1.0)")
        ax4.axhline(2.4, color="#ef4444", linestyle=":", linewidth=1.3, label="Silver Max Compression (2.4)")
        ax4.axvline(-0.6, color="#16a34a", linestyle="--", linewidth=1.0, alpha=0.7, label="Gold Min Logprob (-0.6)")
        ax4.axhline(2.0, color="#16a34a", linestyle=":", linewidth=1.0, alpha=0.7, label="Gold Max Compression (2.0)")

        tier_styles = {
            "GOLD": ("#16a34a", "o"),
            "SILVER": ("#f59e0b", "s"),
            "REJECT": ("#ef4444", "x"),
            "PENDING_ASR": ("#64748b", "^"),
            "UNSET": ("#64748b", "o"),
        }
        if asr_points_by_tier:
            for tier, (xs, ys) in sorted(asr_points_by_tier.items()):
                color, marker = tier_styles.get(tier, ("#64748b", "o"))
                ax4.scatter(
                    xs,
                    ys,
                    label=f"{tier} ({len(xs)})",
                    color=color,
                    marker=marker,
                    alpha=0.78,
                    s=28,
                )
            ax4.grid(True, linestyle=":", alpha=0.5)
            ax4.legend(loc="best", fontsize=7.5, framealpha=0.9)
        else:
            ax4.set_xlim(-2.0, 0.1)
            ax4.set_ylim(0.5, 3.0)
            ax4.legend(loc="best", fontsize=7.5, framealpha=0.9)
            _render_empty_placeholder(ax4, "No ASR confidence points available")

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_png_path), dpi=180)
    finally:
        plt.close(fig)


def plot_category_distribution_png(
    metrics: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """Render stacked category counts (GOLD/SILVER) and delivery mode breakdown."""
    plt = _get_pyplot()
    import numpy as np

    categories = metrics.get("categories", {})
    delivery_modes = metrics.get("delivery_modes", {})
    abstention = metrics.get("abstention", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Emotion Category & Delivery Mode Distribution", fontsize=14, fontweight="bold", y=0.98)

    try:
        ax1, ax2 = axes
        ax1.set_title("Category Breakdown (GOLD vs SILVER)", fontsize=11, fontweight="bold")
        if not categories:
            _render_empty_placeholder(ax1, "No category records available")
        else:
            cat_names = list(categories.keys())[::-1]
            gold_vals = np.array([int(categories[c].get("gold_count", 0)) for c in cat_names])
            silver_vals = np.array([int(categories[c].get("silver_count", 0)) for c in cat_names])
            other_vals = np.array([
                max(0, int(categories[c].get("count", 0)) - int(categories[c].get("gold_count", 0)) - int(categories[c].get("silver_count", 0)))
                for c in cat_names
            ])
            totals = gold_vals + silver_vals + other_vals
            y_pos = np.arange(len(cat_names))

            ax1.barh(y_pos, gold_vals, color="#16a34a", alpha=0.88, height=0.6, label="GOLD")
            ax1.barh(y_pos, silver_vals, left=gold_vals, color="#f59e0b", alpha=0.88, height=0.6, label="SILVER")
            if np.any(other_vals > 0):
                ax1.barh(
                    y_pos,
                    other_vals,
                    left=gold_vals + silver_vals,
                    color="#64748b",
                    alpha=0.8,
                    height=0.6,
                    label="Other/Unscored",
                )

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(cat_names, fontsize=9, fontweight="bold")
            ax1.set_xlim(0, max(list(totals) + [1]) * 1.28)
            ax1.set_xlabel("Utterance Count")
            ax1.grid(axis="x", linestyle="--", alpha=0.4)
            ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

            for y, cat, tot in zip(y_pos, cat_names, totals):
                dur = float(categories[cat].get("duration_sec", 0.0))
                ax1.annotate(
                    f"{int(tot)} ({dur:.1f}s)",
                    xy=(tot, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ab_rate_pct = float(abstention.get("abstention_rate", 0.0)) * 100.0
        ax2.set_title(
            f"Delivery Mode Distribution (Abstention: {ab_rate_pct:.1f}%)",
            fontsize=11,
            fontweight="bold",
        )
        if not delivery_modes:
            _render_empty_placeholder(ax2, "No delivery mode data available")
        else:
            mode_names = list(delivery_modes.keys())
            mode_counts = [int(delivery_modes[m].get("count", 0)) for m in mode_names]
            x_pos = np.arange(len(mode_names))
            mode_palette = {
                "normal": "#3b82f6",
                "whisper_breathy": "#8b5cf6",
                "loud_strained": "#ef4444",
            }
            colors = [mode_palette.get(m, "#64748b") for m in mode_names]
            bars = ax2.bar(x_pos, mode_counts, color=colors, alpha=0.88, width=0.5)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(mode_names, fontsize=9.5, fontweight="bold")
            ax2.set_ylim(0, max(mode_counts + [1]) * 1.25)
            ax2.set_ylabel("Utterance Count")
            ax2.grid(axis="y", linestyle="--", alpha=0.4)

            for bar, mode, cnt in zip(bars, mode_names, mode_counts):
                share = float(delivery_modes[mode].get("share_pct", 0.0))
                ax2.annotate(
                    f"{cnt} ({share:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, cnt),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_png_path), dpi=180)
    finally:
        plt.close(fig)
