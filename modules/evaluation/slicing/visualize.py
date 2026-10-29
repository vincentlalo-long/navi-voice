"""
Visualization Engine for Voice Benchmark Suite.
Generates multi-panel static PNG comparison figures (via Matplotlib)
providing both high-level executive summaries and deep acoustic/duration distributions.
"""

from typing import Any, Dict, List
from pathlib import Path


def plot_benchmark_comparison_png(
    results: Dict[str, Any],
    output_png_path: Path,
    title: str = "Amadeus Voice Slicer Benchmark - Executive Summary",
) -> None:
    """
    Renders a 2x2 multi-panel comparison figure visualizing:
    1. Boundary Quality (CBR % vs WARN %)
    2. Segmentation Efficiency (Speech Coverage % vs Slices Count)
    3. Utterance Duration Profiles (Mean, Median, P95)
    4. Execution Speed (RTF - Real-Time Factor on log scale)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    algorithms = [name.upper() for name in results.keys()]
    if not algorithms:
        return

    n_algos = len(algorithms)
    x = np.arange(n_algos)
    bar_width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    # 1. Boundary Quality (CBR % vs WARN %)
    ax1 = axes[0, 0]
    cbr_vals = [results[k]["boundary"]["cbr_pct"] for k in results]
    warn_vals = [results[k]["boundary"]["warn_pct"] for k in results]

    bars_cbr = ax1.bar(x - bar_width / 2, cbr_vals, bar_width, label="CBR % (Clean+Safe)", color="#10b981", alpha=0.85)
    bars_warn = ax1.bar(x + bar_width / 2, warn_vals, bar_width, label="WARN % (Speech Clip Risk)", color="#ef4444", alpha=0.85)

    ax1.set_title("1. Acoustic Boundary Quality", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)

    for bar in bars_cbr:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_warn:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#b91c1c")

    # 2. Segmentation Efficiency (Coverage % & Slice Count)
    ax2 = axes[0, 1]
    coverage_vals = [results[k]["segmentation"]["speech_coverage_pct"] for k in results]
    slice_counts = [results[k]["segmentation"]["total_slices"] for k in results]

    bars_cov = ax2.bar(x, coverage_vals, width=0.5, color="#3b82f6", alpha=0.85, label="Speech Coverage %")
    ax2.set_title("2. Speech Coverage & Dialogue Slices", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Preserved Audio (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, slice_counts, color="#8b5cf6", marker="o", linewidth=2.5, markersize=8, label="# Total Slices")
    ax2_twin.set_ylabel("Slice Count", color="#8b5cf6", fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#8b5cf6")
    max_count = max(slice_counts) if slice_counts else 10
    ax2_twin.set_ylim(0, max_count * 1.35)

    for bar in bars_cov:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, count in enumerate(slice_counts):
        ax2_twin.annotate(f"{count} slices", xy=(x[i], count),
                          xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                          fontsize=9, fontweight="bold", color="#6d28d9")

    # 3. Duration Profiles (Mean, Median, P95)
    ax3 = axes[1, 0]
    mean_durs = [results[k]["segmentation"]["mean_duration_sec"] for k in results]
    median_durs = [results[k]["segmentation"]["median_duration_sec"] for k in results]
    p95_durs = [results[k]["segmentation"]["p95_duration_sec"] for k in results]

    w3 = 0.25
    b_mean = ax3.bar(x - w3, mean_durs, w3, label="Mean Dur (s)", color="#6366f1", alpha=0.85)
    b_med = ax3.bar(x, median_durs, w3, label="Median Dur (s)", color="#06b6d4", alpha=0.85)
    b_p95 = ax3.bar(x + w3, p95_durs, w3, label="P95 Dur (s)", color="#f97316", alpha=0.85)

    ax3.set_title("3. Slice Duration Profile (Target: 2s - 12s)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Duration (seconds)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, fontweight="bold")
    max_dur = max(p95_durs + mean_durs) if (p95_durs or mean_durs) else 10.0
    ax3.set_ylim(0, max(max_dur * 1.25, 12.5))
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.9)

    for bar in b_mean:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in b_med:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in b_p95:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 4. Execution Speed (RTF)
    ax4 = axes[1, 1]
    rtf_vals = [results[k]["rtf"] for k in results]
    bar_colors = [palette[i % len(palette)] for i in range(n_algos)]

    bars_rtf = ax4.bar(x, rtf_vals, width=0.45, color=bar_colors, alpha=0.85)
    ax4.set_title("4. Real-Time Factor (RTF) - Lower is Faster", fontsize=12, fontweight="bold", pad=10)
    ax4.set_ylabel("RTF (Proc Sec / Audio Sec)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, fontweight="bold")
    max_rtf = max(rtf_vals) if rtf_vals else 0.5
    ax4.set_ylim(0, max(max_rtf * 1.35, 1.15))
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.axhline(1.0, color="#ef4444", linestyle=":", linewidth=1.5, label="Realtime Barrier (RTF=1.0)")
    ax4.legend(loc="upper right", framealpha=0.9)

    for bar in bars_rtf:
        h = bar.get_height()
        ax4.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), dpi=200)
    plt.close()


def plot_boundary_acoustic_detail_png(
    results: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """
    Renders an in-depth acoustic boundary inspection figure:
    1. Boxplot distribution of Edge Onset & Offset RMS dB.
    2. Stacked bar chart: % CLEAN vs % SAFE vs % WARN.
    3. Maximum transient step jump amplitude |x[0]|.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    algorithms = list(results.keys())
    if not algorithms:
        return

    n_algos = len(algorithms)
    x = np.arange(n_algos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Acoustic Boundary Quality & Phoneme Clipping Analysis", fontsize=15, fontweight="bold", y=0.98)

    # Subplot 1: Distribution of Edge RMS (Onset & Offset)
    ax1 = axes[0]
    box_data = []
    box_labels = []
    for k in algorithms:
        # Combine start and end rms list if available, else fallback to avg
        start_list = results[k].get("start_rms_list", [])
        end_list = results[k].get("end_rms_list", [])
        combined = [v for v in (start_list + end_list) if v > -100.0]
        if not combined:
            combined = [results[k]["boundary"].get("mean_edge_rms_db", -99.0)]
        box_data.append(combined)
        box_labels.append(k.upper())

    bp = ax1.boxplot(box_data, tick_labels=box_labels, patch_artist=True, notch=False)
    for box in bp["boxes"]:
        box.set(facecolor="#93c5fd", color="#1e40af", linewidth=1.5)
    for median in bp["medians"]:
        median.set(color="#b91c1c", linewidth=2.0)

    ax1.axhline(-35.0, color="#10b981", linestyle="--", linewidth=1.5, label="Clean Silence (<= -35 dB)")
    ax1.axhline(-27.0, color="#f59e0b", linestyle="--", linewidth=1.5, label="Safe Margin (<= -27 dB)")
    ax1.set_title("Edge RMS Energy Distribution (dBFS)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMS (dBFS) - Lower is Quieter")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # Subplot 2: Stacked Boundary Classification (Clean / Safe / Warn)
    ax2 = axes[1]
    clean_pcts = np.array([results[k]["boundary"]["clean_pct"] for k in algorithms])
    safe_pcts = np.array([results[k]["boundary"]["safe_pct"] for k in algorithms])
    warn_pcts = np.array([results[k]["boundary"]["warn_pct"] for k in algorithms])

    p1 = ax2.bar(x, clean_pcts, width=0.45, label="CLEAN (Silence Edge)", color="#10b981", alpha=0.85)
    p2 = ax2.bar(x, safe_pcts, width=0.45, bottom=clean_pcts, label="SAFE (Margin Buffer)", color="#f59e0b", alpha=0.85)
    p3 = ax2.bar(x, warn_pcts, width=0.45, bottom=clean_pcts + safe_pcts, label="WARN (Clipping Risk)", color="#ef4444", alpha=0.85)

    ax2.set_title("Boundary Classification Breakdown (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.upper() for k in algorithms], fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Annotate WARN% if > 0
    for i, w in enumerate(warn_pcts):
        if w > 0:
            ax2.annotate(f"WARN: {w:.1f}%", xy=(x[i], 100), xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, fontweight="bold", color="#b91c1c")

    # Subplot 3: Max Transient Step Jump |x[0]| (Click Gibbs Artifacts)
    ax3 = axes[2]
    step_jumps = [results[k]["boundary"].get("max_step_jump", 0.0) for k in algorithms]
    colors = ["#10b981" if s <= 0.05 else "#ef4444" for s in step_jumps]

    bars_jump = ax3.bar(x, step_jumps, width=0.45, color=colors, alpha=0.85)
    ax3.axhline(0.05, color="#ef4444", linestyle=":", linewidth=1.5, label="Click Audible Threshold (|x| > 0.05)")
    ax3.set_title("Peak Transient Discontinuity (|x[0]|)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sample Amplitude [0.0 - 1.0]")
    ax3.set_xticks(x)
    ax3.set_xticklabels([k.upper() for k in algorithms], fontweight="bold")
    max_jump = max(step_jumps) if step_jumps else 0.05
    ax3.set_ylim(0, max(max_jump * 1.3, 0.08))
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for bar in bars_jump:
        h = bar.get_height()
        ax3.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), dpi=200)
    plt.close()


def plot_duration_distribution_png(
    results: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """
    Renders an in-depth utterance duration profile figure:
    1. Boxplot distribution of slice durations comparing against the TTS golden window [2s, 12s].
    2. Over-segmentation (<2s) and Under-segmentation (>10s) rate comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    algorithms = list(results.keys())
    if not algorithms:
        return

    n_algos = len(algorithms)
    x = np.arange(n_algos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Utterance Duration & Segmentation Distribution Analysis", fontsize=15, fontweight="bold", y=0.98)

    # Subplot 1: Boxplot of Slice Durations
    ax1 = axes[0]
    box_data = []
    box_labels = []
    under_2s_rates = []
    over_10s_rates = []

    for k in algorithms:
        durs = results[k].get("durations", [])
        if not durs:
            # Fallback synth
            durs = [results[k]["segmentation"]["median_duration_sec"]]
        box_data.append(durs)
        box_labels.append(k.upper())

        total = len(durs)
        u2 = sum(1 for d in durs if d < 2.0) / total * 100 if total > 0 else 0.0
        o10 = sum(1 for d in durs if d > 10.0) / total * 100 if total > 0 else 0.0
        under_2s_rates.append(u2)
        over_10s_rates.append(o10)

    # Shaded Golden Zone [2s - 12s]
    ax1.axhspan(2.0, 12.0, color="#dcfce7", alpha=0.6, label="TTS Optimal Range (2s - 12s)")
    ax1.axhline(2.0, color="#16a34a", linestyle="--", linewidth=1.2)
    ax1.axhline(12.0, color="#dc2626", linestyle="--", linewidth=1.2)

    bp = ax1.boxplot(box_data, tick_labels=box_labels, patch_artist=True, notch=False)
    for box in bp["boxes"]:
        box.set(facecolor="#fed7aa", color="#c2410c", linewidth=1.5)
    for median in bp["medians"]:
        median.set(color="#1d4ed8", linewidth=2.0)

    ax1.set_title("Slice Duration Spread vs TTS Golden Range", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Duration (seconds)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Subplot 2: Over-segmentation (<2s) vs Under-segmentation (>10s)
    ax2 = axes[1]
    bw = 0.35
    b_u2 = ax2.bar(x - bw / 2, under_2s_rates, bw, label="Fragmented Slices (< 2s)", color="#f59e0b", alpha=0.85)
    b_o10 = ax2.bar(x + bw / 2, over_10s_rates, bw, label="Long Slices (> 10s)", color="#8b5cf6", alpha=0.85)

    ax2.set_title("Fragmentation vs Run-on Utterance Rates (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Percentage of Total Slices (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.upper() for k in algorithms], fontweight="bold")
    max_rate = max(under_2s_rates + over_10s_rates) if (under_2s_rates or over_10s_rates) else 10.0
    ax2.set_ylim(0, max(max_rate * 1.35, 20.0))
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for bar in b_u2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in b_o10:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), dpi=200)
    plt.close()

