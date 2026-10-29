"""
Backward compatibility layer for Phase 1 visualization.
Directs imports to modules.evaluation.slicing.visualize.
"""

from modules.evaluation.slicing.visualize import (
    plot_benchmark_comparison_png,
    plot_boundary_acoustic_detail_png,
    plot_duration_distribution_png,
)

__all__ = [
    "plot_benchmark_comparison_png",
    "plot_boundary_acoustic_detail_png",
    "plot_duration_distribution_png",
]
