"""
Audio Slicing & Boundary Evaluation Module.
"""

from .boundary import evaluate_slice_boundary, aggregate_boundary_metrics
from .segmentation import compute_segmentation_integrity
from .benchmark import BenchmarkSuite
from .visualize import (
    plot_benchmark_comparison_png,
    plot_boundary_acoustic_detail_png,
    plot_duration_distribution_png,
)

__all__ = [
    "evaluate_slice_boundary",
    "aggregate_boundary_metrics",
    "compute_segmentation_integrity",
    "BenchmarkSuite",
    "plot_benchmark_comparison_png",
    "plot_boundary_acoustic_detail_png",
    "plot_duration_distribution_png",
]


