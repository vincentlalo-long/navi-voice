"""
Evaluation Module Suite for Voice Pipeline.
Structured by Pipeline Phases:
  - Phase 1: slicing/ (Speech Slicing, VAD Boundaries, and Segmentation Integrity)
  - Phase 3/4: tts/ (TTS Synthesis Quality, Voice Similarity, and Intelligibility)
"""

# Re-export Phase 1 Slicing Evaluation components for backward compatibility
from .slicing import (
    evaluate_slice_boundary,
    aggregate_boundary_metrics,
    calculate_boundary_tolerance_metrics,
    compute_segmentation_integrity,
    SlicingBenchmarkSuite,
    BenchmarkSuite,
    plot_benchmark_comparison_png,
    plot_boundary_acoustic_detail_png,
    plot_duration_distribution_png,
)

__all__ = [
    "evaluate_slice_boundary",
    "aggregate_boundary_metrics",
    "calculate_boundary_tolerance_metrics",
    "compute_segmentation_integrity",
    "SlicingBenchmarkSuite",
    "BenchmarkSuite",
    "plot_benchmark_comparison_png",
    "plot_boundary_acoustic_detail_png",
    "plot_duration_distribution_png",
]
