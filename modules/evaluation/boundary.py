"""
Backward compatibility layer for Phase 1 Boundary Evaluation.
Directs imports to modules.evaluation.slicing.boundary.
"""

from modules.evaluation.slicing.boundary import (
    evaluate_slice_boundary,
    aggregate_boundary_metrics,
    calculate_boundary_tolerance_metrics,
)

__all__ = [
    "evaluate_slice_boundary",
    "aggregate_boundary_metrics",
    "calculate_boundary_tolerance_metrics",
]
