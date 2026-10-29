"""
Backward compatibility layer for Phase 1 Segmentation Evaluation.
Directs imports to modules.evaluation.slicing.segmentation.
"""

from modules.evaluation.slicing.segmentation import compute_segmentation_integrity

__all__ = ["compute_segmentation_integrity"]
