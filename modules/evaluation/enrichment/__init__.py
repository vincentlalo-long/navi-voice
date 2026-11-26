"""Phase 3 enrichment evaluation, metrics aggregation, visualization, and reporting."""

from .metrics import compute_enrichment_metrics
from .report import generate_enrichment_report
from .visualize import (
    plot_category_distribution_png,
    plot_prosody_acoustic_space_png,
    plot_qc_waterfall_png,
)

__all__ = [
    "compute_enrichment_metrics",
    "generate_enrichment_report",
    "plot_category_distribution_png",
    "plot_prosody_acoustic_space_png",
    "plot_qc_waterfall_png",
]
