"""
Backward compatibility layer for Phase 1 Benchmark Suite.
Directs imports to modules.evaluation.slicing.benchmark.
"""

from modules.evaluation.slicing.benchmark import (
    SlicingBenchmarkSuite,
    BenchmarkSuite,
)

__all__ = ["SlicingBenchmarkSuite", "BenchmarkSuite"]
