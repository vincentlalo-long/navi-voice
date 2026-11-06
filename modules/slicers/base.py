"""
Abstract Base Class for Audio Slicers & VAD Segmenters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import gc
import numpy as np


def merge_short_segments(
    chunks: List[Tuple[int, int]],
    min_length_samples: int,
    max_length_samples: int,
    max_gap_samples: int,
) -> List[Tuple[int, int]]:
    """Merge segments shorter than ``min_length_samples`` into the next segment.

    A slicer can only be said to honour ``min_length_ms`` if a shorter fragment is
    still merged when the boundary between the two neighbours is just a short pause
    rather than a real sentence break. Both the pause length and the resulting
    segment length are bounded, so merging never bridges a long silence and never
    creates a segment above ``max_length_ms``.
    """
    if not chunks or min_length_samples <= 0:
        return list(chunks)

    merged: List[Tuple[int, int]] = []
    for start, end in chunks:
        if merged:
            prev_start, prev_end = merged[-1]
            gap = start - prev_end
            if (
                (prev_end - prev_start) < min_length_samples
                and gap <= max_gap_samples
                and (end - prev_start) <= max_length_samples
            ):
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


class BaseSlicer(ABC):
    """
    Abstract strategy interface for voice segmentation algorithms.
    Subclasses must implement the `slice` method.
    """

    def release(self) -> None:
        """Drop the cached model and release accelerator memory (best effort)."""
        if getattr(self, "_model", None) is not None:
            self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @abstractmethod
    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """
        Split a continuous 1D audio waveform into valid speech segments.

        Args:
            waveform: 1D float32 numpy array representing the mono audio signal.
            sampling_rate: Sampling rate (e.g., 44100).

        Returns:
            List of (start_sample, end_sample) tuple boundaries.
        """
        pass
