"""
Abstract Base Class for Audio Slicers & VAD Segmenters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import gc
import numpy as np


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
