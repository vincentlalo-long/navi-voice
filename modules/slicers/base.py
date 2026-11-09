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
    """Merge short segments into a neighbour across a short pause.

    The merge is bidirectional: a short fragment is absorbed by the previous
    segment when the pause is short enough, otherwise the following segment
    absorbs it. Long pauses are never bridged and the result never exceeds
    ``max_length_samples``.
    """
    if not chunks or min_length_samples <= 0:
        return list(chunks)

    merged: List[Tuple[int, int]] = []
    for start, end in chunks:
        start, end = int(start), int(end)
        if end <= start:
            continue

        if merged:
            prev_start, prev_end = merged[-1]
            gap = start - prev_end
            combined = end - prev_start
            prev_short = (prev_end - prev_start) < min_length_samples
            cur_short = (end - start) < min_length_samples
            if (
                gap <= max_gap_samples
                and combined <= max_length_samples
                and (prev_short or cur_short)
            ):
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def validate_intervals(
    chunks: List[Tuple[int, int]],
    total_samples: int,
    sampling_rate: int,
    max_length_samples: int | None = None,
    allow_overlap: bool = False,
) -> List[Tuple[int, int]]:
    """Validate the interval contract shared by every slicer.

    Every returned interval must satisfy ``0 <= start < end <= total_samples``,
    be ordered, not overlap its predecessor, and (when supplied) not exceed
    ``max_length_samples``. A violation is a slicer bug, not bad input, so it is
    raised instead of silently returning a malformed clip.
    """
    prev_start = -1
    prev_end = 0
    cleaned: List[Tuple[int, int]] = []

    for index, (start, end) in enumerate(chunks):
        start, end = int(start), int(end)
        if start < 0 or end > total_samples or start >= end:
            raise ValueError(
                f"Invalid interval #{index} for {total_samples} samples at {sampling_rate} Hz: "
                f"({start}, {end})"
            )
        if max_length_samples is not None and (end - start) > max_length_samples:
            raise ValueError(
                f"Interval #{index} exceeds max length: {end - start} > {max_length_samples} samples"
            )
        if start < prev_start:
            raise ValueError(f"Intervals are not sorted at #{index}: {start} < {prev_start}")
        if not allow_overlap and start < prev_end:
            raise ValueError(
                f"Intervals overlap at #{index}: start {start} < previous end {prev_end}"
            )
        cleaned.append((start, end))
        prev_start, prev_end = start, end

    return cleaned


class BaseSlicer(ABC):
    """
    Abstract strategy interface for voice segmentation algorithms.
    Subclasses must implement the `slice` method.
    """

    @staticmethod
    def find_clean_split_in_range(waveform: np.ndarray, lo: int, hi: int) -> int:
        """Return a low-amplitude cut sample inside ``[lo, hi)``.

        Zero crossings are preferred. For each crossing the quieter of the two
        adjacent samples is chosen, so the first sample of the next clip is the
        one that minimises the transient step. If no crossing exists, the
        quietest sample in the window is used.
        """
        n = len(waveform)
        lo = max(0, min(n, int(lo)))
        hi = max(lo, min(n, int(hi)))
        if hi <= lo:
            return lo

        sub = waveform[lo:hi]
        if len(sub) < 2:
            return lo

        sign_changes = np.where(np.diff(np.signbit(sub)))[0]
        if len(sign_changes) > 0:
            # sign_changes[i] is the sample immediately before the crossing.
            # Cut at whichever adjacent sample has the smaller amplitude.
            candidates = np.where(
                np.abs(sub[sign_changes]) <= np.abs(sub[sign_changes + 1]),
                sign_changes,
                sign_changes + 1,
            )
            best = int(candidates[np.argmin(np.abs(sub[candidates]))])
            return lo + best

        return lo + int(np.argmin(np.abs(sub)))

    @staticmethod
    def _find_clean_split_point(waveform: np.ndarray, target_idx: int, window: int) -> int:
        """Compatibility helper: clean cut within +/- ``window`` of a target."""
        n = len(waveform)
        target_idx = max(0, min(n, int(target_idx)))
        w_start = max(0, target_idx - int(window))
        w_end = min(n, target_idx + int(window) + 1)
        if w_end <= w_start:
            return target_idx
        return BaseSlicer.find_clean_split_in_range(waveform, w_start, w_end)

    def balanced_subdivide(
        self,
        waveform: np.ndarray,
        sampling_rate: int,
        start: int,
        end: int,
        max_samples: int,
        find_cut,
    ) -> List[Tuple[int, int]]:
        """Split ``[start, end)`` into pieces no longer than ``max_samples``.

        Unlike a naive ``target +/- 0.5 s`` search, the cut window is narrowed
        whenever the ideal piece is already close to the ceiling. The remaining
        length is also reserved for the not-yet-created pieces. The result is
        therefore guaranteed to respect the maximum even after zero-crossing
        snapping moves the cut.
        """
        n = len(waveform)
        start = max(0, min(n, int(start)))
        end = max(0, min(n, int(end)))
        if end <= start:
            return []

        length = end - start
        if max_samples <= 0 or length <= max_samples:
            return [(start, end)]

        num_splits = int(np.ceil(length / float(max_samples)))
        ideal = max(1, length // num_splits)
        search_win = max(0, int(sampling_rate * 0.5))
        win_eff = min(search_win, max(0, max_samples - ideal))

        pieces: List[Tuple[int, int]] = []
        cur = start
        for index in range(1, num_splits):
            remaining_pieces = num_splits - index
            target = start + index * ideal

            # Lower bound: the current piece must leave enough room for the
            # remaining pieces; the cut must also be after the previous one.
            lo = max(cur + 1, end - remaining_pieces * max_samples, target - win_eff)
            # Upper bound: the current piece must not exceed the ceiling and
            # at least one sample must remain for every later piece.
            hi = min(cur + max_samples, end - remaining_pieces, target + win_eff)

            if hi <= lo:
                cut = min(cur + max_samples, end - remaining_pieces)
                cut = max(cut, cur + 1)
            else:
                cut = int(find_cut(lo, hi))
                cut = min(max(cut, lo), hi)

            if cut <= cur:
                cut = min(cur + max_samples, end - 1)
            if cut <= cur:
                break

            pieces.append((cur, cut))
            cur = cut

        if cur < end:
            pieces.append((cur, end))

        return [(s, e) for s, e in pieces if e > s]

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
