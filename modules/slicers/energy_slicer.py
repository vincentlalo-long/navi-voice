"""
Energy-based Short-Time RMS Audio Slicer (DSP Strategy).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from .base import BaseSlicer


class EnergySlicer(BaseSlicer):
    """
    Slices audio using short-time Root Mean Square (RMS) energy thresholding.
    Features:
      - Schmitt trigger hysteresis to eliminate boundary chattering.
      - Mathematical zero-overlap guarantee on continuous dialogue.
      - Zero-crossing snapping and balanced subdivision for long phrases (> max_length).
    """

    def __init__(
        self,
        threshold_db: float = -38.0,
        min_length_ms: int = 2000,
        max_length_ms: int = 12000,
        min_interval_ms: int = 400,
        hop_size_ms: int = 10,
        max_sil_kept_ms: int = 400,
        min_speech_ms: int = 800,
    ):
        self.threshold_db = threshold_db
        # Dual-threshold Schmitt trigger: +2 dB to turn speech on, -4 dB to turn speech off
        self.threshold_high = 10 ** ((threshold_db + 2.0) / 20.0)
        self.threshold_low = 10 ** ((threshold_db - 4.0) / 20.0)
        self.min_length_ms = min_length_ms
        self.max_length_ms = max_length_ms
        self.min_interval_ms = min_interval_ms
        self.hop_size_ms = hop_size_ms
        self.max_sil_kept_ms = max_sil_kept_ms
        self.min_speech_ms = min_speech_ms

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergySlicer":
        """Instantiate EnergySlicer by extracting relevant hyperparameters from config."""
        return cls(
            threshold_db=float(config.get("silence_threshold_db", -38.0)),
            min_length_ms=int(config.get("min_length_ms", 2000)),
            max_length_ms=int(config.get("max_length_ms", 12000)),
            min_interval_ms=int(config.get("min_interval_ms", 400)),
            hop_size_ms=int(config.get("hop_size_ms", 10)),
            max_sil_kept_ms=int(config.get("max_sil_kept_ms", 400)),
            min_speech_ms=int(config.get("min_speech_ms", 800)),
        )

    def _get_rms(self, waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Calculate Root Mean Square (RMS) energy across framed signal."""
        pad_width = frame_length // 2
        padded = np.pad(waveform, (pad_width, pad_width), mode="reflect")
        shape = (int((len(padded) - frame_length) / hop_length) + 1, frame_length)
        strides = (padded.strides[0] * hop_length, padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.sqrt(np.mean(frames**2, axis=-1))

    @staticmethod
    def _find_clean_split_point(waveform: np.ndarray, target_idx: int, window: int) -> int:
        """Find the zero-crossing sample with minimum amplitude near target_idx."""
        w_start = max(0, target_idx - window)
        w_end = min(len(waveform) - 1, target_idx + window)
        if w_end <= w_start:
            return target_idx

        sub_wave = waveform[w_start:w_end]
        sign_changes = np.where(np.diff(np.signbit(sub_wave)))[0]
        if len(sign_changes) > 0:
            abs_amps = np.abs(sub_wave[sign_changes])
            best_idx = sign_changes[np.argmin(abs_amps)]
            return int(w_start + best_idx)
        return int(w_start + np.argmin(np.abs(sub_wave)))

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """Slice waveform into valid speech segments using silence RMS thresholds."""
        min_length = int(sampling_rate * self.min_length_ms / 1000)
        min_speech = int(sampling_rate * self.min_speech_ms / 1000)
        max_length = int(sampling_rate * self.max_length_ms / 1000)
        min_interval = int(sampling_rate * self.min_interval_ms / 1000)
        hop_size = int(sampling_rate * self.hop_size_ms / 1000)
        max_sil_kept = int(sampling_rate * self.max_sil_kept_ms / 1000)

        if len(waveform) < min_speech:
            return [(0, len(waveform))]

        rms = self._get_rms(waveform, hop_size * 2, hop_size)

        # Schmitt Trigger Hysteresis to eliminate chattering around threshold
        is_speech = False
        speech_mask = np.zeros(len(rms), dtype=bool)
        for i, r in enumerate(rms):
            if not is_speech and r >= self.threshold_high:
                is_speech = True
            elif is_speech and r < self.threshold_low:
                is_speech = False
            speech_mask[i] = is_speech

        silence_mask = ~speech_mask

        # Detect contiguous silence regions
        silence_regions = []
        is_silent = False
        start_idx = 0

        for i, val in enumerate(silence_mask):
            if val and not is_silent:
                is_silent = True
                start_idx = i
            elif not val and is_silent:
                is_silent = False
                silence_regions.append((start_idx * hop_size, i * hop_size))
        if is_silent:
            silence_regions.append((start_idx * hop_size, len(waveform)))

        chunks = []
        cur_start = 0

        for sil_start, sil_end in silence_regions:
            sil_duration = sil_end - sil_start
            if sil_duration < min_interval:
                continue

            chunk_len = sil_start - cur_start
            # Mathematical zero-overlap guarantee:
            # If silence duration is short (< 2 * max_sil_kept), split right in the middle.
            # If silence duration is long, keep max_sil_kept on each boundary.
            if sil_duration < 2 * max_sil_kept:
                mid = sil_start + sil_duration // 2
                cut_point = mid
                next_start = mid
            else:
                cut_point = sil_start + max_sil_kept
                next_start = sil_end - max_sil_kept

            if chunk_len >= min_speech:
                chunks.append((cur_start, min(cut_point, len(waveform))))
                cur_start = next_start
            else:
                # Preceding audio was too short (< min_speech e.g. noise blip or initial silence).
                # Advance cur_start so it doesn't drag initial noise into next real utterance.
                cur_start = next_start

        # Handle final chunk
        if len(waveform) - cur_start >= min_speech:
            chunks.append((cur_start, len(waveform)))

        # Subdivide segments exceeding max_length with balanced, zero-crossing cuts
        final_chunks = []
        for start, end in chunks:
            chunk_dur = end - start
            if chunk_dur <= max_length:
                final_chunks.append((start, end))
                continue

            # Balanced k-way split to avoid dropping leftover tails
            num_splits = int(np.ceil(chunk_dur / max_length))
            ideal_len = chunk_dur // num_splits
            cur_p = start
            search_win = int(sampling_rate * 0.5)

            for s_idx in range(1, num_splits):
                target_p = start + s_idx * ideal_len
                clean_cut = self._find_clean_split_point(waveform, target_p, search_win)
                final_chunks.append((cur_p, clean_cut))
                cur_p = clean_cut
            final_chunks.append((cur_p, end))

        return final_chunks
