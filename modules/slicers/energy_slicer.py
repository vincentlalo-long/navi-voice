"""
Energy-based Short-Time RMS Audio Slicer (DSP Strategy).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from .base import BaseSlicer, merge_short_segments, validate_intervals


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
        adaptive_threshold: bool = False,
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
        self.adaptive_threshold = adaptive_threshold

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergySlicer":
        """Instantiate EnergySlicer by extracting relevant hyperparameters from config."""
        adaptive_raw = config.get("adaptive_threshold", False)
        adaptive_threshold = (
            adaptive_raw.strip().lower() in ("true", "1", "yes")
            if isinstance(adaptive_raw, str)
            else bool(adaptive_raw)
        )
        return cls(
            threshold_db=float(config.get("silence_threshold_db", -38.0)),
            min_length_ms=int(config.get("min_length_ms", 2000)),
            max_length_ms=int(config.get("max_length_ms", 12000)),
            min_interval_ms=int(config.get("min_interval_ms", 400)),
            hop_size_ms=int(config.get("hop_size_ms", 10)),
            max_sil_kept_ms=int(config.get("max_sil_kept_ms", 400)),
            min_speech_ms=int(config.get("min_speech_ms", 800)),
            adaptive_threshold=adaptive_threshold,
        )

    def _get_rms(self, waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Calculate Root Mean Square (RMS) energy across framed signal."""
        pad_width = frame_length // 2
        padded = np.pad(waveform, (pad_width, pad_width), mode="reflect")
        shape = (int((len(padded) - frame_length) / hop_length) + 1, frame_length)
        strides = (padded.strides[0] * hop_length, padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.sqrt(np.mean(frames**2, axis=-1))

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """Slice waveform into valid speech segments using silence RMS thresholds."""
        if len(waveform) == 0:
            return []

        total = len(waveform)
        min_length = int(sampling_rate * self.min_length_ms / 1000)
        min_speech = int(sampling_rate * self.min_speech_ms / 1000)
        max_length = int(sampling_rate * self.max_length_ms / 1000)
        min_interval = int(sampling_rate * self.min_interval_ms / 1000)
        hop_size = int(sampling_rate * self.hop_size_ms / 1000)
        max_sil_kept = int(sampling_rate * self.max_sil_kept_ms / 1000)

        # A very short file still has to be distinguished from silence. The old
        # fallback returned the whole buffer even for an empty/silent input.
        if total < min_speech:
            rms_amp = float(np.sqrt(np.mean(np.asarray(waveform, dtype=np.float64) ** 2)))
            if rms_amp <= 10 ** (self.threshold_db / 20.0):
                return []
            return [(0, total)]

        rms = self._get_rms(waveform, hop_size * 2, hop_size)

        t_high, t_low = self.threshold_high, self.threshold_low
        if self.adaptive_threshold and len(rms) > 0:
            p10 = float(np.percentile(20 * np.log10(rms + 1e-9), 10))
            dyn_thresh = float(np.clip(p10 + 6.0, -48.0, -28.0))
            t_high = 10 ** ((dyn_thresh + 2.0) / 20.0)
            t_low = 10 ** ((dyn_thresh - 4.0) / 20.0)

        # Schmitt Trigger Hysteresis to eliminate chattering around threshold
        is_speech = False
        speech_mask = np.zeros(len(rms), dtype=bool)
        for i, r in enumerate(rms):
            if not is_speech and r >= t_high:
                is_speech = True
            elif is_speech and r < t_low:
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
            silence_regions.append((start_idx * hop_size, total))

        # ``cur_start`` includes padding, while ``speech_start`` is the actual
        # start of the current speech run. Using the padded start for the
        # min-speech check made a 400 ms noise blip look like an 800 ms word.
        chunks: List[Tuple[int, int]] = []
        cur_start = 0
        speech_start = 0
        keep_floor = max(1, int(sampling_rate * 0.2))

        for sil_start, sil_end in silence_regions:
            sil_duration = sil_end - sil_start
            if sil_duration < min_interval:
                continue

            # Mathematical zero-overlap guarantee:
            # If silence duration is short (< 2 * max_sil_kept), split right in the middle.
            # If silence duration is long, keep max_sil_kept on each boundary.
            if sil_duration < 2 * max_sil_kept:
                cut_point = sil_start + sil_duration // 2
                next_start = cut_point
            else:
                cut_point = min(total, sil_start + max_sil_kept)
                next_start = max(0, sil_end - max_sil_kept)

            speech_len = max(0, sil_start - speech_start)
            # Keep short exclamations too; they are merged below when a
            # neighbour is close. Only clicks/blips under 200 ms are dropped.
            if speech_len >= keep_floor and cut_point > cur_start:
                chunks.append((cur_start, cut_point))

            cur_start = next_start
            speech_start = sil_end

        if total - speech_start >= keep_floor and cur_start < total:
            chunks.append((cur_start, total))

        # Subdivide segments exceeding max_length with balanced, zero-crossing cuts
        final_chunks: List[Tuple[int, int]] = []
        for start, end in chunks:
            final_chunks.extend(
                self.balanced_subdivide(
                    waveform,
                    sampling_rate,
                    start,
                    end,
                    max_length,
                    lambda lo, hi: self.find_clean_split_in_range(waveform, lo, hi),
                )
            )

        # min_length_ms is a target utterance length for TTS: fragments below it are
        # merged into a neighbour when the gap is only a short pause.
        final_chunks = merge_short_segments(
            final_chunks,
            min_length_samples=min_length,
            max_length_samples=max_length,
            max_gap_samples=max_sil_kept * 2,
        )

        return validate_intervals(
            final_chunks,
            total_samples=total,
            sampling_rate=sampling_rate,
            max_length_samples=max_length,
        )
