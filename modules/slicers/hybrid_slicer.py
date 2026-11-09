"""
Hybrid Audio Slicer Strategy (Semantic Macro-Segmentation + Acoustic Micro-Refinement).
Combines Whisper word-level punctuation alignment with DSP short-time energy dips and zero-crossing snapping.
Inherits from WhisperXSlicer for shared streaming and semantic grouping.
"""

from typing import List, Tuple
import numpy as np
from .base import merge_short_segments, validate_intervals
from .whisper_slicer import WhisperXSlicer
from .. import cuda_utils  # Ensure CUDA shared libraries are pre-loaded


class HybridSlicer(WhisperXSlicer):
    """
    Two-Stage Hybrid Audio Slicer:
      - Stage 1 (Macro): Semantic sentence boundary detection via Faster-Whisper & punctuation (。！？)
                         with 5-minute streaming chunking to eliminate VRAM exhaustion.
      - Stage 2 (Micro): Acoustic boundary refinement scanning for the lowest short-time RMS energy dip
                         and snapping to the nearest zero-crossing sample inside the silence gap.
    """

    def _find_deepest_silence_dip(
        self, waveform: np.ndarray, sampling_rate: int, start_sample: int, end_sample: int
    ) -> int:
        """
        Find the sample index of the lowest short-time RMS energy dip and zero-crossing
        strictly clamped within [0, len(waveform)].
        """
        total = len(waveform)
        start_sample = min(total, max(0, start_sample))
        end_sample = min(total, max(start_sample, end_sample))

        if end_sample - start_sample < int(sampling_rate * 0.05):
            return (start_sample + end_sample) // 2

        sub = waveform[start_sample:end_sample]
        hop = int(sampling_rate * 0.01)  # 10ms
        win = int(sampling_rate * 0.02)  # 20ms
        if len(sub) < win * 2:
            mid = (start_sample + end_sample) // 2
            return min(total, max(0, self._find_clean_split_point(waveform, mid, int(sampling_rate * 0.02))))

        pad_w = win // 2
        padded = np.pad(sub, (pad_w, pad_w), mode="reflect")
        shape = (int((len(padded) - win) / hop) + 1, win)
        strides = (padded.strides[0] * hop, padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames**2, axis=-1))

        min_frame_idx = int(np.argmin(rms))
        target_sample = start_sample + min_frame_idx * hop
        clean = self._find_clean_split_point(waveform, target_sample, hop)
        return min(total, max(0, clean))

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """Two-Stage Hybrid slicing pipeline with streaming protection and acoustic dip snapping."""
        total_samples = len(waveform)
        if total_samples == 0:
            return []

        # 1. Macro-Level: Word Timestamp Extraction (with 5-min streaming chunking)
        all_words = self._extract_words(waveform, sampling_rate)
        if not all_words:
            return []

        # 2. Semantic Clause Grouping
        sentences = self._group_words_into_sentences(all_words)
        if not sentences:
            return []

        pad_samples = int(sampling_rate * self.max_sil_kept_ms / 1000.0)
        max_samples = int(sampling_rate * self.max_length_ms / 1000)

        # 3. Build one padded range per sentence. Long inter-sentence pauses are
        # left outside the ranges; short pauses get a single shared dip. The old
        # single-boundary chain kept every second of silence between the first
        # and last sentence and then subdivided that silence into dead clips.
        segments: List[List[int]] = []
        for s_t, e_t in sentences:
            s = max(0, min(total_samples, int(s_t * sampling_rate)))
            e = max(0, min(total_samples, int(e_t * sampling_rate)))
            if e <= s:
                continue
            segments.append([max(0, s - pad_samples), min(total_samples, e + pad_samples)])

        if not segments:
            return []

        first_s = max(0, min(total_samples, int(sentences[0][0] * sampling_rate)))
        segments[0][0] = self._find_deepest_silence_dip(
            waveform, sampling_rate, max(0, first_s - pad_samples), first_s
        )

        last_e = max(0, min(total_samples, int(sentences[-1][1] * sampling_rate)))
        segments[-1][1] = self._find_deepest_silence_dip(
            waveform, sampling_rate, last_e, min(total_samples, last_e + pad_samples)
        )

        for idx in range(len(segments) - 1):
            cur_end = max(0, min(total_samples, int(sentences[idx][1] * sampling_rate)))
            next_start = max(0, min(total_samples, int(sentences[idx + 1][0] * sampling_rate)))

            if next_start > cur_end:
                gap = next_start - cur_end
                if gap <= 2 * pad_samples:
                    # Short pause: one shared micro-refined boundary.
                    dip = self._find_deepest_silence_dip(
                        waveform, sampling_rate, cur_end, next_start
                    )
                    segments[idx][1] = dip
                    segments[idx + 1][0] = dip
                else:
                    # Long pause: keep a bounded tail for the current sentence
                    # and a bounded lead for the next one, dropping the middle.
                    end_cut = self._find_deepest_silence_dip(
                        waveform, sampling_rate, cur_end, min(total_samples, cur_end + pad_samples)
                    )
                    start_cut = self._find_deepest_silence_dip(
                        waveform, sampling_rate, max(0, next_start - pad_samples), next_start
                    )
                    segments[idx][1] = end_cut
                    segments[idx + 1][0] = start_cut
            else:
                # Overlapping word timestamps: resolve at the overlap midpoint.
                overlap_start = max(segments[idx][0], segments[idx + 1][0])
                overlap_end = min(segments[idx][1], segments[idx + 1][1])
                if overlap_end > overlap_start:
                    mid = (overlap_start + overlap_end) // 2
                else:
                    mid = (cur_end + next_start) // 2
                mid = max(segments[idx][0] + 1, min(segments[idx][1], mid))
                segments[idx][1] = mid
                segments[idx + 1][0] = mid

        # 4. Balanced Subdivision for long segments (> max_length)
        final_chunks: List[Tuple[int, int]] = []
        for s_p, e_p in segments:
            s_p = max(0, min(total_samples, int(s_p)))
            e_p = max(0, min(total_samples, int(e_p)))
            if e_p <= s_p:
                continue

            final_chunks.extend(
                self.balanced_subdivide(
                    waveform,
                    sampling_rate,
                    s_p,
                    e_p,
                    max_samples,
                    lambda lo, hi: self._find_deepest_silence_dip(
                        waveform, sampling_rate, lo, hi
                    ),
                )
            )

        final_chunks = merge_short_segments(
            final_chunks,
            min_length_samples=int(sampling_rate * self.min_length_ms / 1000),
            max_length_samples=max_samples,
            max_gap_samples=pad_samples * 2,
        )

        # Keep short exclamations; only drop sub-200 ms clicks.
        keep_floor = max(1, int(sampling_rate * 0.2))
        final_chunks = [(s, e) for s, e in final_chunks if e - s >= keep_floor]

        return validate_intervals(
            final_chunks,
            total_samples=total_samples,
            sampling_rate=sampling_rate,
            max_length_samples=max_samples,
        )
