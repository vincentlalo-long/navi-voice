"""
Hybrid Audio Slicer Strategy (Semantic Macro-Segmentation + Acoustic Micro-Refinement).
Combines Whisper word-level punctuation alignment with DSP short-time energy dips and zero-crossing snapping.
Inherits from WhisperXSlicer for shared streaming and semantic grouping.
"""

from typing import List, Tuple
import numpy as np
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

        # 1. Macro-Level: Word Timestamp Extraction (with 5-min streaming chunking)
        all_words = self._extract_words(waveform, sampling_rate)
        if not all_words:
            return [(0, total_samples)]

        # 2. Semantic Clause Grouping
        sentences = self._group_words_into_sentences(all_words)
        if not sentences:
            return [(0, total_samples)]

        # 3. Micro-Refinement: Scan for deepest energy dip in silence intervals
        pad_samples = int(sampling_rate * self.max_sil_kept_ms / 1000.0)
        refined_splits = []

        # Start boundary for first sentence
        first_s = int(sentences[0][0] * sampling_rate)
        refined_splits.append(self._find_deepest_silence_dip(
            waveform, sampling_rate, first_s - pad_samples, first_s
        ))

        for idx in range(len(sentences) - 1):
            cur_end = int(sentences[idx][1] * sampling_rate)
            next_start = int(sentences[idx + 1][0] * sampling_rate)
            if next_start > cur_end:
                dip = self._find_deepest_silence_dip(waveform, sampling_rate, cur_end, next_start)
            else:
                mid = (cur_end + next_start) // 2
                dip = min(total_samples, max(0, self._find_clean_split_point(waveform, mid, int(sampling_rate * 0.05))))
            refined_splits.append(dip)

        last_e = int(sentences[-1][1] * sampling_rate)
        refined_splits.append(self._find_deepest_silence_dip(
            waveform, sampling_rate, last_e, last_e + pad_samples
        ))

        # 4. Balanced Subdivision for long segments (> max_length)
        final_chunks: List[Tuple[int, int]] = []
        max_samples = int(sampling_rate * self.max_length_ms / 1000)
        min_speech_samples = int(sampling_rate * self.min_speech_ms / 1000)

        for i in range(len(refined_splits) - 1):
            s_p = refined_splits[i]
            e_p = refined_splits[i + 1]
            seg_len = e_p - s_p
            if seg_len < min_speech_samples:
                continue

            if seg_len <= max_samples:
                final_chunks.append((s_p, e_p))
            else:
                num_splits = int(np.ceil(seg_len / max_samples))
                ideal_len = seg_len // num_splits
                cur_p = s_p
                search_win = int(sampling_rate * 0.5)
                for s_i in range(1, num_splits):
                    target_p = s_p + s_i * ideal_len
                    clean_cut = self._find_deepest_silence_dip(
                        waveform, sampling_rate, target_p - search_win, target_p + search_win
                    )
                    final_chunks.append((cur_p, clean_cut))
                    cur_p = clean_cut
                final_chunks.append((cur_p, e_p))

        return final_chunks
