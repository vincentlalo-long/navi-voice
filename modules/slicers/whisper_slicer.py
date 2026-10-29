"""
Semantic & Punctuation Audio Slicer Strategy (WhisperX / Faster-Whisper Word-Timestamp Alignment).
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
import scipy.signal
from .base import BaseSlicer
from .. import cuda_utils  # Ensure CUDA shared libraries are pre-loaded


class WhisperXSlicer(BaseSlicer):
    """
    Slices audio according to semantic sentence boundaries and punctuation marks
    using Faster-Whisper word-level timestamp forced alignment.

    Advantages:
      - Eradicates over-segmentation caused by mid-sentence dramatic breath pauses.
      - Aligns cuts strictly with complete grammatical clauses and punctuation (。！？).
      - Enforces zero-overlap midpoint boundaries between adjacent sentences.
      - Zero-crossing snapping and balanced subdivision for long monologues (> max_length).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
        min_length_ms: int = 2000,
        max_length_ms: int = 12000,
        min_interval_ms: int = 400,
        max_sil_kept_ms: int = 400,
        min_speech_ms: int = 800,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.min_length_ms = min_length_ms
        self.max_length_ms = max_length_ms
        self.min_interval_ms = min_interval_ms
        self.max_sil_kept_ms = max_sil_kept_ms
        self.min_speech_ms = min_speech_ms
        self._model = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WhisperXSlicer":
        """Instantiate WhisperXSlicer from configuration dictionary."""
        return cls(
            model_size=str(config.get("whisper_model_size", config.get("model_size", "base"))),
            device=str(config.get("device", "cuda")),
            compute_type=str(config.get("compute_type", "float16")),
            language=str(config.get("language", "ja")),
            min_length_ms=int(config.get("min_length_ms", 2000)),
            max_length_ms=int(config.get("max_length_ms", 12000)),
            min_interval_ms=int(config.get("min_interval_ms", 400)),
            max_sil_kept_ms=int(config.get("max_sil_kept_ms", 400)),
            min_speech_ms=int(config.get("min_speech_ms", 800)),
        )

    def _get_model(self):
        """Lazy loader for Faster-Whisper ASR model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            actual_compute = self.compute_type if self.device == "cuda" else "int8"
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=actual_compute,
                )
            except Exception as exc:
                # Automatic graceful fallback to CPU if CUDA initialization fails
                import warnings

                warnings.warn(
                    f"Faster-Whisper CUDA init failed ({exc}); falling back to CPU. "
                    "This will be much slower and CPU-heavy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = "cpu"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
        return self._model

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Polyphase anti-aliased resampling to 16kHz for Whisper."""
        target_sr = 16000
        if sampling_rate == target_sr:
            return audio.astype(np.float32)

        gcd = np.gcd(sampling_rate, target_sr)
        up = target_sr // gcd
        down = sampling_rate // gcd
        return scipy.signal.resample_poly(audio, up, down).astype(np.float32)

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

    def _find_chunk_split_index(
        self,
        waveform: np.ndarray,
        target_idx: int,
        search_radius: int,
        sampling_rate: int,
    ) -> int:
        """Find the best split point within search_radius of target_idx using RMS energy dips."""
        w_start = max(0, target_idx - search_radius)
        w_end = min(len(waveform), target_idx + search_radius)
        if w_end <= w_start:
            return target_idx

        sub = waveform[w_start:w_end]
        hop = max(1, int(round(sampling_rate * 0.010)))
        win = max(2, int(round(sampling_rate * 0.020)))
        if len(sub) < win * 2:
            return self._find_clean_split_point(waveform, target_idx, search_radius)

        pad_w = win // 2
        padded = np.pad(sub, (pad_w, pad_w), mode="reflect")
        shape = (int((len(padded) - win) / hop) + 1, win)
        strides = (padded.strides[0] * hop, padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames**2, axis=-1))

        min_frame_idx = int(np.argmin(rms))
        best_rough_idx = w_start + min_frame_idx * hop
        return self._find_clean_split_point(waveform, best_rough_idx, hop)

    def _extract_words(self, waveform: np.ndarray, sampling_rate: int) -> List[Any]:
        """
        Extract word timestamps using streaming chunking (5-minute blocks)
        to prevent VRAM exhaustion and GPU OOM on long audio.
        """
        model = self._get_model()
        total_samples = len(waveform)
        total_duration_sec = total_samples / sampling_rate
        chunk_target_sec = 300.0  # 5-minute streaming blocks

        all_words = []

        if total_duration_sec <= chunk_target_sec:
            audio_16k = self._resample_to_16k(waveform, sampling_rate)
            segments, _ = model.transcribe(
                audio_16k,
                language=self.language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=self.min_interval_ms),
            )
            for seg in segments:
                if seg.words:
                    all_words.extend(seg.words)
                else:
                    all_words.append(
                        type("Word", (), {"start": seg.start, "end": seg.end, "word": seg.text})()
                    )
        else:
            chunk_target_samples = int(chunk_target_sec * sampling_rate)
            search_radius_samples = int(15.0 * sampling_rate)

            split_indices = [0]
            cur_idx = chunk_target_samples
            while cur_idx < total_samples:
                clean_split = self._find_chunk_split_index(
                    waveform, cur_idx, search_radius_samples, sampling_rate
                )
                split_indices.append(clean_split)
                cur_idx = clean_split + chunk_target_samples
            split_indices.append(total_samples)

            for c_i in range(len(split_indices) - 1):
                c_start = split_indices[c_i]
                c_end = split_indices[c_i + 1]
                if c_end <= c_start:
                    continue

                chunk_audio = waveform[c_start:c_end]
                offset_sec = c_start / sampling_rate

                chunk_16k = self._resample_to_16k(chunk_audio, sampling_rate)
                segments, _ = model.transcribe(
                    chunk_16k,
                    language=self.language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=self.min_interval_ms),
                )
                for seg in segments:
                    if seg.words:
                        for w in seg.words:
                            all_words.append(
                                type("Word", (), {
                                    "start": w.start + offset_sec,
                                    "end": w.end + offset_sec,
                                    "word": w.word,
                                })()
                            )
                    else:
                        all_words.append(
                            type("Word", (), {
                                "start": seg.start + offset_sec,
                                "end": seg.end + offset_sec,
                                "word": seg.text,
                            })()
                        )
                del chunk_16k
                import gc
                gc.collect()

        return all_words

    def _group_words_into_sentences(self, all_words: List[Any]) -> List[Tuple[float, float]]:
        """Group word timestamps into semantic clauses using Japanese/Latin punctuation rules."""
        if not all_words:
            return []

        sentence_end_punct = set(["。", "！", "？", "!", "?", "…", "\n"])
        clause_punct = set(["、", ",", ";"])

        sentences: List[Tuple[float, float]] = []
        cur_sentence_words = []

        max_dur_sec = self.max_length_ms / 1000.0
        min_sil_sec = self.min_interval_ms / 1000.0

        for i, w in enumerate(all_words):
            cur_sentence_words.append(w)
            w_text = w.word.strip()

            is_end_punct = any(p in w_text for p in sentence_end_punct)
            is_clause_punct = any(p in w_text for p in clause_punct)

            has_next = (i + 1) < len(all_words)
            gap_to_next = (all_words[i + 1].start - w.end) if has_next else 999.0
            cur_duration = cur_sentence_words[-1].end - cur_sentence_words[0].start

            should_cut = False
            if is_end_punct:
                should_cut = True
            elif gap_to_next >= min_sil_sec and cur_duration >= (self.min_length_ms / 1000.0):
                should_cut = True
            elif is_clause_punct and cur_duration >= (max_dur_sec * 0.75):
                should_cut = True
            elif cur_duration >= max_dur_sec:
                should_cut = True

            if should_cut:
                sentences.append((cur_sentence_words[0].start, cur_sentence_words[-1].end))
                cur_sentence_words = []

        if cur_sentence_words:
            sentences.append((cur_sentence_words[0].start, cur_sentence_words[-1].end))

        return sentences

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """Slice audio according to semantic punctuation boundaries with zero-crossing snapping."""
        all_words = self._extract_words(waveform, sampling_rate)
        if not all_words:
            return [(0, len(waveform))]

        sentences = self._group_words_into_sentences(all_words)

        # Apply padding and enforce zero-overlap between adjacent sentences
        pad_sec = self.max_sil_kept_ms / 1000.0
        total_audio_sec = len(waveform) / sampling_rate

        raw_ranges = []
        for s_t, e_t in sentences:
            s_pad = max(0.0, s_t - pad_sec)
            e_pad = min(total_audio_sec, e_t + pad_sec)
            raw_ranges.append((s_pad, e_pad))

        # Collision resolution at midpoint
        resolved_ranges: List[Tuple[float, float]] = []
        for s, e in raw_ranges:
            if not resolved_ranges:
                resolved_ranges.append((s, e))
                continue

            prev_s, prev_e = resolved_ranges[-1]
            if s < prev_e:
                mid = (prev_e + s) / 2.0
                resolved_ranges[-1] = (prev_s, mid)
                s = mid
            resolved_ranges.append((s, e))

        # Convert to sample indices and apply balanced zero-crossing cuts
        max_samples = int(sampling_rate * self.max_length_ms / 1000)
        min_speech_samples = int(sampling_rate * self.min_speech_ms / 1000)

        final_chunks: List[Tuple[int, int]] = []
        for s_sec, e_sec in resolved_ranges:
            s_idx = int(s_sec * sampling_rate)
            e_idx = min(len(waveform), int(e_sec * sampling_rate))
            chunk_len = e_idx - s_idx

            if chunk_len < min_speech_samples:
                continue

            if chunk_len <= max_samples:
                final_chunks.append((s_idx, e_idx))
                continue

            # Balanced k-way split
            num_splits = int(np.ceil(chunk_len / max_samples))
            ideal_len = chunk_len // num_splits
            cur_p = s_idx
            search_win = int(sampling_rate * 0.5)

            for idx in range(1, num_splits):
                target_p = s_idx + idx * ideal_len
                clean_cut = self._find_clean_split_point(waveform, target_p, search_win)
                final_chunks.append((cur_p, clean_cut))
                cur_p = clean_cut
            final_chunks.append((cur_p, e_idx))

        return final_chunks
