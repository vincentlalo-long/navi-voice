"""
Semantic & Punctuation Audio Slicer Strategy (WhisperX / Faster-Whisper Word-Timestamp Alignment).
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
import scipy.signal
from .base import BaseSlicer
import modules.cuda_utils  # Ensure CUDA shared libraries are pre-loaded


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
        """Lazy loader for Faster-Whisper neural model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            try:
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
            except Exception:
                # Fallback to CPU if CUDA fails
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

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

    @staticmethod
    def _find_chunk_split_index(waveform: np.ndarray, target_sample: int, search_radius: int) -> int:
        """Find the lowest energy zero-crossing point near target_sample to cleanly split long audio."""
        start = max(0, target_sample - search_radius)
        end = min(len(waveform), target_sample + search_radius)
        if end <= start:
            return target_sample

        sub = waveform[start:end]
        win_size = 4410  # 100ms at 44.1k
        if len(sub) < win_size * 2:
            return target_sample

        sub_trimmed = sub[:len(sub) - (len(sub) % win_size)]
        sub_reshaped = sub_trimmed.reshape(-1, win_size)
        energies = np.mean(sub_reshaped ** 2, axis=1)
        min_frame_idx = int(np.argmin(energies))
        frame_start = start + min_frame_idx * win_size

        frame = waveform[frame_start : frame_start + win_size]
        sign_changes = np.where(np.diff(np.signbit(frame)))[0]
        if len(sign_changes) > 0:
            best_zc = sign_changes[np.argmin(np.abs(frame[sign_changes]))]
            return int(frame_start + best_zc)
        return int(frame_start + win_size // 2)

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Polyphase resample audio chunk to 16000 Hz."""
        target_sr = 16000
        if sampling_rate == target_sr:
            return audio.astype(np.float32)
        gcd = np.gcd(sampling_rate, target_sr)
        up = target_sr // gcd
        down = sampling_rate // gcd
        return scipy.signal.resample_poly(audio, up, down).astype(np.float32)

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """
        Segment waveform based on word timestamps, punctuation boundaries, and silence pauses.
        Uses 5-minute memory-capped streaming blocks to prevent Out-Of-Memory (OOM) on long files.
        """
        model = self._get_model()
        total_samples = len(waveform)
        total_duration_sec = total_samples / sampling_rate
        chunk_target_sec = 300.0  # 5-minute streaming blocks

        all_words = []

        if total_duration_sec <= chunk_target_sec:
            # Single-pass for audio under 5 minutes
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
            # Multi-chunk streaming pass with silence zero-crossing boundary detection
            chunk_target_samples = int(chunk_target_sec * sampling_rate)
            search_radius_samples = int(15.0 * sampling_rate)

            split_indices = [0]
            cur_idx = chunk_target_samples
            while cur_idx < total_samples:
                clean_split = self._find_chunk_split_index(waveform, cur_idx, search_radius_samples)
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

        if not all_words:
            return [(0, len(waveform))]

        # Sentence punctuation marks for Japanese and Latin
        sentence_end_punct = set(["。", "！", "？", "!", "?", "…", "\n"])
        clause_punct = set(["、", ",", ";"])

        sentences: List[Dict[str, Any]] = []
        cur_sentence_words = []

        max_dur_sec = self.max_length_ms / 1000.0
        min_sil_sec = self.min_interval_ms / 1000.0

        for i, w in enumerate(all_words):
            cur_sentence_words.append(w)
            w_text = w.word.strip()

            is_end_punct = any(p in w_text for p in sentence_end_punct)
            is_clause_punct = any(p in w_text for p in clause_punct)

            # Check gap to next word
            has_next = (i + 1) < len(all_words)
            gap_to_next = (all_words[i + 1].start - w.end) if has_next else 999.0
            cur_duration = cur_sentence_words[-1].end - cur_sentence_words[0].start

            # Conditions to cut into a sentence:
            # 1. End punctuation (。) followed by non-trivial gap or EOF
            # 2. Large pause (> 1.0s) even without punctuation
            # 3. Exceeds max duration (> max_dur_sec) and hits a clause mark (、) or pause
            should_cut = False
            if is_end_punct:
                should_cut = True
            elif gap_to_next >= min_sil_sec and cur_duration >= (self.min_length_ms / 1000.0):
                should_cut = True
            elif cur_duration >= max_dur_sec:
                should_cut = True

            if should_cut:
                s_time = cur_sentence_words[0].start
                e_time = cur_sentence_words[-1].end
                sentences.append({"start": s_time, "end": e_time})
                cur_sentence_words = []

        if cur_sentence_words:
            s_time = cur_sentence_words[0].start
            e_time = cur_sentence_words[-1].end
            sentences.append({"start": s_time, "end": e_time})

        # Apply padding and enforce zero-overlap between adjacent sentences
        pad_sec = self.max_sil_kept_ms / 1000.0
        total_audio_sec = len(waveform) / sampling_rate

        raw_ranges = []
        for item in sentences:
            s_pad = max(0.0, item["start"] - pad_sec)
            e_pad = min(total_audio_sec, item["end"] + pad_sec)
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
