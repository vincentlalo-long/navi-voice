"""
Neural Voice Activity Detection Slicer powered by Silero VAD (Deep Learning Strategy).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.signal
from .base import BaseSlicer


class SileroSlicer(BaseSlicer):
    """
    Slices audio using Silero VAD (a pre-trained recurrent neural network).
    Features:
      - Separate vad_min_speech_ms (250ms) to preserve short emotional dialogue (e.g. 'Hai!', 'Eh?').
      - Polyphase anti-aliased resampling to 16kHz for neural evaluation.
      - Inverse coordinate mapping back to original 44.1kHz studio resolution.
      - Balanced zero-crossing subdivision for long segments to prevent Gibbs clicks and tail loss.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_length_ms: int = 2000,
        max_length_ms: int = 12000,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 400,
        vad_min_speech_ms: int = 250,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.min_length_ms = min_length_ms
        self.max_length_ms = max_length_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.vad_min_speech_ms = vad_min_speech_ms
        self.device = device
        self._model = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SileroSlicer":
        """Instantiate SileroSlicer by extracting relevant hyperparameters from config."""
        return cls(
            threshold=float(config.get("threshold", 0.5)),
            min_length_ms=int(config.get("min_length_ms", 2000)),
            max_length_ms=int(config.get("max_length_ms", 12000)),
            min_silence_duration_ms=int(config.get("min_interval_ms", 400)),
            speech_pad_ms=int(config.get("max_sil_kept_ms", 400)),
            vad_min_speech_ms=int(config.get("vad_min_speech_ms", 250)),
            device=str(config.get("device", "cpu")),
        )

    def _get_model(self):
        """Lazy loader for Silero VAD neural model."""
        if self._model is None:
            try:
                import warnings
                import torch
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    from silero_vad import load_silero_vad
                    self._model = load_silero_vad(onnx=False)
            except ImportError:
                raise ImportError(
                    "silero-vad is required for Neural VAD. Run: uv pip install silero-vad"
                )
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

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """
        Segment waveform using deep learning voice activity timestamps.
        """
        import torch
        from silero_vad import get_speech_timestamps

        model = self._get_model()

        # Silero VAD operates at 16000 Hz or 8000 Hz
        target_sampling_rate = 16000
        if sampling_rate != target_sampling_rate:
            gcd = np.gcd(sampling_rate, target_sampling_rate)
            up = target_sampling_rate // gcd
            down = sampling_rate // gcd
            resampled = scipy.signal.resample_poly(waveform, up, down).astype(np.float32)
        else:
            resampled = waveform.astype(np.float32)

        tensor_audio = torch.from_numpy(resampled)

        # vad_min_speech_ms (250ms) suppresses blips/clicks while keeping short words
        speech_timestamps = get_speech_timestamps(
            tensor_audio,
            model,
            threshold=self.threshold,
            sampling_rate=target_sampling_rate,
            min_speech_duration_ms=self.vad_min_speech_ms,
            max_speech_duration_s=float(self.max_length_ms / 1000.0),
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        ratio = sampling_rate / target_sampling_rate
        max_samples = int(sampling_rate * self.max_length_ms / 1000)
        min_samples = int(sampling_rate * self.min_length_ms / 1000)

        raw_chunks: List[Tuple[int, int]] = []
        for ts in speech_timestamps:
            s_orig = int(ts["start"] * ratio)
            e_orig = min(len(waveform), int(ts["end"] * ratio))
            raw_chunks.append((s_orig, e_orig))

        # Merge adjacent speech segments separated by tiny pauses (< min_silence_duration_ms)
        merged_chunks: List[Tuple[int, int]] = []
        for s_orig, e_orig in raw_chunks:
            if not merged_chunks:
                merged_chunks.append((s_orig, e_orig))
                continue

            prev_s, prev_e = merged_chunks[-1]
            gap = s_orig - prev_e
            combined_len = e_orig - prev_s

            if gap <= int(sampling_rate * self.min_silence_duration_ms / 1000) and combined_len <= max_samples:
                merged_chunks[-1] = (prev_s, e_orig)
            else:
                if s_orig < prev_e:
                    # Enforce zero overlap if speech padding overlapped across segments
                    mid = (prev_e + s_orig) // 2
                    merged_chunks[-1] = (prev_s, mid)
                    s_orig = mid
                merged_chunks.append((s_orig, e_orig))

        # Filter and subdivide long segments with balanced zero-crossing cuts
        final_chunks: List[Tuple[int, int]] = []
        for start, end in merged_chunks:
            chunk_len = end - start
            if chunk_len < min_samples:
                # Keep short utterances if they are at least 800ms (e.g. exclamations/short answers)
                if chunk_len < int(sampling_rate * 0.8):
                    continue

            if chunk_len <= max_samples:
                final_chunks.append((start, end))
                continue

            num_splits = int(np.ceil(chunk_len / max_samples))
            ideal_len = chunk_len // num_splits
            cur_p = start
            search_win = int(sampling_rate * 0.5)

            for s_idx in range(1, num_splits):
                target_p = start + s_idx * ideal_len
                clean_cut = self._find_clean_split_point(waveform, target_p, search_win)
                final_chunks.append((cur_p, clean_cut))
                cur_p = clean_cut
            final_chunks.append((cur_p, end))

        # Fallback if no speech detected
        if not final_chunks and len(waveform) >= min_samples:
            final_chunks.append((0, min(len(waveform), max_samples)))

        return final_chunks
