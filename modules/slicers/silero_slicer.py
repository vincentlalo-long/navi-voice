"""
Neural Voice Activity Detection Slicer powered by Silero VAD (Deep Learning Strategy).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.signal
from .base import BaseSlicer, merge_short_segments, validate_intervals


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
                    model = load_silero_vad(onnx=False)
                    if self.device != "cpu":
                        try:
                            model = model.to(self.device)
                        except Exception:
                            self.device = "cpu"
                    self._model = model
            except ImportError:
                raise ImportError(
                    "silero-vad is required for Neural VAD. Run: uv pip install silero-vad"
                )
        return self._model

    def slice(self, waveform: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """
        Segment waveform using deep learning voice activity timestamps.
        """
        if len(waveform) == 0:
            return []

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

        tensor_audio = torch.from_numpy(resampled).to(self.device)

        # ``get_speech_timestamps`` pads each side by speech_pad_ms. Merging
        # those already-padded intervals lowered the effective pause threshold
        # to min_silence + 2 * pad (~1.2 s). Ask for raw timestamps, merge on
        # real speech gaps, and apply our own padding afterwards.
        speech_timestamps = get_speech_timestamps(
            tensor_audio,
            model,
            threshold=self.threshold,
            sampling_rate=target_sampling_rate,
            min_speech_duration_ms=self.vad_min_speech_ms,
            max_speech_duration_s=float(self.max_length_ms / 1000.0),
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=0,
            return_seconds=False,
        )

        ratio = sampling_rate / target_sampling_rate
        total = len(waveform)
        max_samples = int(sampling_rate * self.max_length_ms / 1000)
        min_interval = int(sampling_rate * self.min_silence_duration_ms / 1000)
        pad_samples = int(sampling_rate * self.speech_pad_ms / 1000)

        raw_chunks: List[Tuple[int, int]] = []
        for ts in speech_timestamps:
            s_orig = max(0, int(ts["start"] * ratio))
            e_orig = min(total, int(ts["end"] * ratio))
            if e_orig > s_orig:
                raw_chunks.append((s_orig, e_orig))

        if not raw_chunks:
            return []

        # Merge only real pauses shorter than min_interval.
        merged_raw: List[Tuple[int, int]] = []
        for s_orig, e_orig in raw_chunks:
            if merged_raw:
                prev_s, prev_e = merged_raw[-1]
                if s_orig - prev_e <= min_interval and e_orig - prev_s <= max_samples:
                    merged_raw[-1] = (prev_s, e_orig)
                    continue
            merged_raw.append((s_orig, e_orig))

        # Apply padding once, then resolve collisions at the midpoint without
        # gluing two sentences together.
        padded = [
            (max(0, s - pad_samples), min(total, e + pad_samples))
            for s, e in merged_raw
        ]

        resolved: List[Tuple[int, int]] = []
        for s_orig, e_orig in padded:
            if resolved and s_orig < resolved[-1][1]:
                prev_s, prev_e = resolved[-1]
                mid = max(prev_s + 1, (prev_e + s_orig) // 2)
                resolved[-1] = (prev_s, min(mid, prev_e))
                s_orig = max(s_orig, resolved[-1][1])
            if e_orig > s_orig:
                resolved.append((s_orig, e_orig))

        # Preserve short exclamations detected by the VAD (250 ms floor).
        keep_floor = max(1, int(sampling_rate * self.vad_min_speech_ms / 1000))
        chunks = [(s, e) for s, e in resolved if (e - s) >= keep_floor]

        final_chunks: List[Tuple[int, int]] = []
        for start, end in chunks:
            final_chunks.extend(
                self.balanced_subdivide(
                    waveform,
                    sampling_rate,
                    start,
                    end,
                    max_samples,
                    lambda lo, hi: self.find_clean_split_in_range(waveform, lo, hi),
                )
            )

        final_chunks = merge_short_segments(
            final_chunks,
            min_length_samples=int(sampling_rate * self.min_length_ms / 1000),
            max_length_samples=max_samples,
            max_gap_samples=pad_samples * 2,
        )

        return validate_intervals(
            final_chunks,
            total_samples=total,
            sampling_rate=sampling_rate,
            max_length_samples=max_samples,
        )
