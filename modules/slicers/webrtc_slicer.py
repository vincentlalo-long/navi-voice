"""
WebRTC VAD Audio Slicer Strategy (GMM Baseline).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.signal
from .base import BaseSlicer


class WebRTCSlicer(BaseSlicer):
    """
    Slices audio using WebRTC VAD (Gaussian Mixture Model baseline).
    Features:
      - Polyphase anti-aliased resampling to 16kHz for GMM evaluation.
      - Ring-buffer frame smoothing to prevent GMM frame chattering.
      - Zero-overlap midpoint splitting when pauses are short (< 2 * max_sil_kept).
      - Balanced zero-crossing subdivision for long segments (> max_length).
      - Inverse coordinate mapping back to original sampling rate.
    """

    def __init__(
        self,
        mode: int = 3,
        min_length_ms: int = 2000,
        max_length_ms: int = 12000,
        min_interval_ms: int = 400,
        hop_size_ms: int = 20,
        max_sil_kept_ms: int = 400,
        min_speech_ms: int = 800,
    ):
        """
        Args:
            mode: WebRTC VAD aggressiveness mode (0 = least aggressive, 3 = most aggressive).
            min_length_ms: Target minimum sentence length for TTS.
            max_length_ms: Maximum sentence duration before balanced subdivision.
            min_interval_ms: Minimum silence duration to trigger a split.
            hop_size_ms: Frame length in ms (must be 10, 20, or 30 ms for WebRTC VAD).
            max_sil_kept_ms: Padding added to boundaries.
            min_speech_ms: Minimum speech duration to keep short exclamations.
        """
        if hop_size_ms not in (10, 20, 30):
            hop_size_ms = 20

        self.mode = int(np.clip(mode, 0, 3))
        self.min_length_ms = min_length_ms
        self.max_length_ms = max_length_ms
        self.min_interval_ms = min_interval_ms
        self.hop_size_ms = hop_size_ms
        self.max_sil_kept_ms = max_sil_kept_ms
        self.min_speech_ms = min_speech_ms
        self._vad = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WebRTCSlicer":
        """Instantiate WebRTCSlicer from configuration dictionary."""
        return cls(
            mode=int(config.get("webrtc_mode", 3)),
            min_length_ms=int(config.get("min_length_ms", 2000)),
            max_length_ms=int(config.get("max_length_ms", 12000)),
            min_interval_ms=int(config.get("min_interval_ms", 400)),
            hop_size_ms=int(config.get("hop_size_ms", 20)),
            max_sil_kept_ms=int(config.get("max_sil_kept_ms", 400)),
            min_speech_ms=int(config.get("min_speech_ms", 800)),
        )

    def _get_vad(self):
        """Lazy loader for WebRTC VAD engine."""
        if self._vad is None:
            try:
                import webrtcvad
                self._vad = webrtcvad.Vad(self.mode)
            except ImportError:
                raise ImportError(
                    "webrtcvad is required for WebRTCSlicer. Run: uv pip install webrtcvad-wheels"
                )
        return self._vad

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
        """Slice waveform using WebRTC VAD frame decisions and smoothed windowing."""
        vad = self._get_vad()

        # WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz 16-bit PCM
        target_sr = 16000
        if sampling_rate != target_sr:
            gcd = np.gcd(sampling_rate, target_sr)
            up = target_sr // gcd
            down = sampling_rate // gcd
            resampled = scipy.signal.resample_poly(waveform, up, down).astype(np.float32)
        else:
            resampled = waveform.astype(np.float32)

        # Convert float32 [-1.0, 1.0] to int16 bytes
        int16_audio = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame_samples = int(target_sr * self.hop_size_ms / 1000)
        num_frames = len(int16_audio) // frame_samples

        if num_frames == 0:
            return [(0, len(waveform))]

        # Classify each frame
        frame_flags = []
        for i in range(num_frames):
            frame_bytes = int16_audio[i * frame_samples : (i + 1) * frame_samples].tobytes()
            is_speech = vad.is_speech(frame_bytes, target_sr)
            frame_flags.append(is_speech)

        # Smooth frame flags using a sliding window (hysteresis / ring buffer)
        # 10 frames window (~200ms)
        win_size = max(5, int(200 / self.hop_size_ms))
        smoothed = np.zeros(num_frames, dtype=bool)
        state = False

        for i in range(num_frames):
            w_start = max(0, i - win_size + 1)
            window = frame_flags[w_start : i + 1]
            voiced_ratio = sum(window) / len(window)

            if not state:
                if voiced_ratio >= 0.60:
                    state = True
            else:
                if voiced_ratio <= 0.20:
                    state = False
            smoothed[i] = state

        # Find continuous speech intervals
        speech_intervals = []
        in_speech = False
        start_frame = 0

        for i, val in enumerate(smoothed):
            if val and not in_speech:
                in_speech = True
                start_frame = i
            elif not val and in_speech:
                in_speech = False
                speech_intervals.append((start_frame * frame_samples, i * frame_samples))
        if in_speech:
            speech_intervals.append((start_frame * frame_samples, len(resampled)))

        if not speech_intervals:
            return []

        # Merge intervals separated by small pauses (< min_interval_ms)
        min_sil_samples = int(target_sr * self.min_interval_ms / 1000)
        pad_samples = int(target_sr * self.max_sil_kept_ms / 1000)
        max_samples_16k = int(target_sr * self.max_length_ms / 1000)
        min_speech_16k = int(target_sr * self.min_speech_ms / 1000)

        padded_intervals = []
        for s, e in speech_intervals:
            s_pad = max(0, s - pad_samples)
            e_pad = min(len(resampled), e + pad_samples)
            padded_intervals.append((s_pad, e_pad))

        # Collision resolution & merging
        merged = []
        for s, e in padded_intervals:
            if not merged:
                merged.append((s, e))
                continue

            prev_s, prev_e = merged[-1]
            gap = s - prev_e
            combined_len = e - prev_s

            if gap <= min_sil_samples and combined_len <= max_samples_16k:
                merged[-1] = (prev_s, e)
            else:
                if s < prev_e:
                    # Enforce zero-overlap midpoint split on padding collision
                    mid = (prev_e + s) // 2
                    merged[-1] = (prev_s, mid)
                    s = mid
                merged.append((s, e))

        # Map back to original sampling rate and apply balanced zero-crossing cuts
        ratio = sampling_rate / target_sr
        max_samples = int(sampling_rate * self.max_length_ms / 1000)
        min_speech_orig = int(sampling_rate * self.min_speech_ms / 1000)

        final_chunks: List[Tuple[int, int]] = []
        for s_16k, e_16k in merged:
            chunk_len_16k = e_16k - s_16k
            if chunk_len_16k < min_speech_16k:
                continue

            start_orig = int(s_16k * ratio)
            end_orig = min(len(waveform), int(e_16k * ratio))
            chunk_dur = end_orig - start_orig

            if chunk_dur <= max_samples:
                final_chunks.append((start_orig, end_orig))
                continue

            # Balanced k-way split
            num_splits = int(np.ceil(chunk_dur / max_samples))
            ideal_len = chunk_dur // num_splits
            cur_p = start_orig
            search_win = int(sampling_rate * 0.5)

            for s_idx in range(1, num_splits):
                target_p = start_orig + s_idx * ideal_len
                clean_cut = self._find_clean_split_point(waveform, target_p, search_win)
                final_chunks.append((cur_p, clean_cut))
                cur_p = clean_cut
            final_chunks.append((cur_p, end_orig))

        return final_chunks
