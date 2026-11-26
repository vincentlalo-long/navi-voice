"""Hardened Faster-Whisper ASR wrapper collecting confidence and timing diagnostics."""

import gc
import math
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .text_norm import canonicalize_japanese_text, compute_speaking_rates

# Left as None until instantiation (unless patched in tests) so CUDA libs are
# preloaded via modules.cuda_utils before CTranslate2 initializes.
WhisperModel: Optional[Any] = None


def _compute_utf8_compression_ratio(text: str) -> float:
    """Compute UTF-8 byte length divided by zlib-compressed byte length."""
    if not text:
        return 0.0
    raw_bytes = text.encode("utf-8")
    compressed = zlib.compress(raw_bytes)
    return len(raw_bytes) / max(1, len(compressed))


def _compute_token_weighted_avg_logprob(segments: List[Any]) -> float:
    """Compute token-weighted mean avg_logprob across segments (-99.0 if empty)."""
    if not segments:
        return -99.0

    weighted_sum = 0.0
    total_weight = 0.0
    for seg in segments:
        val = getattr(seg, "avg_logprob", None)
        if val is None:
            continue
        try:
            logprob = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(logprob):
            continue

        tokens = getattr(seg, "tokens", None)
        weight = float(len(tokens)) if isinstance(tokens, (list, tuple)) and len(tokens) > 0 else 1.0
        weighted_sum += logprob * weight
        total_weight += weight

    if total_weight <= 0.0:
        return -99.0
    return weighted_sum / total_weight


def _compute_mean_no_speech_prob(segments: List[Any]) -> float:
    """Compute mean no_speech_prob across segments (1.0 if empty)."""
    if not segments:
        return 1.0

    probs: List[float] = []
    for seg in segments:
        val = getattr(seg, "no_speech_prob", None)
        if val is None:
            continue
        try:
            prob = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(prob):
            probs.append(prob)

    if not probs:
        return 1.0
    return sum(probs) / len(probs)


class HardenedWhisperTranscriber:
    """Wrapper around faster_whisper.WhisperModel with hardened decoding & diagnostics."""

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str = "ja",
        beam_size: int = 5,
        model: Optional[Any] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = int(beam_size)

        if model is not None:
            self.model: Optional[Any] = model
            return

        if self.device == "cuda":
            import modules.cuda_utils as cuda_utils

            cuda_utils.setup_cuda_env()

        model_cls = WhisperModel
        if model_cls is None:
            from faster_whisper import WhisperModel as _FasterWhisperModel

            model_cls = _FasterWhisperModel

        actual_compute = (
            "int8" if (self.device == "cpu" and self.compute_type == "float16") else self.compute_type
        )
        try:
            self.model = model_cls(
                self.model_size,
                device=self.device,
                compute_type=actual_compute,
                local_files_only=True,
            )
        except Exception:
            self.model = model_cls(
                self.model_size,
                device=self.device,
                compute_type=actual_compute,
                local_files_only=False,
            )

    def transcribe_file(self, audio_path: Path, duration_sec: float) -> Dict[str, Any]:
        """Transcribe an audio slice and return full Schema v2.0 asr_diagnostics."""
        if self.model is None:
            raise RuntimeError("HardenedWhisperTranscriber model has been released.")

        segments_iter, _ = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=(0.0, 0.2, 0.4),
        )
        segments = list(segments_iter)

        raw_text = "".join((getattr(seg, "text", "") or "") for seg in segments).strip()
        canonical_text = canonicalize_japanese_text(raw_text)
        avg_logprob = _compute_token_weighted_avg_logprob(segments)
        no_speech_prob = _compute_mean_no_speech_prob(segments)
        compression_ratio = _compute_utf8_compression_ratio(raw_text)

        word_probs: List[float] = []
        word_starts: List[float] = []
        word_ends: List[float] = []
        for seg in segments:
            words = getattr(seg, "words", None) or []
            for word in words:
                prob = getattr(word, "probability", None)
                if prob is not None:
                    try:
                        prob_f = float(prob)
                        if math.isfinite(prob_f):
                            word_probs.append(prob_f)
                    except (TypeError, ValueError):
                        pass

                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if start is not None and end is not None:
                    try:
                        start_f = float(start)
                        end_f = float(end)
                        if math.isfinite(start_f) and math.isfinite(end_f):
                            word_starts.append(start_f)
                            word_ends.append(end_f)
                    except (TypeError, ValueError):
                        pass

        mean_word_prob = sum(word_probs) / len(word_probs) if word_probs else 0.0

        if word_starts and word_ends:
            speech_duration_sec = max(0.2, max(word_ends) - min(word_starts))
        else:
            fallback_dur = float(duration_sec) if math.isfinite(float(duration_sec)) else 0.2
            speech_duration_sec = max(0.2, fallback_dur)

        rates = compute_speaking_rates(canonical_text, speech_duration_sec)

        return {
            "raw_text": raw_text,
            "canonical_text": canonical_text,
            "avg_logprob": round(avg_logprob, 4),
            "no_speech_prob": round(no_speech_prob, 4),
            "compression_ratio": round(compression_ratio, 4),
            "mean_word_prob": round(mean_word_prob, 4),
            "speech_duration_sec": round(float(rates["speech_duration_sec"]), 4),
            "clean_char_count": int(rates["clean_char_count"]),
            "estimated_mora_count": float(rates["estimated_mora_count"]),
            "effective_cps": round(float(rates["effective_cps"]), 4),
            "effective_mora_rate": round(float(rates["effective_mora_rate"]), 4),
            "hallucination_flags": [],
        }

    def release(self) -> None:
        """Release the Whisper model and free CUDA/CPU memory."""
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
