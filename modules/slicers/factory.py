"""
Factory for creating audio slicer instances via Dictionary Dispatch.
"""

from typing import Any, Dict, Optional, Type
from .base import BaseSlicer
from .energy_slicer import EnergySlicer
from .silero_slicer import SileroSlicer
from .webrtc_slicer import WebRTCSlicer
from .whisper_slicer import WhisperXSlicer

# Dictionary Dispatch Table: Map algorithm aliases directly to Slicer classes
SLICER_DISPATCH: Dict[str, Type[BaseSlicer]] = {
    # Energy-based RMS DSP Slicers
    "energy": EnergySlicer,
    "rms": EnergySlicer,
    "dsp": EnergySlicer,
    # Neural Deep Learning VAD Slicers (RNN LSTM)
    "silero": SileroSlicer,
    "neural": SileroSlicer,
    "vad": SileroSlicer,
    # WebRTC Gaussian Mixture Model (GMM Baseline)
    "webrtc": WebRTCSlicer,
    "webrtcvad": WebRTCSlicer,
    "gmm": WebRTCSlicer,
    # WhisperX / Faster-Whisper Semantic & Punctuation Alignment
    "whisper": WhisperXSlicer,
    "whisperx": WhisperXSlicer,
    "semantic": WhisperXSlicer,
}


def get_slicer(method: str = "energy", config: Optional[Dict[str, Any]] = None) -> BaseSlicer:
    """
    Factory function returning an instantiated BaseSlicer strategy via Dictionary Dispatch.

    Args:
        method: Slicing algorithm name or alias ('energy', 'silero', etc.).
        config: Optional dictionary containing algorithm hyperparameters.

    Returns:
        Configured BaseSlicer instance.

    Raises:
        ValueError: If the requested slicing method is not supported.
    """
    cfg = config or {}
    key = method.lower().strip()

    slicer_cls = SLICER_DISPATCH.get(key)
    if slicer_cls is None:
        valid_options = sorted(list(set(SLICER_DISPATCH.keys())))
        raise ValueError(
            f"Unsupported slicing method: '{method}'. Valid options are: {valid_options}"
        )

    return slicer_cls.from_config(cfg)
