"""
Audio Slicing & VAD Strategies package.
"""

from .base import BaseSlicer
from .energy_slicer import EnergySlicer
from .silero_slicer import SileroSlicer
from .webrtc_slicer import WebRTCSlicer
from .whisper_slicer import WhisperXSlicer
from .hybrid_slicer import HybridSlicer
from .factory import get_slicer, SLICER_DISPATCH

__all__ = [
    "BaseSlicer",
    "EnergySlicer",
    "SileroSlicer",
    "WebRTCSlicer",
    "WhisperXSlicer",
    "HybridSlicer",
    "get_slicer",
    "SLICER_DISPATCH",
]
