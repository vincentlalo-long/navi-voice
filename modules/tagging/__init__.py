"""
Semantic and acoustic category classifier for TTS dialogue datasets.
"""

from .classifier import TAGGER_VERSION, CategoryClassifier
from .prosody import (
    PROSODY_EXTRACTOR_VERSION,
    assign_speaker_prosody_bins,
    extract_prosody_features,
)

__all__ = [
    "TAGGER_VERSION",
    "CategoryClassifier",
    "PROSODY_EXTRACTOR_VERSION",
    "extract_prosody_features",
    "assign_speaker_prosody_bins",
]
