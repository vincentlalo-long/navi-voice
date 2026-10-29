"""
Semantic & Acoustic Category Classifier.
Assigns emotional and prosody labels to dialogue segments based on
lexical pattern matching and acoustic energy profiles.
"""

import re
from typing import Any, Dict


class CategoryClassifier:
    """
    Classifies utterances into semantic or emotional categories based on
    configurable pattern matching rules and acoustic energy metrics.
    """

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules or {}
        self.default_label = self.rules.get("default_label", "Neutral")

    def classify(self, text: str, stats: Dict[str, float]) -> str:
        """Evaluate text and acoustic stats against loaded category rules."""
        if not text:
            return "Unknown"

        rms_db = float(stats.get("rms_db", 0.0))

        # First Pass: Semantic / Keyword Matching with acoustic sub-classification
        for cat_key, cat_cfg in self.rules.items():
            if not isinstance(cat_cfg, dict):
                continue

            keywords = cat_cfg.get("keywords", [])
            for kw in keywords:
                if re.search(re.escape(kw), text):
                    # Check for acoustic whisper modifier within matched category
                    whisper_thresh = cat_cfg.get("whisper_threshold_db")
                    if whisper_thresh is not None and rms_db < float(whisper_thresh):
                        return cat_cfg.get("whisper_label", f"{cat_key.capitalize()}/Whisper")
                    return cat_cfg.get("label", cat_key.capitalize())

        # Second Pass: Acoustic Fallback (e.g. general whisper if unclassified by text)
        for cat_key, cat_cfg in self.rules.items():
            if not isinstance(cat_cfg, dict):
                continue
            whisper_thresh = cat_cfg.get("whisper_threshold_db")
            if whisper_thresh is not None and rms_db < float(whisper_thresh):
                return cat_cfg.get("whisper_label", f"{cat_key.capitalize()}/Whisper")

        return self.default_label
