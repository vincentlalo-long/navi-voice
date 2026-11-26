"""
Orthogonal Vocal Expression & Semantic Category Classifier.
Decomposes utterance expression into three orthogonal axes:
1. delivery_mode (acoustic vocal delivery: normal / whisper_breathy / loud_strained)
2. text_cue (lexical keyword & punctuation cues from script)
3. auto_tags (soft multi-class distribution, valence/arousal, confidence & abstention)
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

TAGGER_VERSION: str = "2.0.0"

_PUNCTUATION_CUES: Tuple[str, ...] = ("！", "？", "…", "!", "?")

_DEFAULT_ACOUSTIC_HINTS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    (
        "high_emotion",
        {
            "pitch": ["high"],
            "speed": ["fast", "normal"],
            "energy": ["high"],
            "valence": -0.35,
            "punctuation": ["！", "!"],
        },
    ),
    (
        "defensive",
        {
            "pitch": ["high"],
            "speed": ["fast", "normal"],
            "energy": ["high", "normal"],
            "valence": -0.20,
            "punctuation": ["？", "?"],
        },
    ),
    (
        "gentle",
        {
            "pitch": ["low", "normal", "unvoiced"],
            "speed": ["slow", "normal"],
            "energy": ["low", "normal"],
            "valence": 0.45,
            "punctuation": ["…"],
        },
    ),
    (
        "intellectual",
        {
            "pitch": ["normal", "low"],
            "speed": ["normal"],
            "energy": ["normal", "low"],
            "valence": 0.05,
            "punctuation": [],
        },
    ),
    (
        "playful",
        {
            "pitch": ["high", "normal"],
            "speed": ["normal", "fast"],
            "energy": ["normal"],
            "valence": 0.50,
            "punctuation": ["？", "?"],
        },
    ),
)


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_pure_punctuation(token: str) -> bool:
    return bool(token) and all(ch in _PUNCTUATION_CUES or ch in ".,、。・〜~" for ch in token)


def _resolve_acoustic_hints(cat_key: str, cat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    key_lower = cat_key.lower()
    default_hints: Dict[str, Any] = {}
    for substr, hints in _DEFAULT_ACOUSTIC_HINTS:
        if substr in key_lower:
            default_hints = dict(hints)
            break
    explicit = cat_cfg.get("acoustic_hints")
    if isinstance(explicit, dict):
        merged = dict(default_hints)
        merged.update(explicit)
        return merged
    return default_hints


class CategoryClassifier:
    """
    Classifies utterances along orthogonal acoustic delivery, lexical cues,
    and soft-scored emotional expression with abstention on ambiguous inputs.
    """

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules or {}
        self.default_label = str(self.rules.get("default_label", "Neutral"))
        self._categories: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
        self._label_to_cfg: Dict[str, Dict[str, Any]] = {}

        for cat_key, cat_cfg in self.rules.items():
            if not isinstance(cat_cfg, dict):
                continue
            label = str(cat_cfg.get("label", cat_key.capitalize()))
            hints = _resolve_acoustic_hints(cat_key, cat_cfg)
            self._categories.append((cat_key, label, cat_cfg, hints))
            self._label_to_cfg[label] = cat_cfg

    @staticmethod
    def _determine_delivery_mode(
        raw_rms_db: Optional[float],
        prosody: Optional[Dict[str, Any]],
    ) -> str:
        # Never infer whisper from raw_rms_db alone when prosody is absent.
        if isinstance(prosody, dict):
            vf = _finite_float(prosody.get("voiced_fraction"))
            flatness = _finite_float(prosody.get("spectral_flatness"))
            centroid = _finite_float(prosody.get("spectral_centroid_hz"))
            bins = prosody.get("bins") if isinstance(prosody.get("bins"), dict) else {}
            energy_bin = bins.get("energy")

            is_quiet_unvoiced = (
                raw_rms_db is not None
                and raw_rms_db < -32.0
                and (
                    (vf is not None and vf < 0.35)
                    or (flatness is not None and flatness > 0.12)
                )
            )
            is_breathy_unvoiced = (
                vf is not None
                and 0.0 < vf < 0.22
                and flatness is not None
                and flatness > 0.15
            )
            if is_quiet_unvoiced or is_breathy_unvoiced:
                return "whisper_breathy"

            if (raw_rms_db is not None and raw_rms_db > -14.0) or (
                energy_bin == "high" and centroid is not None and centroid > 1800.0
            ):
                return "loud_strained"
        elif raw_rms_db is not None and raw_rms_db > -14.0:
            return "loud_strained"

        return "normal"

    @staticmethod
    def _extract_punctuation_cues(text: str) -> List[str]:
        return [cue for cue in _PUNCTUATION_CUES if cue in text]

    @staticmethod
    def _score_prosody(
        hints: Dict[str, Any], bins: Dict[str, Any]
    ) -> float:
        if not bins or not hints:
            return 0.0
        axes_checked = 0
        axis_score = 0.0
        for axis in ("pitch", "speed", "energy"):
            allowed = hints.get(axis)
            observed = bins.get(axis)
            if not isinstance(allowed, list) or not allowed or observed is None:
                continue
            axes_checked += 1
            if observed in allowed:
                # Normal bins represent baseline delivery and get half-weight so
                # neutral speech is not pulled into non-neutral categories.
                axis_score += 0.5 if observed == "normal" else 1.0
        if axes_checked == 0:
            return 0.0
        return _clip(axis_score / float(axes_checked), 0.0, 1.0)

    @staticmethod
    def _estimate_arousal(
        raw_rms_db: Optional[float],
        stats: Dict[str, Any],
        prosody: Optional[Dict[str, Any]],
    ) -> float:
        weighted_sum = 0.0
        weight_total = 0.0

        if raw_rms_db is not None:
            rms_norm = _clip((raw_rms_db - (-40.0)) / 30.0, 0.0, 1.0)
            weighted_sum += 0.45 * rms_norm
            weight_total += 0.45

        if isinstance(prosody, dict):
            f0_iqr = _finite_float(prosody.get("f0_iqr_hz"))
            if f0_iqr is not None:
                iqr_norm = _clip(f0_iqr / 80.0, 0.0, 1.0)
                weighted_sum += 0.35 * iqr_norm
                weight_total += 0.35

        mora_rate = None
        for source in (prosody, stats):
            if isinstance(source, dict):
                for key in ("effective_mora_rate", "effective_cps", "chars_per_sec"):
                    mora_rate = _finite_float(source.get(key))
                    if mora_rate is not None:
                        break
            if mora_rate is not None:
                break

        if mora_rate is not None:
            rate_norm = _clip((mora_rate - 2.0) / 10.0, 0.0, 1.0)
            weighted_sum += 0.20 * rate_norm
            weight_total += 0.20

        if weight_total <= 0.0:
            return 0.5
        return round(_clip(weighted_sum / weight_total, 0.0, 1.0), 4)

    def _estimate_valence(
        self,
        scores: Dict[str, float],
        prosody: Optional[Dict[str, Any]],
    ) -> float:
        valence = 0.0
        for _, label, _, hints in self._categories:
            cat_valence = _finite_float(hints.get("valence"))
            if cat_valence is not None:
                valence += scores.get(label, 0.0) * cat_valence

        if isinstance(prosody, dict):
            slope = _finite_float(prosody.get("f0_slope_hz_per_sec"))
            if slope is not None:
                valence += 0.10 * _clip(slope / 60.0, -1.0, 1.0)

        return round(_clip(valence, -1.0, 1.0), 4)

    def classify_rich(
        self,
        text: str,
        stats: Optional[Dict[str, Any]] = None,
        prosody: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify an utterance into delivery_mode, text_cue, and auto_tags."""
        safe_stats = stats if isinstance(stats, dict) else {}
        safe_text = text or ""
        raw_rms_db = _finite_float(safe_stats.get("raw_rms_db"))
        if raw_rms_db is None:
            raw_rms_db = _finite_float(safe_stats.get("rms_db"))

        delivery_mode = self._determine_delivery_mode(raw_rms_db, prosody)
        punctuation_cues = self._extract_punctuation_cues(safe_text)
        bins = (
            prosody.get("bins")
            if isinstance(prosody, dict) and isinstance(prosody.get("bins"), dict)
            else {}
        )

        # Axis 2: Lexical & punctuation cue matching
        best_lex_label: Optional[str] = None
        best_lex_hits: List[str] = []
        best_lex_rank: Tuple[int, float] = (0, 0.0)

        raw_scores: Dict[str, float] = {}
        lex_scores: Dict[str, float] = {}
        has_any_lexical_match = False

        for _, label, cat_cfg, hints in self._categories:
            keywords = cat_cfg.get("keywords", [])
            matched_kws: List[str] = []
            punct_kw_hit = False

            for kw in keywords:
                if not isinstance(kw, str) or not kw:
                    continue
                if re.search(re.escape(kw), safe_text):
                    if _is_pure_punctuation(kw):
                        punct_kw_hit = True
                    else:
                        matched_kws.append(kw)

            explicit_punct = cat_cfg.get(
                "punctuation_cues", hints.get("punctuation", [])
            )
            punct_match = 1 if (
                punct_kw_hit
                or any(p in punctuation_cues for p in explicit_punct)
            ) else 0

            if matched_kws or punct_match:
                has_any_lexical_match = True

            s_lex = min(1.0, 0.75 * len(matched_kws) + 0.25 * punct_match)
            if matched_kws and (len(matched_kws), s_lex) > best_lex_rank:
                best_lex_rank = (len(matched_kws), s_lex)
                best_lex_label = label
                best_lex_hits = matched_kws

            s_pros = self._score_prosody(hints, bins)
            lex_scores[label] = s_lex
            raw_scores[label] = 0.60 * s_lex + 0.40 * s_pros

        text_cue = {
            "matched_category": best_lex_label,
            "matched_keywords": best_lex_hits,
            "punctuation_cues": punctuation_cues,
        }

        # Handle empty emotion_rules safely
        if not self._categories:
            return {
                "delivery_mode": delivery_mode,
                "text_cue": text_cue,
                "auto_tags": {
                    "primary": self.default_label,
                    "secondary": [],
                    "scores": {self.default_label: 1.0},
                    "valence_est": 0.0,
                    "arousal_est": self._estimate_arousal(
                        raw_rms_db, safe_stats, prosody
                    ),
                    "confidence": 1.0,
                    "status": "suggestion_only",
                },
            }

        # Baseline score for default_label (Neutral/Dialogue)
        if not bins:
            is_normal_prosody = True
            normal_ratio = 1.0
        else:
            axis_vals = [bins.get(a) for a in ("pitch", "speed", "energy") if bins.get(a) is not None]
            normal_ratio = (
                sum(1 for v in axis_vals if v == "normal") / float(len(axis_vals))
                if axis_vals
                else 1.0
            )
            is_normal_prosody = bool(axis_vals) and all(v == "normal" for v in axis_vals)

        if not has_any_lexical_match and is_normal_prosody:
            neutral_raw = 0.35
        else:
            neutral_raw = 0.10 * normal_ratio

        all_raw: Dict[str, float] = dict(raw_scores)
        if self.default_label not in all_raw:
            all_raw[self.default_label] = neutral_raw
        else:
            all_raw[self.default_label] = max(all_raw[self.default_label], neutral_raw)

        total_raw = sum(all_raw.values())
        if total_raw <= 1e-12:
            all_raw[self.default_label] = 1.0
            total_raw = 1.0

        scores = {k: round(v / total_raw, 4) for k, v in all_raw.items()}
        # Keep sum(scores) normalized to 1.0 after rounding
        rounding_diff = round(1.0 - sum(scores.values()), 4)
        if rounding_diff != 0.0:
            max_key = max(scores, key=lambda k: scores[k])
            scores[max_key] = round(max(0.0, scores[max_key] + rounding_diff), 4)

        sorted_all = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        p_1 = sorted_all[0][1]
        p_2 = sorted_all[1][1] if len(sorted_all) > 1 else 0.0
        confidence = round(_clip(p_1 + 0.5 * (p_1 - p_2), 0.0, 1.0), 4)

        # Abstention checks on non-neutral categories
        non_neutral_items = [
            (k, raw_scores[k], scores[k])
            for k in raw_scores
            if k != self.default_label
        ]
        non_neutral_items.sort(key=lambda item: item[1], reverse=True)

        all_non_neutral_lex_zero = all(
            v == 0.0 for k, v in lex_scores.items() if k != self.default_label
        )
        strong_contradiction = (
            bins.get("pitch") == "high" and bins.get("energy") == "low"
        )
        # Unlexicalized high-pitch + low-energy prosody is ambiguous across all
        # speed bins (including slow), so abstain instead of guessing from prosody alone.
        abstain_contradiction = strong_contradiction and all_non_neutral_lex_zero

        abstain_tie = False
        if len(non_neutral_items) >= 2:
            _, s_top1, p_top1 = non_neutral_items[0]
            _, s_top2, p_top2 = non_neutral_items[1]
            if s_top1 > 0.25 and s_top2 > 0.25 and abs(p_top1 - p_top2) < 0.08:
                abstain_tie = True

        if abstain_contradiction or abstain_tie:
            status = "abstained_uncertain"
            primary = "Uncertain"
        else:
            status = "suggestion_only"
            primary = sorted_all[0][0]

        secondary = [k for k, prob in sorted_all if k != primary and prob >= 0.22]
        arousal_est = self._estimate_arousal(raw_rms_db, safe_stats, prosody)
        valence_est = self._estimate_valence(scores, prosody)

        return {
            "delivery_mode": delivery_mode,
            "text_cue": text_cue,
            "auto_tags": {
                "primary": primary,
                "secondary": secondary,
                "scores": scores,
                "valence_est": valence_est,
                "arousal_est": arousal_est,
                "confidence": confidence,
                "status": status,
            },
        }

    def classify(self, text: str, stats: Dict[str, float]) -> str:
        """Backward-compatible single-label classifier."""
        if not text:
            return "Unknown"

        safe_stats = stats if isinstance(stats, dict) else {}
        prosody = safe_stats.get("prosody")
        rich = self.classify_rich(
            text,
            safe_stats,
            prosody=prosody if isinstance(prosody, dict) else None,
        )
        primary = rich["auto_tags"]["primary"]
        if primary == "Uncertain":
            return self.default_label

        matched_category = rich["text_cue"]["matched_category"]
        if matched_category is not None and matched_category == primary:
            cat_cfg = self._label_to_cfg.get(matched_category, {})
            whisper_thresh = _finite_float(cat_cfg.get("whisper_threshold_db"))
            rms_db = _finite_float(safe_stats.get("raw_rms_db"))
            if rms_db is None:
                rms_db = _finite_float(safe_stats.get("rms_db"))
            if (
                whisper_thresh is not None
                and rms_db is not None
                and rms_db < whisper_thresh
            ):
                return str(
                    cat_cfg.get("whisper_label", f"{matched_category}/Whisper")
                )

        return primary
