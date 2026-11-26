"""Japanese text canonicalization, mora estimation, and ASR hallucination detectors."""

import math
import re
import unicodedata
from typing import Dict, Optional, Sequence, Tuple

DEFAULT_BOH_SUBSTRINGS: Tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録",
    "高評価",
    "字幕:",
    "字幕：",
    "翻訳:",
    "翻訳：",
    "サブタイトル",
    "MBCニュース",
    "Amara.org",
    "Thank you for watching",
    "Subtitles by",
    "提供:",
)

_SUBTITLE_TAG_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"【.*?】"),
    re.compile(r"\[.*?\]"),
    re.compile(r"♪.*?♪"),
    re.compile(r"\(\s*(?:笑|ため息|咳払い|泣|拍手|音楽|BGM|息)\s*\)"),
)

_ELLIPSIS_PATTERN = re.compile(r"(?:。{2,}|\.{2,}|…{2,})")
_EXCLAMATION_PATTERN = re.compile(r"[!！]+")
_QUESTION_PATTERN = re.compile(r"[?？]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_SMALL_KANA_NO_MORA = frozenset("ゃゅょぁぃぅぇぉャュョァィゥェォゎヮ")


def _strip_control_chars(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if ch in ("\t", "\n", "\r")
        or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def _extract_phonetic_chars(text: str) -> str:
    """Return only spoken/phonetic characters (exclude P*, S*, Z*, C* and whitespace)."""
    if not text:
        return ""
    return "".join(
        ch
        for ch in text
        if not ch.isspace()
        and unicodedata.category(ch)[0] not in ("P", "S", "Z", "C")
    )


def canonicalize_japanese_text(text: str) -> str:
    """Normalize Japanese transcript text and strip non-speech subtitle tags."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _strip_control_chars(normalized)

    for pattern in _SUBTITLE_TAG_PATTERNS:
        normalized = pattern.sub("", normalized)

    normalized = _ELLIPSIS_PATTERN.sub("…", normalized)
    normalized = _EXCLAMATION_PATTERN.sub("！", normalized)
    normalized = _QUESTION_PATTERN.sub("？", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def count_japanese_phonetic_units(canonical_text: str) -> Tuple[int, float]:
    """Count clean phonetic characters and estimate Japanese mora count."""
    clean_text = _extract_phonetic_chars(canonical_text)
    clean_char_count = len(clean_text)
    if clean_char_count == 0:
        return 0, 0.0

    mora_count = 0.0
    for ch in clean_text:
        if ch in _SMALL_KANA_NO_MORA:
            continue
        code = ord(ch)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or ch == "ー":
            mora_count += 1.0
        elif (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            mora_count += 1.8
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            mora_count += 1.0
        else:
            mora_count += 1.0

    return clean_char_count, round(mora_count, 4)


def compute_speaking_rates(
    canonical_text: str,
    speech_duration_sec: float,
) -> Dict[str, float]:
    """Compute effective characters/sec and morae/sec with a 0.2s duration floor."""
    clean_char_count, estimated_mora_count = count_japanese_phonetic_units(
        canonical_text
    )
    if not math.isfinite(speech_duration_sec) or speech_duration_sec < 0.2:
        clamped_duration = 0.2
    else:
        clamped_duration = float(speech_duration_sec)

    return {
        "clean_char_count": clean_char_count,
        "estimated_mora_count": estimated_mora_count,
        "speech_duration_sec": clamped_duration,
        "effective_cps": clean_char_count / clamped_duration,
        "effective_mora_rate": estimated_mora_count / clamped_duration,
    }


def detect_boh_hallucination(
    raw_text: str,
    blacklist: Sequence[str] = DEFAULT_BOH_SUBSTRINGS,
) -> Optional[str]:
    """Return the matched blacklist substring if raw_text contains a BOH hallucination."""
    if not raw_text:
        return None

    raw_lower = raw_text.lower()
    nfkc_lower = unicodedata.normalize("NFKC", raw_text).lower()
    for entry in blacklist:
        if not entry:
            continue
        entry_lower = entry.lower()
        entry_nfkc_lower = unicodedata.normalize("NFKC", entry).lower()
        if entry_lower in raw_lower or entry_nfkc_lower in nfkc_lower:
            return entry
    return None


def detect_repetition_anomaly(
    canonical_text: str,
    max_consecutive_repeats: int = 4,
) -> bool:
    """Return True if any 1-4 char phonetic unit repeats > max_consecutive_repeats times."""
    if max_consecutive_repeats < 1:
        raise ValueError("max_consecutive_repeats must be >= 1")
    clean_text = _extract_phonetic_chars(canonical_text)
    if not clean_text:
        return False
    pattern = re.compile(rf"(.{{1,4}})\1{{{max_consecutive_repeats},}}")
    return pattern.search(clean_text) is not None
