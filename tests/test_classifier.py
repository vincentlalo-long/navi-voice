import json
import unittest
from pathlib import Path
from modules.classifier import CategoryClassifier as LegacyShimClassifier
from modules.tagging.classifier import TAGGER_VERSION, CategoryClassifier


PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "profiles" / "profile.example.json"


class TestClassifier(unittest.TestCase):
    def setUp(self):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            self.rules = json.load(f)["emotion_rules"]
        self.clf = CategoryClassifier(self.rules)

    def test_legacy_shim_stays_identical(self):
        self.assertIs(LegacyShimClassifier, CategoryClassifier)
        self.assertEqual(TAGGER_VERSION, "2.0.0")

    def test_quiet_modal_speech_does_not_trigger_false_gentle_whisper_and_ambiguous_scores_abstain(self):
        # Quiet analytical speech (-35 dB) with healthy modal voicing (0.75) and no gentle keywords
        res = self.clf.classify_rich(
            "この数値は昨日と同じ結果を示している。",
            stats={"raw_rms_db": -35.0},
            prosody={
                "voiced_fraction": 0.75,
                "spectral_flatness": 0.03,
                "spectral_centroid_hz": 1200.0,
                "f0_iqr_hz": 22.0,
                "f0_slope_hz_per_sec": -5.0,
                "bins": {"pitch": "normal", "speed": "normal", "energy": "low"},
            },
        )
        self.assertEqual(res["delivery_mode"], "normal")
        self.assertNotEqual(res["auto_tags"]["primary"], "Gentle/Whisper")
        self.assertNotEqual(
            self.clf.classify("この数値は昨日と同じ結果を示している。", {"raw_rms_db": -35.0}),
            "Gentle/Whisper",
        )
        self.assertNotEqual(
            self.clf.classify(
                "この数値は昨日と同じ結果を示している。",
                {
                    "raw_rms_db": -35.0,
                    "prosody": {
                        "voiced_fraction": 0.75,
                        "spectral_flatness": 0.03,
                        "spectral_centroid_hz": 1200.0,
                        "f0_iqr_hz": 22.0,
                        "f0_slope_hz_per_sec": -5.0,
                        "bins": {"pitch": "normal", "speed": "normal", "energy": "low"},
                    },
                },
            ),
            "Gentle/Whisper",
        )

        # True acoustic whisper (-36 dB + low voiced_fraction 0.18 + high flatness 0.16)
        whisper_res = self.clf.classify_rich(
            "聞こえるかしら",
            stats={"raw_rms_db": -36.0},
            prosody={
                "voiced_fraction": 0.18,
                "spectral_flatness": 0.16,
                "spectral_centroid_hz": 2400.0,
                "f0_iqr_hz": 10.0,
                "f0_slope_hz_per_sec": 0.0,
                "bins": {"pitch": "unvoiced", "speed": "slow", "energy": "low"},
            },
        )
        self.assertEqual(whisper_res["delivery_mode"], "whisper_breathy")

        # Conflicting lexical cues (one keyword from defensive_flustered + one from gentle_soft)
        tie_res = self.clf.classify_rich(
            "バカ、ありがとう",
            stats={"raw_rms_db": -22.0},
            prosody={
                "voiced_fraction": 0.65,
                "spectral_flatness": 0.04,
                "spectral_centroid_hz": 1500.0,
                "f0_iqr_hz": 35.0,
                "f0_slope_hz_per_sec": 0.0,
                "bins": {"pitch": "normal", "speed": "normal", "energy": "normal"},
            },
        )
        self.assertEqual(tie_res["auto_tags"]["status"], "abstained_uncertain")
        self.assertEqual(tie_res["auto_tags"]["primary"], "Uncertain")
        self.assertAlmostEqual(sum(tie_res["auto_tags"]["scores"].values()), 1.0, places=3)

    def test_clear_keyword_and_prosody_alignment_produces_high_confidence_suggestion(self):
        res = self.clf.classify_rich(
            "べ、別にあんたのためじゃないわよ！",
            stats={"raw_rms_db": -15.0},
            prosody={
                "voiced_fraction": 0.72,
                "spectral_flatness": 0.04,
                "spectral_centroid_hz": 1750.0,
                "f0_iqr_hz": 68.0,
                "f0_slope_hz_per_sec": 35.0,
                "bins": {"pitch": "high", "speed": "fast", "energy": "high"},
            },
        )
        self.assertEqual(res["auto_tags"]["primary"], "Defensive/Flustered")
        self.assertEqual(res["auto_tags"]["status"], "suggestion_only")
        self.assertGreater(res["auto_tags"]["confidence"], 0.55)
        self.assertGreater(res["auto_tags"]["arousal_est"], 0.5)

    def test_empty_emotion_rules_and_absent_prosody(self):
        empty_clf = CategoryClassifier({})
        res = empty_clf.classify_rich("普通の会話です。", stats={"raw_rms_db": -36.0})
        self.assertEqual(res["delivery_mode"], "normal")
        self.assertEqual(res["auto_tags"]["primary"], "Neutral")
        self.assertEqual(res["auto_tags"]["status"], "suggestion_only")
        self.assertEqual(res["auto_tags"]["scores"], {"Neutral": 1.0})
        self.assertGreaterEqual(res["auto_tags"]["arousal_est"], 0.0)
        self.assertLessEqual(res["auto_tags"]["arousal_est"], 1.0)
        self.assertEqual(empty_clf.classify("普通の会話です。", {"raw_rms_db": -36.0}), "Neutral")

        # Never classify whisper from raw_rms_db alone when prosody is absent
        no_prosody_res = self.clf.classify_rich("普通の会話です。", stats={"raw_rms_db": -40.0}, prosody=None)
        self.assertEqual(no_prosody_res["delivery_mode"], "normal")

    def test_delivery_mode_loud_strained_and_unvoiced_breathy(self):
        loud_rms = self.clf.classify_rich(
            "やめて！",
            stats={"raw_rms_db": -12.5},
            prosody={"voiced_fraction": 0.7, "spectral_flatness": 0.05, "spectral_centroid_hz": 1400.0},
        )
        self.assertEqual(loud_rms["delivery_mode"], "loud_strained")

        loud_centroid = self.clf.classify_rich(
            "ちょっと",
            stats={"raw_rms_db": -16.0},
            prosody={
                "voiced_fraction": 0.65,
                "spectral_flatness": 0.05,
                "spectral_centroid_hz": 1950.0,
                "bins": {"pitch": "high", "speed": "normal", "energy": "high"},
            },
        )
        self.assertEqual(loud_centroid["delivery_mode"], "loud_strained")

        breathy_unvoiced = self.clf.classify_rich(
            "あの…",
            stats={"raw_rms_db": -28.0},
            prosody={"voiced_fraction": 0.15, "spectral_flatness": 0.18, "spectral_centroid_hz": 1900.0},
        )
        self.assertEqual(breathy_unvoiced["delivery_mode"], "whisper_breathy")

    def test_prosody_contradiction_abstains_and_legacy_whisper_lexical_guard(self):
        contradiction = self.clf.classify_rich(
            "今日は晴れです。",
            stats={"raw_rms_db": -30.0},
            prosody={
                "voiced_fraction": 0.6,
                "spectral_flatness": 0.05,
                "spectral_centroid_hz": 1400.0,
                "f0_iqr_hz": 20.0,
                "f0_slope_hz_per_sec": 0.0,
                "bins": {"pitch": "high", "speed": "normal", "energy": "low"},
            },
        )
        self.assertEqual(contradiction["auto_tags"]["status"], "abstained_uncertain")
        self.assertEqual(contradiction["auto_tags"]["primary"], "Uncertain")
        self.assertEqual(self.clf.classify("今日は晴れです。", {"raw_rms_db": -30.0, "prosody": {"bins": {"pitch": "high", "speed": "normal", "energy": "low"}}}), "Neutral/Dialogue")

        # Legacy classify only appends /Whisper when lexically matched AND rms_db < whisper_threshold_db
        self.assertEqual(
            self.clf.classify("ありがとう", {"raw_rms_db": -35.0}),
            "Gentle/Whisper",
        )
        self.assertEqual(
            self.clf.classify("ありがとう", {"raw_rms_db": -20.0}),
            "Gentle/Affectionate",
        )

    def test_legacy_profile_without_acoustic_hints_and_partial_override_merges_punctuation(self):
        legacy_rules = {
            "high_emotion": {
                "label": "High-Emotion/Agitated",
                "keywords": ["やめて"],
            },
            "defensive_flustered": {
                "label": "Defensive/Flustered",
                "keywords": ["バカ"],
            },
            "gentle_soft": {
                "label": "Gentle/Affectionate",
                "keywords": ["大丈夫"],
                "whisper_threshold_db": -32.0,
                "whisper_label": "Gentle/Whisper",
            },
            "intellectual_analytical": {
                "label": "Intellectual/Analytical",
                "keywords": ["理論"],
            },
            "playful_teasing": {
                "label": "Playful/Teasing",
                "keywords": ["面白い"],
            },
            "default_label": "Neutral/Dialogue",
        }
        legacy_clf = CategoryClassifier(legacy_rules)

        gentle_res = legacy_clf.classify_rich(
            "大丈夫…",
            stats={"raw_rms_db": -28.0},
            prosody={
                "voiced_fraction": 0.7,
                "spectral_flatness": 0.03,
                "f0_iqr_hz": 18.0,
                "f0_slope_hz_per_sec": 5.0,
                "bins": {"pitch": "low", "speed": "slow", "energy": "low"},
            },
        )
        self.assertEqual(gentle_res["auto_tags"]["primary"], "Gentle/Affectionate")
        self.assertGreater(gentle_res["auto_tags"]["valence_est"], 0.0)
        self.assertEqual(gentle_res["text_cue"]["punctuation_cues"], ["…"])

        high_res = legacy_clf.classify_rich(
            "やめて！",
            stats={"raw_rms_db": -15.0},
            prosody={
                "voiced_fraction": 0.75,
                "spectral_flatness": 0.04,
                "f0_iqr_hz": 65.0,
                "f0_slope_hz_per_sec": -10.0,
                "bins": {"pitch": "high", "speed": "fast", "energy": "high"},
            },
        )
        self.assertEqual(high_res["auto_tags"]["primary"], "High-Emotion/Agitated")
        self.assertLess(high_res["auto_tags"]["valence_est"], 0.0)

        intel_res = legacy_clf.classify_rich(
            "理論通りね",
            stats={"raw_rms_db": -22.0},
            prosody={"bins": {"pitch": "normal", "speed": "normal", "energy": "normal"}},
        )
        self.assertEqual(intel_res["auto_tags"]["primary"], "Intellectual/Analytical")

        playful_res = legacy_clf.classify_rich(
            "面白い？",
            stats={"raw_rms_db": -21.0},
            prosody={"bins": {"pitch": "high", "speed": "normal", "energy": "normal"}},
        )
        self.assertEqual(playful_res["auto_tags"]["primary"], "Playful/Teasing")

        # Partial acoustic_hints (without "punctuation") still inherits default "…" cue
        partial_rules = {
            "other_cat": {
                "label": "Other/Cat",
                "keywords": ["確認"],
                "acoustic_hints": {"pitch": ["normal"], "speed": ["normal"], "energy": ["normal"], "valence": 0.0},
            },
            "gentle_soft": {
                "label": "Gentle/Affectionate",
                "keywords": ["大丈夫"],
                "acoustic_hints": {"pitch": ["normal"], "speed": ["normal"], "energy": ["normal"], "valence": 0.4},
            },
            "default_label": "Neutral/Dialogue",
        }
        partial_clf = CategoryClassifier(partial_rules)
        tie_broken = partial_clf.classify_rich(
            "確認、大丈夫…",
            stats={"raw_rms_db": -22.0},
            prosody={"bins": {"pitch": "normal", "speed": "normal", "energy": "normal"}},
        )
        self.assertEqual(tie_broken["auto_tags"]["primary"], "Gentle/Affectionate")
        self.assertEqual(tie_broken["text_cue"]["matched_category"], "Gentle/Affectionate")
        self.assertEqual(tie_broken["text_cue"]["punctuation_cues"], ["…"])

    def test_rich_output_fields_and_punct_tie_breaking(self):
        res = self.clf.classify_rich(
            "べ、別にあんたのためじゃないわよ！",
            stats={"raw_rms_db": -15.0},
            prosody={
                "voiced_fraction": 0.72,
                "spectral_flatness": 0.04,
                "spectral_centroid_hz": 1750.0,
                "f0_iqr_hz": 68.0,
                "f0_slope_hz_per_sec": 35.0,
                "bins": {"pitch": "high", "speed": "fast", "energy": "high"},
            },
        )
        self.assertEqual(res["text_cue"]["matched_category"], "Defensive/Flustered")
        self.assertEqual(res["text_cue"]["matched_keywords"], ["べ、別に", "あんた"])
        self.assertEqual(res["text_cue"]["punctuation_cues"], ["！"])
        self.assertIn("High-Emotion/Agitated", res["auto_tags"]["secondary"])
        self.assertLess(res["auto_tags"]["valence_est"], 0.0)

        # Earlier rule_a (1 kw, no punct -> s_lex=0.75) vs later rule_b (1 kw + punct -> s_lex=1.0)
        ordered_rules = {
            "rule_a": {
                "label": "RuleA",
                "keywords": ["テスト"],
                "acoustic_hints": {"pitch": ["normal"], "speed": ["normal"], "energy": ["normal"], "punctuation": []},
            },
            "rule_b": {
                "label": "RuleB",
                "keywords": ["サンプル"],
                "whisper_threshold_db": -32.0,
                "whisper_label": "RuleB/Whisper",
                "acoustic_hints": {"pitch": ["normal"], "speed": ["normal"], "energy": ["normal"], "punctuation": ["！"]},
            },
            "default_label": "Neutral",
        }
        ordered_clf = CategoryClassifier(ordered_rules)
        rich_tie = ordered_clf.classify_rich(
            "テスト、サンプル！",
            stats={"raw_rms_db": -35.0},
            prosody={"voiced_fraction": 0.18, "spectral_flatness": 0.16, "bins": {"pitch": "normal", "speed": "normal", "energy": "normal"}},
        )
        self.assertEqual(rich_tie["text_cue"]["matched_category"], "RuleB")
        self.assertEqual(rich_tie["text_cue"]["matched_keywords"], ["サンプル"])
        self.assertEqual(rich_tie["auto_tags"]["primary"], "RuleB")
        self.assertEqual(ordered_clf.classify("テスト、サンプル！", {"raw_rms_db": -35.0}), "RuleB/Whisper")


if __name__ == "__main__":
    unittest.main()
