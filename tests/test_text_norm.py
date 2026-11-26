import math
import unittest
from modules.enrichment import (
    DEFAULT_BOH_SUBSTRINGS,
    canonicalize_japanese_text,
    compute_speaking_rates,
    count_japanese_phonetic_units,
    detect_boh_hallucination,
    detect_repetition_anomaly,
)


class TestTextNorm(unittest.TestCase):
    def test_canonicalize_nfkc_punctuation_and_subtitle_tags(self):
        raw = "（笑） べ、別に。。。あんたのためじゃないんだからね！！！ (ため息) 【字幕】"
        self.assertEqual(
            canonicalize_japanese_text(raw),
            "べ、別に…あんたのためじゃないんだからね！",
        )

    def test_punctuation_and_subtitle_tag_only_inputs_yield_zero_clean_chars(self):
        self.assertEqual(canonicalize_japanese_text("（笑）"), "")
        clean_empty, mora_empty = count_japanese_phonetic_units(
            canonicalize_japanese_text("……！？")
        )
        self.assertEqual(clean_empty, 0)
        self.assertEqual(mora_empty, 0.0)
        clean_sokuon, mora_sokuon = count_japanese_phonetic_units(
            canonicalize_japanese_text("……っ！？")
        )
        self.assertEqual(clean_sokuon, 1)
        self.assertAlmostEqual(mora_sokuon, 1.0)

    def test_japanese_mora_estimation_handles_digraphs_and_kanji(self):
        # 承知 (2 kanji -> 3.6) + しました (4 hiragana -> 4.0) = 7.6; しゃ (2 chars -> 1.0 mora)
        chars_1, mora_1 = count_japanese_phonetic_units("承知しました。")
        self.assertEqual(chars_1, 6)
        self.assertAlmostEqual(mora_1, 7.6, places=2)
        chars_2, mora_2 = count_japanese_phonetic_units("しゃべる")
        self.assertEqual(chars_2, 4)
        self.assertAlmostEqual(mora_2, 3.0, places=2)

    def test_compute_speaking_rates_clamps_minimum_duration(self):
        rates = compute_speaking_rates("ありがとう", speech_duration_sec=0.05)
        self.assertEqual(rates["clean_char_count"], 5)
        self.assertAlmostEqual(rates["speech_duration_sec"], 0.2)
        self.assertAlmostEqual(rates["effective_cps"], 25.0)
        self.assertAlmostEqual(rates["effective_mora_rate"], 25.0)

    def test_boh_blacklist_catches_youtube_subtitles_not_valid_dialogue(self):
        self.assertIsNotNone(detect_boh_hallucination("ご視聴ありがとうございました！"))
        self.assertIsNotNone(detect_boh_hallucination("Subtitles by Amara.org community"))
        # Legitimate anime dialogue phrase must NOT be blacklisted
        self.assertIsNone(detect_boh_hallucination("今日も一日お疲れ様でした。"))
        self.assertIn("チャンネル登録", DEFAULT_BOH_SUBSTRINGS)

    def test_repetition_anomaly_detects_runaway_loops_only(self):
        self.assertTrue(detect_repetition_anomaly("あああああ"))
        self.assertTrue(detect_repetition_anomaly("好き好き好き好き好き"))
        self.assertFalse(detect_repetition_anomaly("べ、別にそんなことないわよ"))
        self.assertFalse(detect_repetition_anomaly("はははっ"))

    def test_single_punctuation_restoration_and_subtitle_variants(self):
        self.assertEqual(canonicalize_japanese_text(""), "")
        self.assertEqual(canonicalize_japanese_text("そうだ! 本当?"), "そうだ！ 本当？")
        self.assertEqual(canonicalize_japanese_text("承知しました。"), "承知しました。")
        self.assertEqual(canonicalize_japanese_text("♪BGM♪ [拍手]\u200b こんにちは\n世界"), "こんにちは 世界")

    def test_mora_estimation_katakana_prolonged_and_non_finite_duration(self):
        chars, morae = count_japanese_phonetic_units("ニュース")
        self.assertEqual(chars, 4)
        self.assertAlmostEqual(morae, 3.0)

        rates = compute_speaking_rates("承知しました。", speech_duration_sec=math.nan)
        self.assertAlmostEqual(rates["speech_duration_sec"], 0.2)
        self.assertAlmostEqual(rates["estimated_mora_count"], 7.6, places=2)
        self.assertAlmostEqual(rates["effective_mora_rate"], 38.0, places=2)

    def test_repetition_and_boh_edge_cases(self):
        self.assertFalse(detect_repetition_anomaly(""))
        self.assertFalse(detect_repetition_anomaly("ああああ"))
        self.assertTrue(detect_repetition_anomaly("あ、あ、あ、あ、あ"))
        self.assertIsNotNone(detect_boh_hallucination("THANK YOU FOR WATCHING"))
        with self.assertRaises(ValueError):
            detect_repetition_anomaly("あああ", max_consecutive_repeats=0)


if __name__ == "__main__":
    unittest.main()
