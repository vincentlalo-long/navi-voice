import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from modules.enrichment import (
    SCHEMA_VERSION,
    HardenedWhisperTranscriber,
    compute_file_sha256,
    default_human_review,
    evaluate_asr_gates,
    evaluate_slice_qc,
    finalize_record_v2,
    merge_cached_record,
    resolve_effective_emotion,
)


def _base_gold_record() -> dict:
    return {
        "duration_sec": 4.8,
        "status": "CLEAN",
        "clipping_rate": 0.0,
        "asr_diagnostics": {
            "raw_text": "理論的には可能よ。",
            "canonical_text": "理論的には可能よ。",
            "avg_logprob": -0.25,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.15,
            "mean_word_prob": 0.88,
            "clean_char_count": 8,
            "effective_cps": 3.5,
            "effective_mora_rate": 4.2,
            "hallucination_flags": [],
        },
        "vocal_expression": {
            "delivery_mode": "normal",
            "auto_tags": {"primary": "Intellectual/Analytical", "status": "suggestion_only"},
        },
        "human_review": default_human_review(),
    }


class _FakeWord:
    def __init__(self, start: float, end: float, probability: float) -> None:
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSegment:
    def __init__(self, text: str, avg_logprob: float, no_speech_prob: float, tokens, words) -> None:
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.tokens = tokens
        self.words = words


class _FakeWhisperModel:
    def __init__(self, segments_to_return) -> None:
        self.segments_to_return = segments_to_return
        self.calls = []

    def transcribe(self, audio_path: str, **kwargs):
        self.calls.append((audio_path, kwargs))
        return iter(self.segments_to_return), types.SimpleNamespace(language="ja")


class TestQCAndCache(unittest.TestCase):
    def test_cache_preserves_human_review_on_matching_sha256_and_invalidates_on_audio_change(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(b"RIFF-fake-wav-bytes-v1")
            tmp.flush()
            sha_v1 = compute_file_sha256(Path(tmp.name))

        cached = {
            "filename": "audio_00001.wav",
            "provenance": {"schema_version": SCHEMA_VERSION, "audio_sha256": sha_v1, "asr_model": "large-v3"},
            "asr_diagnostics": {"canonical_text": "テスト成功", "avg_logprob": -0.2},
            "human_review": {
                "reviewed": True,
                "verified_primary": "Intellectual/Analytical",
                "verified_delivery": "normal",
                "notes": "checked",
            },
        }
        fresh = {"filename": "audio_00001.wav", "start_time_sec": 1.25, "raw_rms_db": -22.5}
        merged_same, reuse_same = merge_cached_record(
            fresh, cached, sha_v1, expected_asr_model="large-v3"
        )
        self.assertTrue(reuse_same)
        self.assertTrue(merged_same["human_review"]["reviewed"])
        self.assertEqual(merged_same["human_review"]["verified_primary"], "Intellectual/Analytical")
        self.assertEqual(merged_same["asr_diagnostics"]["canonical_text"], "テスト成功")
        self.assertEqual(merged_same["start_time_sec"], 1.25)
        self.assertEqual(merged_same["raw_rms_db"], -22.5)
        # Ensure fresh_base was not mutated
        self.assertNotIn("human_review", fresh)

        # Model changed -> keep human_review (audio is identical) but force ASR rerun
        merged_diff_model, reuse_diff_model = merge_cached_record(
            fresh, cached, sha_v1, expected_asr_model="large-v3-turbo"
        )
        self.assertFalse(reuse_diff_model)
        self.assertTrue(merged_diff_model["human_review"]["reviewed"])
        self.assertNotIn("asr_diagnostics", merged_diff_model)

        # Audio SHA changed -> invalidate BOTH cached ASR and stale human_review
        merged_new_wav, reuse_new_wav = merge_cached_record(
            fresh, cached, "deadbeef" * 8, expected_asr_model="large-v3"
        )
        self.assertFalse(reuse_new_wav)
        self.assertFalse(merged_new_wav["human_review"]["reviewed"])
        self.assertIsNone(merged_new_wav["human_review"]["verified_primary"])
        self.assertNotIn("asr_diagnostics", merged_new_wav)

    def test_four_gate_asr_and_three_tier_qc_classification(self):
        gold_rec = _base_gold_record()
        gold_rec["asr_diagnostics"]["hallucination_flags"] = evaluate_asr_gates(
            gold_rec["asr_diagnostics"], gold_rec["duration_sec"]
        )
        qc_gold = evaluate_slice_qc(gold_rec)
        self.assertEqual(qc_gold["tier"], "GOLD")
        self.assertEqual(qc_gold["reasons"], [])
        self.assertTrue(qc_gold["usable_for_training"])
        self.assertTrue(qc_gold["usable_for_reference"])

        # Uncertain label stays GOLD for training if acoustics/ASR are clean, but NOT usable_for_reference
        uncertain_rec = {
            **gold_rec,
            "vocal_expression": {
                "delivery_mode": "normal",
                "auto_tags": {"primary": "Uncertain", "status": "abstained_uncertain"},
            },
        }
        qc_uncertain = evaluate_slice_qc(uncertain_rec)
        self.assertEqual(qc_uncertain["tier"], "GOLD")
        self.assertTrue(qc_uncertain["usable_for_training"])
        self.assertFalse(qc_uncertain["usable_for_reference"])

        # BoH hallucination -> REJECT
        boh_diag = {
            **gold_rec["asr_diagnostics"],
            "raw_text": "ご視聴ありがとうございました",
            "canonical_text": "ご視聴ありがとうございました",
        }
        flags = evaluate_asr_gates(boh_diag, 4.8)
        self.assertIn("ASR_BOH_BLACKLIST", flags)
        qc_rej = evaluate_slice_qc({**gold_rec, "asr_diagnostics": {**boh_diag, "hallucination_flags": flags}})
        self.assertEqual(qc_rej["tier"], "REJECT")
        self.assertFalse(qc_rej["usable_for_training"])
        self.assertFalse(qc_rej["usable_for_reference"])
        self.assertIn("ASR_BOH_BLACKLIST", qc_rej["reasons"])

    def test_resolve_effective_emotion_prioritizes_human_review(self):
        rec = {
            "vocal_expression": {
                "delivery_mode": "normal",
                "auto_tags": {"primary": "Neutral/Dialogue", "status": "suggestion_only"},
            },
            "human_review": {
                "reviewed": True,
                "verified_primary": "Playful/Teasing",
                "verified_delivery": "whisper_breathy",
                "notes": "",
            },
        }
        cat, delivery, verified = resolve_effective_emotion(rec)
        self.assertEqual(cat, "Playful/Teasing")
        self.assertEqual(delivery, "whisper_breathy")
        self.assertTrue(verified)

        # Unreviewed falls back to auto_tags
        rec_unreviewed = {
            **rec,
            "human_review": default_human_review(),
        }
        cat_u, del_u, ver_u = resolve_effective_emotion(rec_unreviewed)
        self.assertEqual(cat_u, "Neutral/Dialogue")
        self.assertEqual(del_u, "normal")
        self.assertFalse(ver_u)

    def test_finalize_record_v2_syncs_flat_fields_and_preserves_human_review(self):
        rec = {
            "asr_diagnostics": {
                "canonical_text": "理論的には可能よ！",
                "effective_cps": 4.567,
            },
            "vocal_expression": {
                "delivery_mode": "normal",
                "auto_tags": {"primary": "Intellectual/Analytical"},
            },
            "human_review": {
                "reviewed": True,
                "verified_primary": "Playful/Teasing",
            },
        }
        out = finalize_record_v2(rec)
        self.assertIs(out, rec)
        self.assertEqual(out["text"], "理論的には可能よ！")
        self.assertEqual(out["chars_per_sec"], 4.57)
        self.assertEqual(out["category"], "Playful/Teasing")
        self.assertEqual(out["provenance"]["schema_version"], SCHEMA_VERSION)
        self.assertTrue(out["human_review"]["reviewed"])
        self.assertIsNone(out["human_review"]["verified_delivery"])
        self.assertEqual(out["human_review"]["notes"], "")

    def test_silver_tier_and_reference_boundaries(self):
        # Slightly noisy logprob (-0.75) -> SILVER, usable_for_training=True, usable_for_reference=False
        silver_rec = _base_gold_record()
        silver_rec["asr_diagnostics"]["avg_logprob"] = -0.75
        qc_silver = evaluate_slice_qc(silver_rec)
        self.assertEqual(qc_silver["tier"], "SILVER")
        self.assertTrue(qc_silver["usable_for_training"])
        self.assertFalse(qc_silver["usable_for_reference"])
        self.assertTrue(len(qc_silver["reasons"]) > 0)

        # GOLD slice with status="SAFE" or duration=2.5s -> GOLD for training, False for reference
        safe_rec = _base_gold_record()
        safe_rec["status"] = "SAFE"
        qc_safe = evaluate_slice_qc(safe_rec)
        self.assertEqual(qc_safe["tier"], "GOLD")
        self.assertTrue(qc_safe["usable_for_training"])
        self.assertFalse(qc_safe["usable_for_reference"])

        short_gold = _base_gold_record()
        short_gold["duration_sec"] = 2.5
        qc_short = evaluate_slice_qc(short_gold)
        self.assertEqual(qc_short["tier"], "GOLD")
        self.assertTrue(qc_short["usable_for_training"])
        self.assertFalse(qc_short["usable_for_reference"])

        # TOO_SHORT or clipping >= 0.001 -> REJECT
        clip_rec = _base_gold_record()
        clip_rec["clipping_rate"] = 0.001
        qc_clip = evaluate_slice_qc(clip_rec)
        self.assertEqual(qc_clip["tier"], "REJECT")
        self.assertFalse(qc_clip["usable_for_training"])
        self.assertTrue(len(qc_clip["reasons"]) > 0)

    def test_all_asr_gate_flags_and_cfg_override(self):
        base = {
            "raw_text": "承知しました。",
            "canonical_text": "承知しました。",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.1,
            "mean_word_prob": 0.9,
            "clean_char_count": 6,
            "effective_cps": 4.0,
            "effective_mora_rate": 5.0,
        }
        self.assertEqual(evaluate_asr_gates(base, 2.0), [])

        self.assertIn("ASR_EMPTY_TEXT", evaluate_asr_gates({**base, "clean_char_count": 0}, 2.0))
        self.assertIn("ASR_HIGH_COMPRESSION", evaluate_asr_gates({**base, "compression_ratio": 2.5}, 2.0))
        self.assertIn("ASR_LOW_LOGPROB", evaluate_asr_gates({**base, "avg_logprob": -1.1}, 2.0))
        self.assertIn(
            "ASR_HIGH_NOSPEECH",
            evaluate_asr_gates({**base, "no_speech_prob": 0.65, "avg_logprob": -0.85}, 2.0),
        )
        # High no_speech_prob alone without low avg_logprob must NOT fire ASR_HIGH_NOSPEECH
        self.assertNotIn(
            "ASR_HIGH_NOSPEECH",
            evaluate_asr_gates({**base, "no_speech_prob": 0.65, "avg_logprob": -0.3}, 2.0),
        )
        self.assertIn("ASR_LOW_WORD_PROB", evaluate_asr_gates({**base, "mean_word_prob": 0.40}, 2.0))
        self.assertIn(
            "ASR_BOH_BLACKLIST",
            evaluate_asr_gates({**base, "raw_text": "チャンネル登録お願いします"}, 2.0),
        )
        self.assertIn(
            "ASR_NGRAM_REPETITION",
            evaluate_asr_gates({**base, "canonical_text": "あああああ"}, 2.0),
        )
        self.assertIn(
            "ASR_UNDER_RATE",
            evaluate_asr_gates({**base, "effective_cps": 1.0}, 3.5),
        )
        self.assertIn(
            "ASR_OVER_RATE",
            evaluate_asr_gates({**base, "effective_cps": 18.5}, 2.0),
        )
        self.assertIn(
            "ASR_OVER_RATE",
            evaluate_asr_gates({**base, "effective_mora_rate": 26.0}, 2.0),
        )

        # Config override test (nested enrichment.qc_gates.asr)
        custom_cfg = {"enrichment": {"qc_gates": {"asr": {"min_avg_logprob": -0.5}}}}
        self.assertIn(
            "ASR_LOW_LOGPROB",
            evaluate_asr_gates({**base, "avg_logprob": -0.6}, 2.0, cfg=custom_cfg),
        )

    def test_hardened_whisper_transcriber_with_fake_model(self):
        seg1 = _FakeSegment(
            text="（笑） 理論的には",
            avg_logprob=-0.20,
            no_speech_prob=0.05,
            tokens=[1, 2, 3, 4],
            words=[_FakeWord(0.2, 0.9, 0.90), _FakeWord(1.0, 1.5, 0.80)],
        )
        seg2 = _FakeSegment(
            text="可能よ！！！",
            avg_logprob=-0.60,
            no_speech_prob=0.15,
            tokens=[5, 6],
            words=[_FakeWord(1.6, 2.2, 0.70)],
        )
        fake_model = _FakeWhisperModel([seg1, seg2])
        transcriber = HardenedWhisperTranscriber(
            model_size="large-v3",
            device="cpu",
            compute_type="int8",
            language="ja",
            beam_size=5,
            model=fake_model,
        )
        diag = transcriber.transcribe_file(Path("/tmp/fake.wav"), duration_sec=3.0)

        self.assertEqual(len(fake_model.calls), 1)
        _, call_kwargs = fake_model.calls[0]
        self.assertEqual(call_kwargs["language"], "ja")
        self.assertEqual(call_kwargs["beam_size"], 5)
        self.assertFalse(call_kwargs["vad_filter"])
        self.assertFalse(call_kwargs["condition_on_previous_text"])
        self.assertTrue(call_kwargs["word_timestamps"])
        self.assertEqual(call_kwargs["temperature"], (0.0, 0.2, 0.4))

        self.assertEqual(diag["raw_text"], "（笑） 理論的には可能よ！！！")
        self.assertEqual(diag["canonical_text"], "理論的には可能よ！")
        # Token-weighted avg_logprob: (-0.20 * 4 + -0.60 * 2) / 6 = -2.0 / 6
        self.assertAlmostEqual(diag["avg_logprob"], -2.0 / 6.0, places=3)
        self.assertAlmostEqual(diag["no_speech_prob"], 0.10, places=3)
        self.assertAlmostEqual(diag["mean_word_prob"], 0.80, places=3)
        self.assertAlmostEqual(diag["speech_duration_sec"], 2.0, places=3)
        self.assertGreater(diag["compression_ratio"], 0.0)
        self.assertEqual(diag["clean_char_count"], 8)
        self.assertEqual(diag["hallucination_flags"], [])

        # Empty segments fallback
        empty_transcriber = HardenedWhisperTranscriber(
            model_size="large-v3",
            device="cpu",
            compute_type="int8",
            model=_FakeWhisperModel([]),
        )
        empty_diag = empty_transcriber.transcribe_file(Path("/tmp/empty.wav"), duration_sec=2.5)
        self.assertEqual(empty_diag["raw_text"], "")
        self.assertEqual(empty_diag["canonical_text"], "")
        self.assertEqual(empty_diag["avg_logprob"], -99.0)
        self.assertEqual(empty_diag["no_speech_prob"], 1.0)
        self.assertEqual(empty_diag["compression_ratio"], 0.0)
        self.assertEqual(empty_diag["mean_word_prob"], 0.0)
        self.assertEqual(empty_diag["speech_duration_sec"], 2.5)

        transcriber.release()
        self.assertIsNone(transcriber.model)

    def test_hardened_whisper_local_files_only_fallback(self):
        fake_loaded = _FakeWhisperModel([])
        calls = []

        def _fake_whisper_ctor(*args, **kwargs):
            calls.append((args, kwargs))
            if kwargs.get("local_files_only") is True:
                raise RuntimeError("not in local cache")
            return fake_loaded

        with mock.patch("modules.enrichment.asr.WhisperModel", side_effect=_fake_whisper_ctor):
            tr = HardenedWhisperTranscriber("large-v3", device="cpu", compute_type="float16")
            self.assertIs(tr.model, fake_loaded)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0][1]["local_files_only"])
            self.assertFalse(calls[1][1]["local_files_only"])
            # float16 on CPU should safely downgrade to int8
            self.assertEqual(calls[1][1]["compute_type"], "int8")

    def test_qc_handles_missing_or_none_diagnostics_safely(self):
        qc_missing = evaluate_slice_qc({"duration_sec": 4.0, "status": "CLEAN", "asr_diagnostics": {"avg_logprob": None}})
        self.assertEqual(qc_missing["tier"], "REJECT")
        self.assertIn("empty_canonical_text", qc_missing["reasons"])

        flags = evaluate_asr_gates({"raw_text": "", "canonical_text": "", "avg_logprob": None}, 2.0)
        self.assertIn("ASR_EMPTY_TEXT", flags)
        self.assertIn("ASR_LOW_LOGPROB", flags)


if __name__ == "__main__":
    unittest.main()
