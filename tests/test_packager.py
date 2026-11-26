import csv
import json
import tempfile
import unittest
from pathlib import Path
from modules.enrichment.packager import (
    apply_human_curation_file,
    build_leakage_safe_splits,
    export_enriched_dataset,
    sanitize_list_field,
    select_reference_prompts,
)


def _make_record(
    idx: int,
    text: str,
    cat: str = "Neutral/Dialogue",
    tier: str = "GOLD",
    reviewed: bool = False,
    source_file: str = "kurisu_raw.wav",
    delivery_mode: str = "normal",
):
    fname = f"audio_{idx:05d}.wav"
    return {
        "id": idx,
        "filename": fname,
        "relative_path": f"energy/slices/{fname}",
        "absolute_path": f"/tmp/{fname}",
        "source_file": source_file,
        "start_time_sec": round(idx * 5.0, 3),
        "end_time_sec": round(idx * 5.0 + 4.8, 3),
        "slicer_method": "energy",
        "duration_sec": 4.8,
        "status": "CLEAN" if tier == "GOLD" else ("BORDERLINE" if tier == "SILVER" else "TOO_SHORT"),
        "clean_boundary": True,
        "raw_rms_db": -22.0,
        "raw_peak_db": -6.0,
        "applied_gain_db": 5.0,
        "text": text,
        "chars_per_sec": 4.0,
        "category": cat,
        "provenance": {"schema_version": "2.0", "audio_sha256": f"sha-{idx:05d}", "asr_model": "large-v3"},
        "asr_diagnostics": {
            "raw_text": text,
            "canonical_text": text,
            "avg_logprob": -0.20,
            "no_speech_prob": 0.02,
            "compression_ratio": 1.1,
            "mean_word_prob": 0.92,
            "speech_duration_sec": 4.0,
            "clean_char_count": len(text),
            "estimated_mora_count": float(len(text)),
            "effective_cps": 4.0,
            "effective_mora_rate": 4.0,
            "hallucination_flags": [],
        },
        "prosody": {
            "f0_median_hz": 240.0,
            "f0_iqr_hz": 35.0,
            "f0_slope_hz_per_sec": 0.0,
            "voiced_fraction": 0.75,
            "spectral_centroid_hz": 1500.0,
            "spectral_flatness": 0.03,
            "energy_std_db": 5.0,
            "crest_factor_db": 15.0,
            "bins": {"pitch": "high", "speed": "normal", "energy": "normal"},
        },
        "vocal_expression": {
            "delivery_mode": delivery_mode,
            "text_cue": {"matched_category": cat, "matched_keywords": [], "punctuation_cues": []},
            "auto_tags": {
                "primary": cat,
                "secondary": [],
                "scores": {cat: 0.85, "Neutral/Dialogue": 0.15},
                "valence_est": 0.0,
                "arousal_est": 0.5,
                "confidence": 0.80,
                "status": "suggestion_only",
            },
        },
        "human_review": {
            "reviewed": reviewed,
            "verified_primary": cat if reviewed else None,
            "verified_delivery": delivery_mode if reviewed else None,
            "notes": "",
        },
        "qc": {
            "tier": tier,
            "reasons": [] if tier != "REJECT" else ["ASR_BOH_BLACKLIST"],
            "usable_for_training": tier in ("GOLD", "SILVER"),
            "usable_for_reference": tier == "GOLD" and cat != "Uncertain",
        },
    }


class TestPackager(unittest.TestCase):
    def test_block_split_prevents_duplicate_text_leakage_and_sanitizes_pipe_delimiters(self):
        self.assertEqual(sanitize_list_field("あんた|バカ\nなの？"), "あんた バカ なの？")
        self.assertEqual(sanitize_list_field("  line1\r\n\tline2 |  line3  "), "line1 line2 line3")
        records = [_make_record(i, f"ユニークセリフ番号{i}") for i in range(60)]
        # Inject exact catchphrase duplicates across far-apart temporal blocks (idx 2 and idx 55)
        records[2]["asr_diagnostics"]["canonical_text"] = "べ、別に！"
        records[2]["text"] = "べ、別に！"
        records[55]["asr_diagnostics"]["canonical_text"] = "べ、別に！"
        records[55]["text"] = "べ、別に！"

        train_recs, val_recs = build_leakage_safe_splits(records, val_ratio=0.15, block_size=10, seed=42)
        train_texts = {r["asr_diagnostics"]["canonical_text"] for r in train_recs}
        val_texts = {r["asr_diagnostics"]["canonical_text"] for r in val_recs}
        self.assertGreater(len(val_recs), 0)
        self.assertGreater(len(train_recs), 0)
        self.assertEqual(train_texts.intersection(val_texts), set())

    def test_block_split_respects_source_file_boundaries_and_empty_inputs(self):
        self.assertEqual(build_leakage_safe_splits([]), ([], []))
        rejected = [_make_record(i, f"ダメ{i}", tier="REJECT") for i in range(5)]
        self.assertEqual(build_leakage_safe_splits(rejected), ([], []))

        # Two separate source files, 2 records each, block_size=10 -> 2 separate components
        records = [
            _make_record(0, "s1_a", cat="Neutral/Dialogue", source_file="file_a.wav"),
            _make_record(1, "s1_b", cat="Neutral/Dialogue", source_file="file_a.wav"),
            _make_record(2, "s2_a", cat="Playful/Teasing", source_file="file_b.wav"),
            _make_record(3, "s2_b", cat="Playful/Teasing", source_file="file_b.wav"),
        ]
        train_recs, val_recs = build_leakage_safe_splits(records, val_ratio=0.05, block_size=10, seed=42)
        self.assertGreaterEqual(len(val_recs), 1)
        self.assertGreaterEqual(len(train_recs), 1)
        # Neither source_file should be split across train and val since block_size=10 > 2
        train_sources = {r["source_file"] for r in train_recs}
        val_sources = {r["source_file"] for r in val_recs}
        self.assertEqual(train_sources.intersection(val_sources), set())

    def test_reference_selection_attaches_neighbor_context_and_prioritizes_human_verified(self):
        records = [
            _make_record(0, "前のセリフ", cat="Neutral/Dialogue"),
            _make_record(1, "自動候補セリフ", cat="Gentle/Affectionate", reviewed=False),
            _make_record(2, "人間確認済みセリフ", cat="Gentle/Affectionate", reviewed=True),
            _make_record(3, "次のセリフ", cat="Neutral/Dialogue"),
            _make_record(4, "除外されるゴミ", cat="Gentle/Affectionate", tier="REJECT"),
            _make_record(5, "不確定セリフ", cat="Uncertain", tier="GOLD"),
        ]
        # Force usable_for_reference True on Uncertain to verify select_reference_prompts still excludes it
        records[5]["qc"]["usable_for_reference"] = True
        # Make unreviewed clip have slightly higher ASR logprob than reviewed clip
        records[1]["asr_diagnostics"]["avg_logprob"] = -0.05
        records[2]["asr_diagnostics"]["avg_logprob"] = -0.35

        candidates, active = select_reference_prompts(records, top_candidates_per_cat=5, active_per_cat=2)
        self.assertNotIn("Uncertain", candidates)
        self.assertNotIn("Uncertain", active)

        gentle_active = active["Gentle/Affectionate"]
        self.assertEqual(gentle_active[0]["filename"], "audio_00002.wav")
        self.assertTrue(gentle_active[0]["human_verified"])
        self.assertEqual(gentle_active[0]["prev_slice_filename"], "audio_00001.wav")
        self.assertEqual(gentle_active[0]["next_slice_filename"], "audio_00003.wav")

        # Verify required candidate fields and source_file boundary None
        neutral_cands = candidates["Neutral/Dialogue"]
        first_neutral = next(c for c in neutral_cands if c["filename"] == "audio_00000.wav")
        self.assertIsNone(first_neutral["prev_slice_filename"])
        self.assertEqual(first_neutral["next_slice_filename"], "audio_00001.wav")
        expected_keys = {
            "filename",
            "audio_sha256",
            "relative_path",
            "absolute_path",
            "source_file",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "canonical_text",
            "effective_category",
            "effective_delivery",
            "human_verified",
            "prev_slice_filename",
            "next_slice_filename",
            "score",
            "human_review",
        }
        self.assertTrue(expected_keys.issubset(first_neutral.keys()))

    def test_export_and_apply_curation_roundtrip(self):
        records = [_make_record(i, f"セリフ{i}|テスト", cat="Gentle/Affectionate") for i in range(10)]
        records.append(_make_record(10, "銀レベルのセリフ", cat="Neutral/Dialogue", tier="SILVER"))
        records.append(_make_record(11, "銀レベル2", cat="Neutral/Dialogue", tier="SILVER"))
        records.append(_make_record(12, "ダメなセリフ", tier="REJECT"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            paths = export_enriched_dataset(
                records,
                out_dir,
                "Kurisu",
                "JA",
                cfg={"enrichment": {"packager": {"val_ratio": 0.1, "block_size": 4, "seed": 42}}},
            )
            expected_path_keys = {
                "metadata_json",
                "metadata_csv",
                "dataset_list",
                "dataset_gold_list",
                "train_list",
                "val_list",
                "manifest_jsonl",
                "reference_candidates",
                "reference_prompts",
            }
            self.assertEqual(set(paths.keys()), expected_path_keys)
            for p in paths.values():
                self.assertTrue(p.exists(), f"Missing exported file: {p}")

            # dataset.list has GOLD + SILVER (12), dataset_gold.list has GOLD only (10)
            dataset_lines = paths["dataset_list"].read_text(encoding="utf-8").splitlines()
            gold_lines = paths["dataset_gold_list"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(dataset_lines), 12)
            self.assertEqual(len(gold_lines), 10)
            for line in dataset_lines:
                self.assertEqual(line.count("|"), 3)

            # metadata.json contains all 13 records (including REJECT)
            exported_meta = json.loads(paths["metadata_json"].read_text(encoding="utf-8"))
            self.assertEqual(len(exported_meta), 13)

            # metadata.csv has flattened scalar columns only (no nested dict/list JSON strings)
            with paths["metadata_csv"].open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 13)
            required_csv_cols = {
                "qc_tier",
                "delivery_mode",
                "effective_category",
                "auto_confidence",
                "human_reviewed",
                "f0_median_hz",
                "voiced_fraction",
                "effective_mora_rate",
                "avg_logprob",
            }
            self.assertTrue(required_csv_cols.issubset(rows[0].keys()))
            for row in rows:
                for val in row.values():
                    self.assertNotIn("{", val)
                    self.assertNotIn("[", val)

            # manifest.jsonl has GOLD + SILVER (12) and natural-language style_prompt
            manifest_lines = paths["manifest_jsonl"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), 12)
            first_manifest = json.loads(manifest_lines[0])
            self.assertIn("style_prompt", first_manifest)
            self.assertIn("Gentle/Affectionate", first_manifest["style_prompt"])
            self.assertIn("Japanese speaker", first_manifest["style_prompt"])

            # Edit reference_candidates.json to mark audio_00000.wav as human-reviewed Playful/Teasing
            cands = json.loads(paths["reference_candidates"].read_text(encoding="utf-8"))
            cands["Gentle/Affectionate"][0]["human_review"] = {
                "reviewed": True,
                "verified_primary": "Playful/Teasing",
                "verified_delivery": "normal",
                "notes": "teasing tone",
            }
            edited_path = out_dir / "edited_candidates.json"
            edited_path.write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")

            updated = apply_human_curation_file(records, edited_path)
            self.assertEqual(updated, 1)
            self.assertEqual(records[0]["category"], "Playful/Teasing")
            self.assertTrue(records[0]["human_review"]["reviewed"])

    def test_apply_human_curation_rejects_sha_mismatch_and_supports_list_format(self):
        records = [
            _make_record(0, "テストゼロ", cat="Neutral/Dialogue", reviewed=False),
            _make_record(1, "テストイチ", cat="Neutral/Dialogue", reviewed=False),
        ]
        # Pre-mark record 1 as already reviewed to verify unreviewed curation items don't overwrite it
        records[1]["human_review"] = {
            "reviewed": True,
            "verified_primary": "Playful/Teasing",
            "verified_delivery": "whisper",
            "notes": "keep me",
        }
        curation_entries = [
            {
                "filename": "audio_00000.wav",
                "audio_sha256": "WRONG-SHA",
                "human_review": {
                    "reviewed": True,
                    "verified_primary": "Defensive/Flustered",
                    "verified_delivery": "shout",
                    "notes": "should be rejected by SHA check",
                },
            },
            {
                "filename": "audio_00001.wav",
                "audio_sha256": "sha-00001",
                "human_review": {
                    "reviewed": False,
                    "verified_primary": None,
                    "verified_delivery": None,
                    "notes": "unreviewed entry should be skipped",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            curation_path = Path(tmpdir) / "curation_list.json"
            curation_path.write_text(json.dumps(curation_entries, ensure_ascii=False), encoding="utf-8")

            updated = apply_human_curation_file(records, curation_path)
            self.assertEqual(updated, 0)
            self.assertFalse(records[0]["human_review"]["reviewed"])
            self.assertEqual(records[1]["human_review"]["verified_primary"], "Playful/Teasing")

            # Now fix SHA on audio_00000.wav and verify it applies cleanly
            curation_entries[0]["audio_sha256"] = "sha-00000"
            curation_path.write_text(json.dumps(curation_entries, ensure_ascii=False), encoding="utf-8")
            updated_valid = apply_human_curation_file(records, curation_path)
            self.assertEqual(updated_valid, 1)
            self.assertTrue(records[0]["human_review"]["reviewed"])
            self.assertEqual(records[0]["category"], "Defensive/Flustered")
            self.assertEqual(records[0]["qc"]["tier"], "GOLD")

    def test_apply_human_curation_rejects_missing_none_or_empty_sha256(self):
        records = [
            _make_record(0, "テストゼロ", cat="Neutral/Dialogue", reviewed=False),
            _make_record(1, "テストイチ", cat="Neutral/Dialogue", reviewed=False),
            _make_record(2, "テストニ", cat="Neutral/Dialogue", reviewed=False),
        ]
        curation_entries = [
            {
                "filename": "audio_00000.wav",
                # omitted audio_sha256
                "human_review": {
                    "reviewed": True,
                    "verified_primary": "Playful/Teasing",
                    "verified_delivery": "normal",
                    "notes": "omitted sha",
                },
            },
            {
                "filename": "audio_00001.wav",
                "audio_sha256": None,
                "human_review": {
                    "reviewed": True,
                    "verified_primary": "Playful/Teasing",
                    "verified_delivery": "normal",
                    "notes": "none sha",
                },
            },
            {
                "filename": "audio_00002.wav",
                "audio_sha256": "",
                "human_review": {
                    "reviewed": True,
                    "verified_primary": "Playful/Teasing",
                    "verified_delivery": "normal",
                    "notes": "empty sha",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            curation_path = Path(tmpdir) / "missing_sha.json"
            curation_path.write_text(json.dumps(curation_entries, ensure_ascii=False), encoding="utf-8")
            updated = apply_human_curation_file(records, curation_path)
            self.assertEqual(updated, 0)
            for rec in records:
                self.assertFalse(rec["human_review"]["reviewed"])
                self.assertIsNone(rec["human_review"]["verified_primary"])
                self.assertEqual(rec["category"], "Neutral/Dialogue")

    def test_block_split_pass1_guards_against_large_merged_component_overshoot(self):
        # 100 records, block_size=10 -> 10 blocks.
        # Blocks 0..5 (60 records) cover 4 distinct categories and merge via a shared catchphrase,
        # while blocks 6..9 (10 records each) only cover a single category each.
        cats = ["Neutral/Dialogue", "Playful/Teasing", "Defensive/Flustered", "Gentle/Affectionate"]
        records = [
            _make_record(i, f"個別発話{i}", cat=cats[i % len(cats)] if i < 60 else "Neutral/Dialogue")
            for i in range(100)
        ]
        for block_idx in range(6):
            idx = block_idx * 10
            records[idx]["asr_diagnostics"]["canonical_text"] = "共通の口癖"
            records[idx]["text"] = "共通の口癖"

        # target_val_count = max(1, round(100 * 0.10)) = 10.
        # Giant merged component has size 60 and covers 4 categories (higher gain than size-10 blocks);
        # Pass 1 must guard against overshoot and keep val near target_val_count.
        train_recs, val_recs = build_leakage_safe_splits(records, val_ratio=0.10, block_size=10, seed=42)
        self.assertGreaterEqual(len(train_recs), 80)
        self.assertLessEqual(len(val_recs), 20)
        self.assertGreater(len(val_recs), 0)
        train_texts = {r["asr_diagnostics"]["canonical_text"] for r in train_recs}
        val_texts = {r["asr_diagnostics"]["canonical_text"] for r in val_recs}
        self.assertEqual(train_texts.intersection(val_texts), set())


if __name__ == "__main__":
    unittest.main()
