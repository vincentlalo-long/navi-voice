import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_wav(wav_path: Path, freq_hz: float, duration_sec: float = 4.5, sr: int = 44100) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    wave = (0.35 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    wave[:2000] = 0.0
    wave[-2000:] = 0.0
    sf.write(str(wav_path), wave, sr, subtype="PCM_16")


def _seed_record(idx: int, wav_path: Path, text: str, sha: str, asr_model: str = "large-v3"):
    return {
        "id": idx,
        "filename": wav_path.name,
        "relative_path": f"energy/slices/{wav_path.name}",
        "absolute_path": str(wav_path),
        "source_file": "synthetic.wav",
        "start_time_sec": idx * 5.0,
        "end_time_sec": idx * 5.0 + 4.5,
        "slicer_method": "energy",
        "duration_sec": 4.5,
        "status": "CLEAN",
        "clean_boundary": True,
        "clipping_rate": 0.0,
        "raw_rms_db": -20.0,
        "raw_peak_db": -5.0,
        "applied_gain_db": 4.0,
        "provenance": {"schema_version": "2.0", "audio_sha256": sha, "asr_model": asr_model},
        "asr_diagnostics": {
            "raw_text": text,
            "canonical_text": text,
            "avg_logprob": -0.22,
            "no_speech_prob": 0.03,
            "compression_ratio": 1.12,
            "mean_word_prob": 0.90,
            "speech_duration_sec": 3.8,
            "clean_char_count": len(text),
            "estimated_mora_count": float(len(text)),
            "effective_cps": 3.5,
            "effective_mora_rate": 4.0,
            "hallucination_flags": [],
        },
    }


class TestPipelineEnrichmentCLI(unittest.TestCase):
    def test_miner_retag_only_and_enrichment_audit_end_to_end_without_gpu_asr(self):
        from modules.enrichment.schema import compute_file_sha256

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            slices_dir = tmp / "energy" / "slices"
            slices_dir.mkdir(parents=True)

            seed_records = []
            for idx, (freq, text) in enumerate([
                (230.0, "理論的にはその仮説が正しいわ。"),
                (290.0, "べ、別にあんたのためじゃないわよ！"),
                (210.0, "ご視聴ありがとうございました"),
            ]):
                wav_path = slices_dir / f"audio_{idx:05d}.wav"
                _write_synthetic_wav(wav_path, freq)
                sha = compute_file_sha256(wav_path)
                seed_records.append(_seed_record(idx, wav_path, text, sha))

            meta_path = tmp / "energy" / "metadata.json"
            meta_path.write_text(json.dumps(seed_records, ensure_ascii=False, indent=2), encoding="utf-8")

            cfg_path = tmp / "test_config.json"
            cfg_data = json.loads((REPO_ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
            cfg_data["paths"]["output_dir"] = str(tmp)
            cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

            # 1. Run miner in --retag-only mode (must NOT load Whisper, must filter BoH slice #2)
            proc_noprof = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "voice_dataset_miner.py"),
                    "--config", str(cfg_path),
                    "--slicer", "energy",
                    "--retag-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_noprof.returncode, 0, msg=proc_noprof.stderr + proc_noprof.stdout)
            list_lines = (tmp / "energy" / "dataset.list").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(list_lines), 2)

            # Verify Phase 2 fields preserved and Schema v2 fields populated
            retagged = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(len(retagged), 3)
            self.assertEqual(retagged[0]["start_time_sec"], 0.0)
            self.assertEqual(retagged[0]["raw_rms_db"], -20.0)
            self.assertEqual(retagged[0]["applied_gain_db"], 4.0)
            self.assertEqual(retagged[0]["qc"]["tier"], "GOLD")
            self.assertEqual(retagged[2]["qc"]["tier"], "REJECT")
            self.assertIn("ASR_BOH_BLACKLIST", retagged[2]["qc"]["reasons"])

            # 2. Run enrichment audit to generate reports & PNG charts
            report_dir = tmp / "reports_enrichment"
            proc2 = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_enrichment_audit.py"),
                    "--processed-dir", str(tmp / "energy"),
                    "--output-dir", str(report_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr + proc2.stdout)
            for expected_file in (
                "enrichment_report.md",
                "enrichment_summary.json",
                "qc_waterfall.png",
                "prosody_acoustic_space.png",
                "category_distribution.png",
            ):
                self.assertTrue((report_dir / expected_file).is_file(), expected_file)

            summary = json.loads((report_dir / "enrichment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["waterfall"]["total_slices"], 3)
            self.assertEqual(summary["waterfall"]["gold_count"], 2)
            self.assertEqual(summary["waterfall"]["reject_count"], 1)
            self.assertIn("ASR_BOH_BLACKLIST", summary["rejection_reasons"])

            report_md = (report_dir / "enrichment_report.md").read_text(encoding="utf-8")
            self.assertIn("qc_waterfall.png", report_md)
            self.assertIn("prosody_acoustic_space.png", report_md)
            self.assertIn("category_distribution.png", report_md)

    def test_enrichment_audit_apply_curation_roundtrip_infers_speaker_from_dataset_list(self):
        from modules.enrichment.schema import compute_file_sha256

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            processed_dir = tmp / "energy" / "kurisu_profile"
            slices_dir = processed_dir / "slices"
            slices_dir.mkdir(parents=True)

            seed_records = []
            for idx, (freq, text) in enumerate([
                (230.0, "理論的にはその仮説が正しいわ。"),
                (290.0, "べ、別にあんたのためじゃないわよ！"),
            ]):
                wav_path = slices_dir / f"audio_{idx:05d}.wav"
                _write_synthetic_wav(wav_path, freq)
                sha = compute_file_sha256(wav_path)
                rec = _seed_record(idx, wav_path, text, sha)
                rec["relative_path"] = f"energy/kurisu_profile/slices/{wav_path.name}"
                seed_records.append(rec)

            meta_path = processed_dir / "metadata.json"
            meta_path.write_text(json.dumps(seed_records, ensure_ascii=False, indent=2), encoding="utf-8")

            cfg_path = tmp / "test_config.json"
            cfg_data = json.loads((REPO_ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
            cfg_data["paths"]["output_dir"] = str(tmp)
            cfg_data["speaker"]["name"] = "Kurisu"
            cfg_data["speaker"]["language"] = "JA"
            cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

            # Use example profile rules via temporary profile so emotion categories exist
            profile_data = json.loads((REPO_ROOT / "configs" / "profiles" / "profile.example.json").read_text(encoding="utf-8"))
            profile_data["speaker"]["name"] = "Kurisu"
            profile_path = tmp / "kurisu_profile.json"
            profile_path.write_text(json.dumps(profile_data, ensure_ascii=False), encoding="utf-8")

            proc_mine = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "voice_dataset_miner.py"),
                    "--config", str(cfg_path),
                    "--profile", str(profile_path),
                    "--slicer", "energy",
                    "--retag-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_mine.returncode, 0, msg=proc_mine.stderr + proc_mine.stdout)

            cands_path = processed_dir / "reference_candidates.json"
            cands = json.loads(cands_path.read_text(encoding="utf-8"))
            first_cat = next(iter(cands))
            cands[first_cat][0]["human_review"] = {
                "reviewed": True,
                "verified_primary": "Playful/Teasing",
                "verified_delivery": "whisper_breathy",
                "notes": "verified in test",
            }
            edited_path = tmp / "edited_curation.json"
            edited_path.write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")

            report_dir = tmp / "audit_out"
            # Do NOT pass --profile to verify run_enrichment_audit infers Kurisu|JA from dataset.list
            proc_audit = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_enrichment_audit.py"),
                    "--processed-dir", str(processed_dir),
                    "--output-dir", str(report_dir),
                    "--apply-curation", str(edited_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_audit.returncode, 0, msg=proc_audit.stderr + proc_audit.stdout)

            updated_meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
            reviewed_items = [r for r in updated_meta if r["human_review"]["reviewed"]]
            self.assertEqual(len(reviewed_items), 1)
            self.assertEqual(reviewed_items[0]["category"], "Playful/Teasing")

            reexported_list = (processed_dir / "dataset.list").read_text(encoding="utf-8").splitlines()
            self.assertTrue(all("|Kurisu|JA|" in line for line in reexported_list))

            summary = json.loads((report_dir / "enrichment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["waterfall"]["human_reviewed_count"], 1)

    def test_miner_skip_transcribe_without_cached_asr_exports_without_rejecting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            slices_dir = tmp / "energy" / "slices"
            slices_dir.mkdir(parents=True)

            wav_path = slices_dir / "audio_00000.wav"
            _write_synthetic_wav(wav_path, 240.0)

            seed_phase2 = [{
                "id": 0,
                "filename": wav_path.name,
                "relative_path": f"energy/slices/{wav_path.name}",
                "absolute_path": str(wav_path),
                "source_file": "raw_ep01.wav",
                "start_time_sec": 12.5,
                "end_time_sec": 17.0,
                "slicer_method": "energy",
                "duration_sec": 4.5,
                "status": "CLEAN",
                "clean_boundary": True,
                "clipping_rate": 0.0,
                "raw_rms_db": -26.5,
                "raw_peak_db": -8.2,
                "applied_gain_db": 7.2,
                "text": "",
                "chars_per_sec": 0.0,
                "category": "",
            }]
            meta_path = tmp / "energy" / "metadata.json"
            meta_path.write_text(json.dumps(seed_phase2, ensure_ascii=False, indent=2), encoding="utf-8")

            cfg_path = tmp / "test_config.json"
            cfg_data = json.loads((REPO_ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
            cfg_data["paths"]["output_dir"] = str(tmp)
            cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "voice_dataset_miner.py"),
                    "--config", str(cfg_path),
                    "--slicer", "energy",
                    "--skip-slicing",
                    "--skip-transcribe",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)

            exported = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(len(exported), 1)
            self.assertNotEqual(exported[0]["qc"]["tier"], "REJECT")
            self.assertEqual(exported[0]["start_time_sec"], 12.5)
            self.assertEqual(exported[0]["raw_rms_db"], -26.5)
            self.assertIn("f0_median_hz", exported[0]["prosody"])

            with (tmp / "energy" / "metadata.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0]["qc_tier"], "REJECT")

    def test_miner_retag_only_invalidates_stale_cache_when_wav_changes(self):
        from modules.enrichment.schema import compute_file_sha256

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            slices_dir = tmp / "energy" / "slices"
            slices_dir.mkdir(parents=True)

            wav_path = slices_dir / "audio_00000.wav"
            _write_synthetic_wav(wav_path, 230.0)
            old_sha = compute_file_sha256(wav_path)

            seed = _seed_record(0, wav_path, "理論的には正しいわ。", old_sha)
            seed["human_review"] = {
                "reviewed": True,
                "verified_primary": "Intellectual/Analytical",
                "verified_delivery": "normal",
                "notes": "stale review",
            }
            meta_path = tmp / "energy" / "metadata.json"
            meta_path.write_text(json.dumps([seed], ensure_ascii=False), encoding="utf-8")

            # Overwrite WAV so SHA256 no longer matches cached metadata
            _write_synthetic_wav(wav_path, 310.0)

            cfg_path = tmp / "test_config.json"
            cfg_data = json.loads((REPO_ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
            cfg_data["paths"]["output_dir"] = str(tmp)
            cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "voice_dataset_miner.py"),
                    "--config", str(cfg_path),
                    "--slicer", "energy",
                    "--retag-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)

            updated = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertFalse(updated[0]["human_review"]["reviewed"])
            self.assertEqual(updated[0]["qc"]["tier"], "REJECT")
            self.assertIn("ASR_EMPTY_TEXT", updated[0]["qc"]["reasons"])

    def test_enrichment_metrics_and_report_handle_empty_and_single_record_safely(self):
        from modules.evaluation import compute_enrichment_metrics, generate_enrichment_report

        empty_metrics = compute_enrichment_metrics([])
        self.assertEqual(empty_metrics["waterfall"]["total_slices"], 0)
        self.assertEqual(empty_metrics["abstention"]["abstention_rate"], 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_empty = Path(tmpdir) / "empty_report"
            paths_empty = generate_enrichment_report([], out_empty)
            for p in paths_empty.values():
                self.assertTrue(p.is_file(), str(p))

            out_single = Path(tmpdir) / "single_report"
            single_rec = {
                "id": 0,
                "filename": "audio_00000.wav",
                "duration_sec": 3.8,
                "status": "CLEAN",
                "clipping_rate": 0.0,
                "raw_rms_db": -22.0,
                "asr_diagnostics": {
                    "canonical_text": "こんにちは。",
                    "avg_logprob": -0.25,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.1,
                    "effective_cps": 2.5,
                    "effective_mora_rate": 2.5,
                    "hallucination_flags": [],
                },
                "prosody": {
                    "f0_median_hz": 0.0,
                    "f0_iqr_hz": 0.0,
                    "voiced_fraction": 0.0,
                    "spectral_centroid_hz": 1100.0,
                    "spectral_flatness": 0.2,
                    "energy_std_db": 2.0,
                    "crest_factor_db": 10.0,
                },
                "vocal_expression": {
                    "delivery_mode": "normal",
                    "auto_tags": {"primary": "Neutral/Dialogue", "status": "suggestion_only"},
                },
                "human_review": {"reviewed": False},
                "qc": {
                    "tier": "GOLD",
                    "reasons": [],
                    "usable_for_training": True,
                    "usable_for_reference": True,
                },
            }
            paths_single = generate_enrichment_report([single_rec], out_single)
            for p in paths_single.values():
                self.assertTrue(p.is_file(), str(p))

            out_voiced_single = Path(tmpdir) / "voiced_single_report"
            voiced_rec = {
                **single_rec,
                "prosody": {
                    "f0_median_hz": 240.0,
                    "f0_iqr_hz": 20.0,
                    "voiced_fraction": 0.85,
                    "spectral_centroid_hz": 1400.0,
                    "spectral_flatness": 0.03,
                    "energy_std_db": 4.0,
                    "crest_factor_db": 12.0,
                },
            }
            paths_voiced = generate_enrichment_report([voiced_rec], out_voiced_single)
            for p in paths_voiced.values():
                self.assertTrue(p.is_file(), str(p))


if __name__ == "__main__":
    unittest.main()
