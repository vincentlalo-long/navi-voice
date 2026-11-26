import math
import unittest
import numpy as np
from modules.tagging.prosody import (
    PROSODY_EXTRACTOR_VERSION,
    assign_speaker_prosody_bins,
    extract_prosody_features,
)


class TestProsody(unittest.TestCase):
    def test_voiced_sine_wave_tracks_fundamental_frequency(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # 240 Hz fundamental + 480 Hz harmonic
        wave = 0.5 * np.sin(2 * np.pi * 240.0 * t) + 0.2 * np.sin(2 * np.pi * 480.0 * t)
        feat = extract_prosody_features(wave, sr)
        self.assertEqual(PROSODY_EXTRACTOR_VERSION, "1.0.0")
        self.assertGreater(feat["voiced_fraction"], 0.85)
        self.assertAlmostEqual(feat["f0_median_hz"], 240.0, delta=12.0)
        self.assertLess(feat["spectral_flatness"], 0.05)

    def test_white_noise_has_low_voicing_and_high_spectral_flatness(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.1, 44100).astype(np.float32)
        feat = extract_prosody_features(noise, 44100)
        self.assertLess(feat["voiced_fraction"], 0.20)
        self.assertGreater(feat["spectral_flatness"], 0.25)

    def test_silent_or_dc_audio_returns_finite_defaults_and_rejected_slices_do_not_skew_bins(self):
        for arr in (
            np.zeros(0, dtype=np.float32),
            np.zeros(500, dtype=np.float32),
            np.full(44100, 0.05, dtype=np.float32),
            np.array([np.nan, 0.1, -0.1] * 1000, dtype=np.float32),
            np.array([np.inf, -np.inf] * 1000, dtype=np.float32),
        ):
            feat = extract_prosody_features(arr, 44100)
            self.assertEqual(
                set(feat.keys()),
                {
                    "f0_median_hz",
                    "f0_iqr_hz",
                    "f0_slope_hz_per_sec",
                    "voiced_fraction",
                    "spectral_centroid_hz",
                    "spectral_flatness",
                    "energy_std_db",
                    "crest_factor_db",
                },
            )
            self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in feat.values()))
            self.assertEqual(feat["voiced_fraction"], 0.0)
            self.assertEqual(feat["f0_median_hz"], 0.0)

        # 5 valid speech clips (200..280 Hz) + 3 REJECT noise outliers (540 Hz, -60 dB)
        records = []
        for idx, f0 in enumerate([200.0, 220.0, 240.0, 260.0, 280.0]):
            records.append({
                "raw_rms_db": -25.0 + idx * 2.0,
                "asr_diagnostics": {"effective_mora_rate": 5.0 + idx},
                "prosody": {"f0_median_hz": f0, "voiced_fraction": 0.8},
                "qc": {"tier": "GOLD"},
            })
        for _ in range(3):
            records.append({
                "raw_rms_db": -60.0,
                "asr_diagnostics": {"effective_mora_rate": 0.2},
                "prosody": {"f0_median_hz": 540.0, "voiced_fraction": 0.9},
                "qc": {"tier": "REJECT"},
            })

        quantiles = assign_speaker_prosody_bins(records)
        q = quantiles["default"]
        self.assertLess(q["pitch_p80"], 290.0)
        self.assertEqual(records[0]["prosody"]["bins"]["pitch"], "low")
        self.assertEqual(records[2]["prosody"]["bins"]["pitch"], "normal")
        self.assertEqual(records[4]["prosody"]["bins"]["pitch"], "high")

    def test_f0_slope_last_250ms_and_multichannel_and_spectral_metrics(self):
        sr = 16000
        dur = 0.6
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        # Constant 180 Hz for first 0.35s, then linear chirp from 180 Hz to 300 Hz over last 0.25s
        f_inst = np.where(t < 0.35, 180.0, 180.0 + (300.0 - 180.0) * (t - 0.35) / 0.25)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
        mono = 0.5 * np.sin(phase).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1)  # (N, 2)

        feat_mono = extract_prosody_features(mono, sr)
        feat_stereo = extract_prosody_features(stereo, sr)

        self.assertGreater(feat_mono["f0_slope_hz_per_sec"], 200.0)
        self.assertGreater(feat_mono["f0_iqr_hz"], 0.0)
        self.assertAlmostEqual(feat_mono["f0_median_hz"], feat_stereo["f0_median_hz"], delta=1e-3)
        self.assertAlmostEqual(feat_mono["crest_factor_db"], 3.01, delta=0.3)
        self.assertAlmostEqual(feat_mono["spectral_centroid_hz"], 210.0, delta=60.0)
        self.assertGreaterEqual(feat_mono["energy_std_db"], 0.0)

    def test_f0_slope_is_zero_when_tail_250ms_has_fewer_than_two_voiced_frames(self):
        sr = 16000
        # 0.25s voiced tone followed by 0.35s unvoiced white noise (active tail has no voicing)
        t_voice = np.linspace(0, 0.25, int(0.25 * sr), endpoint=False, dtype=np.float32)
        voiced = 0.5 * np.sin(2 * np.pi * 220.0 * t_voice)
        rng = np.random.default_rng(7)
        noise_tail = rng.normal(0.0, 0.15, int(0.35 * sr)).astype(np.float32)
        wave = np.concatenate([voiced, noise_tail])
        feat = extract_prosody_features(wave, sr)
        self.assertAlmostEqual(feat["f0_median_hz"], 220.0, delta=10.0)
        self.assertEqual(feat["f0_slope_hz_per_sec"], 0.0)

    def test_assign_speaker_bins_multispk_unvoiced_fallbacks_and_empty(self):
        self.assertEqual(assign_speaker_prosody_bins([]), {})

        records = [
            # Speaker A: 3 voiced + 1 unvoiced
            {
                "speaker": "alice",
                "rms_db": -26.0,
                "chars_per_sec": 3.0,
                "prosody": {"f0_median_hz": 190.0, "voiced_fraction": 0.7},
                "qc": {"tier": "SILVER"},
            },
            {
                "speaker": "alice",
                "rms_db": -20.0,
                "asr_diagnostics": {"effective_cps": 5.0},
                "prosody": {"f0_median_hz": 220.0, "voiced_fraction": 0.7},
                "qc": {"tier": "GOLD"},
            },
            {
                "speaker": "alice",
                "rms_db": -14.0,
                "asr_diagnostics": {"effective_mora_rate": 8.0},
                "prosody": {"f0_median_hz": 260.0, "voiced_fraction": 0.8},
                "qc": {"tier": "GOLD"},
            },
            {
                "speaker": "alice",
                "rms_db": -20.0,
                "asr_diagnostics": {"effective_mora_rate": 5.0},
                "prosody": {"f0_median_hz": 0.0, "voiced_fraction": 0.05},
                "qc": {"tier": "GOLD"},
            },
            # Speaker B: only unvoiced / no prosody dict
            {
                "speaker": "bob",
                "qc": {"tier": "GOLD"},
            },
        ]
        quantiles = assign_speaker_prosody_bins(records)
        self.assertIn("alice", quantiles)
        self.assertIn("bob", quantiles)
        self.assertEqual(records[0]["prosody"]["bins"], {"pitch": "low", "speed": "low", "energy": "low"})
        self.assertEqual(records[2]["prosody"]["bins"], {"pitch": "high", "speed": "high", "energy": "high"})
        self.assertEqual(records[3]["prosody"]["bins"]["pitch"], "unvoiced")
        self.assertEqual(records[4]["prosody"]["bins"], {"pitch": "unvoiced", "speed": "normal", "energy": "normal"})
        self.assertEqual(quantiles["bob"], {
            "pitch_p20": 0.0,
            "pitch_p80": 0.0,
            "speed_p20": 0.0,
            "speed_p80": 0.0,
            "energy_p20": 0.0,
            "energy_p80": 0.0,
        })

    def test_package_reexports(self):
        import modules.tagging as tagging

        self.assertTrue(hasattr(tagging, "CategoryClassifier"))
        self.assertIs(tagging.PROSODY_EXTRACTOR_VERSION, PROSODY_EXTRACTOR_VERSION)
        self.assertIs(tagging.extract_prosody_features, extract_prosody_features)
        self.assertIs(tagging.assign_speaker_prosody_bins, assign_speaker_prosody_bins)


if __name__ == "__main__":
    unittest.main()
