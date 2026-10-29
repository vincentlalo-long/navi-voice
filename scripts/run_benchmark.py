#!/usr/bin/env python3
"""
Benchmark Runner CLI - Multi-Slicer Comparative Evaluation.
Evaluates Energy, Silero, WebRTC, and WhisperX against Boundary Quality,
Speech Coverage, Overlaps, and Processing Speed (RTF).
"""

import argparse
import sys
import json
from pathlib import Path

# Add voice root to sys.path
VOICE_ROOT = Path(__file__).resolve().parent.parent
if str(VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_ROOT))

import modules.cuda_utils
from modules.evaluation import BenchmarkSuite


def main():
    parser = argparse.ArgumentParser(description="Multi-Algorithm Slicer Benchmark Suite")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="datasets/raw",
        help="Directory containing source raw audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/benchmarks",
        help="Directory where benchmark reports and metrics will be saved",
    )
    parser.add_argument(
        "--slicers",
        nargs="+",
        default=["energy", "silero", "webrtc", "whisper", "hybrid"],
        help="List of slicers to compare: energy, silero, webrtc, whisper, hybrid",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=None,
        help="Optional dry run duration in seconds per file (default: None, processes full audio)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Whisper model size for benchmark (default: 'base' or from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Base configuration path",
    )
    args = parser.parse_args()

    raw_path = (VOICE_ROOT / args.raw_dir).resolve()
    out_path = (VOICE_ROOT / args.output_dir).resolve()
    cfg_path = (VOICE_ROOT / args.config).resolve()

    cfg = {}
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    audio_cfg = cfg.get("audio", {})
    asr_cfg = cfg.get("asr", {})

    device = str(asr_cfg.get("device", "cuda"))
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("[!] Warning: CUDA is not available. Falling back to CPU.")
            device = "cpu"

    bench_cfg = audio_cfg.copy()
    model_size = args.model_size or str(audio_cfg.get("whisper_model_size", "base"))
    bench_cfg.update({
        "device": device,
        "model_size": model_size,
        "compute_type": str(asr_cfg.get("compute_type", "float16")),
        "language": str(asr_cfg.get("language", "ja")),
    })
    bench_cfg.setdefault("whisper_model_size", model_size)

    raw_files = sorted(
        list(raw_path.glob("*.wav")) + list(raw_path.glob("*.mp3")) + list(raw_path.glob("*.flac"))
    )
    if not raw_files:
        print(f"[!] Error: No audio files found in {raw_path}")
        sys.exit(1)

    sample_desc = f"{args.sample_seconds}s/file" if args.sample_seconds else "FULL (100%)"

    print("=" * 88)
    print(f"AMADEUS VOICE BENCHMARK | Target: {len(raw_files)} files | Mode: {sample_desc} | Slicers: {args.slicers}")
    print("=" * 88)

    suite = BenchmarkSuite(config=bench_cfg, output_dir=out_path)
    results = suite.run_full_benchmark(
        args.slicers, raw_files, sample_seconds=args.sample_seconds
    )

    print("\n" + suite.format_terminal_table(results))
    print(f"[✓] Artifacts successfully saved to: {out_path}/")
    print(f"    - PNG Charts: benchmark_comparison.png, boundary_acoustic_detail.png, duration_distribution.png")
    print(f"    - Reports:    benchmark_report.md, benchmark_summary.json")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
