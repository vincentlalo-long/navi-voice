#!/usr/bin/env python3
"""
Boundary Verification Tool - Acoustic Edge Energy & Clipping Inspector

Mathematically inspects audio slices to verify whether dialogue cuts occurred
smoothly inside silence intervals or accidentally clipped active speech phonemes.

Metrics evaluated per slice:
1. Edge Start RMS (dB): Energy in the first 25ms. Clean cuts must be < -30 dB.
2. Edge End RMS (dB): Energy in the last 25ms.
3. Edge Energy Ratio (EER): Ratio of edge energy to core utterance energy.
4. Transient Step Jump: First sample amplitude |x[0]| (checks for click artifacts).
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# Ensure voice root directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
VOICE_ROOT = SCRIPT_DIR.parent
if str(VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_ROOT))

from modules.audio_utils import evaluate_boundary_quality


def inspect_slice_boundaries(audio: np.ndarray, sampling_rate: int, edge_ms: float = 25.0) -> Dict[str, Any]:
    """Calculate acoustic edge statistics using unified evaluate_boundary_quality."""
    return evaluate_boundary_quality(audio, sampling_rate, edge_ms=edge_ms)


def main():
    parser = argparse.ArgumentParser(description="Inspect audio slices for boundary clipping")
    parser.add_argument(
        "--slices-dir",
        type=str,
        default="datasets/processed/energy/slices",
        help="Path to folder containing sliced WAV files (default: datasets/processed/energy/slices)",
    )
    parser.add_argument(
        "--edge-ms",
        type=float,
        default=25.0,
        help="Length of edge window in milliseconds (default: 25ms)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path to export JSON boundary evaluation report (e.g. reports/boundaries/boundary_report.json)",
    )
    args = parser.parse_args()

    slices_path = (VOICE_ROOT / args.slices_dir).resolve() if not Path(args.slices_dir).is_absolute() else Path(args.slices_dir)
    if not slices_path.is_dir():
        print(f"[!] Directory not found: {slices_path}")
        sys.exit(1)

    wav_files = sorted(list(slices_path.glob("*.wav")))
    if not wav_files:
        print(f"[!] No WAV files found in {slices_path}")
        sys.exit(0)

    print("=" * 70)
    print("ACOUSTIC BOUNDARY VERIFICATION (PHONEME CLIPPING DETECTOR)")
    print("=" * 70)
    print(f"Target directory: {slices_path}")
    print(f"Total slices:     {len(wav_files)} files")
    print(f"Edge window:      {args.edge_ms} ms")
    print("=" * 70)

    results = []
    clean_count = 0
    safe_count = 0
    warn_count = 0
    warnings: List[Tuple[str, Dict[str, float]]] = []

    for wav_file in tqdm(wav_files, desc="Inspecting boundaries"):
        data, sampling_rate = sf.read(str(wav_file), dtype="float32")
        metrics = inspect_slice_boundaries(data, sampling_rate, edge_ms=args.edge_ms)
        metrics["filename"] = wav_file.name
        results.append(metrics)

        if metrics["status"] == "CLEAN":
            clean_count += 1
        elif metrics["status"] == "SAFE":
            safe_count += 1
        else:
            warn_count += 1
            warnings.append((wav_file.name, metrics))

    total = len(wav_files)
    clean_pct = (clean_count / total) * 100
    safe_pct = (safe_count / total) * 100
    warn_pct = (warn_count / total) * 100

    print("\n" + "=" * 50)
    print("BOUNDARY VERIFICATION REPORT:")
    print("=" * 50)
    print(f"  * [CLEAN] (Quiet silence onset/offset):  {clean_count:3d} ({clean_pct:.1f}%)")
    print(f"  * [SAFE]  (Gentle margin buffer onset):   {safe_count:3d} ({safe_pct:.1f}%)")
    print(f"  * [WARN]  (Potential mid-speech cut):     {warn_count:3d} ({warn_pct:.1f}%)")
    print("=" * 50)

    if warnings:
        print("\n[!] WARNING DETAILS (Files flagged for manual inspection):")
        for fname, m in warnings[:10]:
            print(f"    - {fname:20s} | Start: {m['start_rms_db']} dB | End: {m['end_rms_db']} dB | Core: {m['core_rms_db']} dB")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings) - 10} more files.")
    else:
        print("\n[✓] EXCELLENT: 100% of tested slices have smooth silence margins at boundaries!")
        print("    No phoneme clipping or transient jumps detected.")

    if args.output_report:
        import json
        out_path = (VOICE_ROOT / args.output_report).resolve() if not Path(args.output_report).is_absolute() else Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "slices_dir": str(slices_path),
            "total_slices": total,
            "clean_count": clean_count,
            "clean_percent": round(clean_pct, 2),
            "safe_count": safe_count,
            "safe_percent": round(safe_pct, 2),
            "warn_count": warn_count,
            "warn_percent": round(warn_pct, 2),
            "details": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[+] Saved boundary verification report to: {out_path}")


if __name__ == "__main__":
    main()
