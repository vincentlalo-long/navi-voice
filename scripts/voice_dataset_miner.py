#!/usr/bin/env python3
"""
Voice Dataset Miner - High-Level Audio Segmentation, Normalization & ASR Pipeline.

Architecture:
- Slicing Strategy Pattern: Supports multiple VAD algorithms ('energy', 'silero').
- Acoustic Quality Control: Peak normalization, RMS, clipping rate, and boundary edge checks.
- GPU-Accelerated ASR: Faster-Whisper Large-V3 for transcription and confidence scores.
- Dynamic Tagging: Rule-based acoustic/semantic categorization.
- Standard Exports: JSON, CSV, and TTS .list formats.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Ensure voice root directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
VOICE_ROOT = SCRIPT_DIR.parent
if str(VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_ROOT))

from modules.audio_utils import (
    apply_raised_cosine_fade,
    compute_audio_stats,
    compute_edge_metrics,
    evaluate_boundary_quality,
    normalize_peak,
)
from modules.classifier import CategoryClassifier
from modules.slicers import get_slicer


# ==============================================================================
# DYNAMIC CUDA PRELOADER
# ==============================================================================
def preload_nvidia_cuda_libraries():
    """
    Dynamically preloads bundled NVIDIA CUDA shared libraries from Python site-packages
    into global process memory (mode=RTLD_GLOBAL) so CTranslate2 can resolve dependencies
    (e.g., libcublas.so.12, libcudnn.so) without manual LD_LIBRARY_PATH environment setup.
    """
    import ctypes
    import glob

    target_prefixes = ("libcublas.so", "libcublasLt.so", "libcudnn.so")
    for site_pkg in sys.path:
        for so in glob.glob(os.path.join(site_pkg, "nvidia", "**", "lib", "*.so*"), recursive=True):
            if "nvblas" in so:
                continue
            base = os.path.basename(so)
            if any(base.startswith(p) for p in target_prefixes):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass


# ==============================================================================
# CONFIGURATION & PROFILE LOADER
# ==============================================================================
def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON configuration file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay dictionary into base dictionary."""
    result = base.copy()
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def resolve_config(config_arg: str, profile_arg: Optional[str], root_dir: Path) -> Dict[str, Any]:
    """
    Resolve base configuration and optional profile overlays.
    Profiles can be specified as a direct path or as a name in configs/profiles/<name>.json.
    """
    base_cfg_path = Path(config_arg).resolve()
    config = load_json_file(base_cfg_path)

    if profile_arg:
        profile_path = Path(profile_arg)
        if not profile_path.is_file():
            profile_path = root_dir / "configs" / "profiles" / f"{profile_arg}.json"
        if profile_path.is_file():
            print(f"[i] Applying profile overlay: {profile_path.name}")
            profile_data = load_json_file(profile_path)
            config = merge_dicts(config, profile_data)
        else:
            print(f"[!] Warning: Profile '{profile_arg}' not found at {profile_path}. Proceeding with base config.")

    return config


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    default_config_path = VOICE_ROOT / "configs" / "default_config.json"

    parser = argparse.ArgumentParser(
        description="Voice Dataset Miner - Audio Slicing, Normalization & ASR Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to base pipeline configuration JSON file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile name (e.g. 'kurisu') or path to overlay profile JSON",
    )
    parser.add_argument(
        "--slicer",
        type=str,
        default=None,
        choices=["energy", "silero", "webrtc", "whisper", "whisperx"],
        help="Choose slicing algorithm: 'energy', 'silero', 'webrtc', or 'whisper' (semantic)",
    )
    parser.add_argument(
        "--sample-seconds",
        type=int,
        default=None,
        help="Dry run: process only the first N seconds of each raw file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Override Whisper model size (e.g. large-v3, large-v3-turbo, medium)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override execution device (cuda or cpu)",
    )
    parser.add_argument(
        "--skip-slicing",
        action="store_true",
        help="Skip the audio slicing step if slices are already generated",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Only slice and compute audio stats, skip Whisper transcription",
    )
    args = parser.parse_args()

    # Load and resolve configuration with optional profile overlay
    config = resolve_config(args.config, args.profile, VOICE_ROOT)

    # Resolve paths
    paths_cfg = config.get("paths", {})
    raw_dir = (VOICE_ROOT / paths_cfg.get("raw_dir", "datasets/raw")).resolve()
    output_dir = (VOICE_ROOT / paths_cfg.get("output_dir", "datasets/processed")).resolve()
    slices_subfolder = paths_cfg.get("slices_subfolder", "slices")

    metadata_json_name = paths_cfg.get("metadata_json", "metadata.json")
    metadata_csv_name = paths_cfg.get("metadata_csv", "metadata.csv")
    dataset_list_name = paths_cfg.get("dataset_list", "dataset.list")

    # Speaker parameters
    speaker_cfg = config.get("speaker", {})
    speaker_name = speaker_cfg.get("name", "Speaker")
    speaker_lang = speaker_cfg.get("language", "JA")
    file_prefix = speaker_cfg.get("file_prefix", "audio")

    # Audio & Slicer parameters
    audio_cfg = config.get("audio", {})
    sampling_rate = int(audio_cfg.get("sample_rate", 44100))
    target_peak_db = float(audio_cfg.get("target_peak_db", -1.0))

    # Slicer Strategy Resolution
    slicer_method = args.slicer or audio_cfg.get("slicer_method", "energy")
    slicer = get_slicer(slicer_method, audio_cfg)

    # Separate output directory per slicer algorithm for clean, un-overwritten A/B benchmarking
    slicer_output_dir = output_dir / slicer_method
    slices_dir = slicer_output_dir / slices_subfolder
    slices_dir.mkdir(parents=True, exist_ok=True)

    # ASR parameters
    asr_cfg = config.get("asr", {})
    model_size = args.model_size or asr_cfg.get("model_size", "large-v3")
    device = args.device or asr_cfg.get("device", "cuda")
    compute_type = asr_cfg.get("compute_type", "float16")
    language = asr_cfg.get("language", "ja")
    beam_size = int(asr_cfg.get("beam_size", 5))

    # Semantic / Emotion Classifier
    classifier = CategoryClassifier(config.get("emotion_rules", {}))

    print("=" * 70)
    print("VOICE DATASET MINER - PIPELINE INITIALIZED")
    print("=" * 70)
    print(f"Base Config:        {args.config}")
    if args.profile:
        print(f"Active Profile:     {args.profile}")
    print(f"Slicing Algorithm:  {slicer_method.upper()} ({slicer.__class__.__name__})")
    print(f"Raw Input Dir:      {raw_dir}")
    print(f"Processed Dir:      {slicer_output_dir}")
    print(f"Speaker Tag:        {speaker_name} [{speaker_lang}]")
    print(f"Slice Prefix:       {file_prefix}_XXXXX.wav")
    print(f"ASR Model:          {model_size} ({device}, {compute_type})")
    if args.sample_seconds:
        print(f"Dry-run Mode:       First {args.sample_seconds} seconds only")
    print("=" * 70)

    # Discover raw audio files
    raw_files = sorted(list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.flac")))
    if not raw_files:
        print(f"[!] Error: No audio files found in {raw_dir}")
        sys.exit(1)

    metadata_records = []
    global_slice_idx = 0

    # --------------------------------------------------------------------------
    # STEP 1 & 2: SLICING & NORMALIZATION
    # --------------------------------------------------------------------------
    if not args.skip_slicing:
        print(f"\n>>> STEP 1: Slicing audio files using [{slicer_method.upper()}] and normalizing peak volume...")
        for raw_path in raw_files:
            print(f"\n[*] Slicing source: {raw_path.name}")
            data, file_sampling_rate = sf.read(str(raw_path), dtype="float32")

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Polyphase anti-aliased resampling if input file sample rate differs from target
            if file_sampling_rate != sampling_rate:
                import scipy.signal
                gcd = np.gcd(file_sampling_rate, sampling_rate)
                up = sampling_rate // gcd
                down = file_sampling_rate // gcd
                data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
                print(f"    [i] Polyphase resampled from {file_sampling_rate}Hz to {sampling_rate}Hz (L={up}, M={down})")
                file_sampling_rate = sampling_rate

            if args.sample_seconds:
                max_samples = sampling_rate * args.sample_seconds
                data = data[:max_samples]
                print(f"    (Truncated to first {args.sample_seconds}s for dry-run)")

            chunks = slicer.slice(data, sampling_rate)
            print(f"    -> Extracted {len(chunks)} valid dialogue slices.")

            for start_idx, end_idx in tqdm(chunks, desc=f"    Writing {raw_path.stem}"):
                chunk_audio = data[start_idx:end_idx]
                faded_audio = apply_raised_cosine_fade(chunk_audio, sampling_rate, fade_ms=5.0)
                norm_audio = normalize_peak(faded_audio, target_peak_db=target_peak_db, max_gain_db=18.0)

                slice_filename = f"{file_prefix}_{global_slice_idx:05d}.wav"
                slice_path = slices_dir / slice_filename

                sf.write(str(slice_path), norm_audio, sampling_rate, subtype="PCM_16")

                stats = compute_audio_stats(norm_audio, sampling_rate)
                edge_metrics = compute_edge_metrics(norm_audio, sampling_rate)

                metadata_records.append({
                    "id": global_slice_idx,
                    "filename": slice_filename,
                    "relative_path": f"{slicer_method}/{slices_subfolder}/{slice_filename}",
                    "absolute_path": str(slice_path),
                    "source_file": raw_path.name,
                    "start_time_sec": round(start_idx / sampling_rate, 3),
                    "end_time_sec": round(end_idx / sampling_rate, 3),
                    "slicer_method": slicer_method,
                    **stats,
                    **edge_metrics,
                    "text": "",
                    "chars_per_sec": 0.0,
                    "category": "",
                })
                global_slice_idx += 1
    else:
        print("\n[i] Slicing skipped (--skip-slicing). Reading existing slices...")
        existing_slices = sorted(list(slices_dir.glob("*.wav")))
        for idx, s_path in enumerate(existing_slices):
            data, file_sampling_rate = sf.read(str(s_path), dtype="float32")
            stats = compute_audio_stats(data, file_sampling_rate)
            edge_metrics = compute_edge_metrics(data, file_sampling_rate)
            metadata_records.append({
                "id": idx,
                "filename": s_path.name,
                "relative_path": f"{slices_subfolder}/{s_path.name}",
                "absolute_path": str(s_path),
                "source_file": "existing",
                "start_time_sec": 0,
                "end_time_sec": stats["duration_sec"],
                "slicer_method": "existing",
                **stats,
                **edge_metrics,
                "text": "",
                "chars_per_sec": 0.0,
                "category": "",
            })

    print(f"\n[+] Total dialogue slices prepared: {len(metadata_records)}")

    # --------------------------------------------------------------------------
    # STEP 3 & 4: WHISPER ASR & CATEGORY MINING
    # --------------------------------------------------------------------------
    if not args.skip_transcribe and metadata_records:
        print(f"\n>>> STEP 2: Initializing Faster-Whisper ({model_size}) on '{device}'...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[!] Error: 'faster-whisper' package is not installed.")
            print("    Run: uv pip install -r voice/requirements.txt")
            sys.exit(1)

        if device == "cuda":
            preload_nvidia_cuda_libraries()

        actual_compute = compute_type if device == "cuda" else "int8"
        try:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=actual_compute,
                local_files_only=True,
            )
        except Exception:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=actual_compute,
                local_files_only=False,
            )

        print("\n>>> STEP 3: Transcribing audio speech and classifying categories...")
        for item in tqdm(metadata_records, desc="    Transcribing"):
            audio_path = item["absolute_path"]
            segments, _ = model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=False,
            )
            transcription = "".join([seg.text for seg in segments]).strip()
            item["text"] = transcription

            cps = len(transcription) / (item["duration_sec"] + 1e-6)
            item["chars_per_sec"] = round(cps, 2)
            item["category"] = classifier.classify(transcription, item)

    # --------------------------------------------------------------------------
    # STEP 5: EXPORT METADATA & DATASET LISTS
    # --------------------------------------------------------------------------
    print("\n>>> STEP 4: Exporting metadata and dataset lists...")

    json_path = slicer_output_dir / metadata_json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
    print(f"    -> Exported JSON metadata: {json_path}")

    csv_path = slicer_output_dir / metadata_csv_name
    if metadata_records:
        keys = list(metadata_records[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_records)
        print(f"    -> Exported CSV spreadsheet: {csv_path}")

    list_path = slicer_output_dir / dataset_list_name
    with open(list_path, "w", encoding="utf-8") as f:
        for item in metadata_records:
            if item.get("text"):
                f.write(f"{item['absolute_path']}|{speaker_name}|{speaker_lang}|{item['text']}\n")
    print(f"    -> Exported Dataset list: {list_path}")

    # Category distribution summary
    if not args.skip_transcribe:
        category_counts: Dict[str, int] = {}
        for item in metadata_records:
            cat = item.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n" + "=" * 50)
        print("IDENTIFIED CATEGORIES SUMMARY:")
        print("=" * 50)
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  * {cat:25s}: {count:4d} utterances")
        print("=" * 50)

    print("\n[✓] Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
