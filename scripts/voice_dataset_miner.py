#!/usr/bin/env python3
"""
Voice Dataset Miner - High-Level Audio Segmentation, Normalization & Phase 3 Enrichment Pipeline.

Architecture:
- Slicing Strategy Pattern: Supports multiple VAD algorithms ('energy', 'silero', 'webrtc', 'whisper', 'hybrid').
- Acoustic Quality Control: Peak/EBU R128 normalization, RMS, clipping rate, and boundary edge checks.
- Phase 3 Enrichment: Hardened Faster-Whisper ASR with 4-gate anti-hallucination checks,
  pure-DSP prosody extraction, speaker quantile binning, orthogonal expression classification,
  3-tier QC (GOLD/SILVER/REJECT), and leakage-safe TTS packaging.
"""

import argparse
import gc
import json
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

from modules.audio import (
    apply_raised_cosine_fade,
    compute_audio_stats,
    compute_edge_metrics,
    normalize_loudness_ebur128,
    normalize_peak,
)
from modules.enrichment import (
    HardenedWhisperTranscriber,
    canonicalize_japanese_text,
    compute_file_sha256,
    compute_speaking_rates,
    evaluate_asr_gates,
    evaluate_slice_qc,
    export_enriched_dataset,
    finalize_record_v2,
    merge_cached_record,
    mirror_custom_export_paths,
    resolve_effective_emotion,
)
from modules.tagging import (
    PROSODY_EXTRACTOR_VERSION,
    TAGGER_VERSION,
    CategoryClassifier,
    assign_speaker_prosody_bins,
    extract_prosody_features,
)

_EDGE_METRIC_KEYS = (
    "start_rms_db",
    "end_rms_db",
    "core_rms_db",
    "eer_start",
    "eer_end",
    "step_jump",
    "status",
    "clean_boundary",
)


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
    base_cfg_path = Path(config_arg)
    if not base_cfg_path.is_absolute():
        base_cfg_path = (root_dir / base_cfg_path).resolve()
    else:
        base_cfg_path = base_cfg_path.resolve()
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


def _load_existing_metadata_map(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing metadata.json into a filename lookup dictionary."""
    if not meta_path.is_file():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            old_records = json.load(f)
        if isinstance(old_records, list):
            return {
                str(rec.get("filename")): rec
                for rec in old_records
                if isinstance(rec, dict) and rec.get("filename")
            }
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    [!] Warning: could not read existing metadata ({exc}); rebuilding provenance.")
    return {}


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON payload via a sibling temp file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _empty_asr_diagnostics(duration_sec: float) -> Dict[str, Any]:
    """Create a canonical empty-ASR diagnostics block for uncached slices."""
    rates = compute_speaking_rates("", float(duration_sec))
    return {
        "raw_text": "",
        "canonical_text": "",
        "avg_logprob": -99.0,
        "no_speech_prob": 1.0,
        "compression_ratio": 0.0,
        "mean_word_prob": 0.0,
        "speech_duration_sec": round(float(rates["speech_duration_sec"]), 4),
        "clean_char_count": int(rates["clean_char_count"]),
        "estimated_mora_count": float(rates["estimated_mora_count"]),
        "effective_cps": round(float(rates["effective_cps"]), 4),
        "effective_mora_rate": round(float(rates["effective_mora_rate"]), 4),
        "hallucination_flags": [],
    }


def _select_phase2_field(
    old_record: Dict[str, Any],
    key: str,
    fallback: Any,
    can_keep: bool,
) -> Any:
    """Return cached Phase 2 telemetry field when valid and non-None, else fallback."""
    if can_keep and old_record.get(key) is not None:
        return old_record[key]
    return fallback


def _prepare_slice_record(
    fresh_base: Dict[str, Any],
    old_record: Optional[Dict[str, Any]],
    slice_path: Path,
    waveform: np.ndarray,
    sampling_rate: int,
    speaker_name: str,
    model_size: str,
    retag_only: bool,
    explicit_model_size: Optional[str],
    f0_min_hz: float,
    f0_max_hz: float,
    audio_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Hash slice WAV, merge safe cache fields, attach speaker, and extract prosody."""
    resolved_sha256 = audio_sha256 or compute_file_sha256(slice_path)

    expected_asr_model = model_size
    if retag_only and not explicit_model_size and isinstance(old_record, dict):
        old_prov = old_record.get("provenance")
        if isinstance(old_prov, dict) and old_prov.get("asr_model"):
            expected_asr_model = str(old_prov["asr_model"])

    record, can_reuse_asr = merge_cached_record(
        fresh_base,
        old_record,
        resolved_sha256,
        expected_asr_model=expected_asr_model,
    )
    record["speaker"] = speaker_name

    prov = record.setdefault("provenance", {})
    prov["prosody_version"] = PROSODY_EXTRACTOR_VERSION
    prov["tagger_version"] = TAGGER_VERSION
    if not can_reuse_asr:
        prov.setdefault("asr_model", None)

    record["prosody"] = extract_prosody_features(
        waveform,
        sampling_rate,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
    )
    return record, can_reuse_asr


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    default_config_path = VOICE_ROOT / "configs" / "default_config.json"

    parser = argparse.ArgumentParser(
        description="Voice Dataset Miner - Audio Slicing, Normalization & Phase 3 Enrichment Pipeline"
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
        help="Optional profile name (e.g. 'my_speaker') or path to overlay profile JSON",
    )
    parser.add_argument(
        "--slicer",
        type=str,
        default=None,
        choices=["energy", "silero", "webrtc", "whisper", "whisperx", "hybrid"],
        help="Choose slicing algorithm: 'energy', 'silero', 'webrtc', 'whisper', or 'hybrid'",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default=None,
        choices=["peak", "ebu_r128"],
        help="Loudness normalization method: 'peak' (default) or 'ebu_r128' (-20 LUFS)",
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
        help="Override ASR Whisper model size (e.g. large-v3, large-v3-turbo, medium)",
    )
    parser.add_argument(
        "--slicer-model-size",
        type=str,
        default=None,
        help="Whisper model size used only for semantic slicing (default: audio.whisper_model_size or 'base')",
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
        help="Only slice/load audio and compute DSP telemetry, skip Whisper transcription",
    )
    parser.add_argument(
        "--retag-only",
        action="store_true",
        help="Implies --skip-slicing; reuse cached ASR by audio SHA-256 and recompute prosody, tags, QC, and exports on CPU without loading Whisper",
    )
    args = parser.parse_args()
    if args.retag_only:
        args.skip_slicing = True

    # Load and resolve configuration with optional profile overlay
    config = resolve_config(args.config, args.profile, VOICE_ROOT)

    # Resolve paths
    paths_cfg = config.get("paths", {})
    raw_dir = (VOICE_ROOT / paths_cfg.get("raw_dir", "datasets/raw")).resolve()
    output_dir = (VOICE_ROOT / paths_cfg.get("output_dir", "datasets/processed")).resolve()
    slices_subfolder = paths_cfg.get("slices_subfolder", "slices")

    metadata_json_name = str(paths_cfg.get("metadata_json") or "metadata.json")

    # Speaker parameters
    speaker_cfg = config.get("speaker", {})
    speaker_name = speaker_cfg.get("name", "Speaker")
    speaker_lang = speaker_cfg.get("language", "JA")
    file_prefix = speaker_cfg.get("file_prefix", "audio")

    # Audio & Slicer parameters
    audio_cfg = config.get("audio", {})
    sampling_rate = int(audio_cfg.get("sample_rate", 44100))
    target_peak_db = float(audio_cfg.get("target_peak_db", -1.0))

    # Enrichment / Prosody parameters
    enrichment_cfg = config.get("enrichment", {})
    prosody_cfg = enrichment_cfg.get("prosody", {}) if isinstance(enrichment_cfg, dict) else {}
    f0_min_hz = float(prosody_cfg.get("f0_min_hz", 75.0))
    f0_max_hz = float(prosody_cfg.get("f0_max_hz", 550.0))

    # ASR parameters
    asr_cfg = config.get("asr", {})
    model_size = args.model_size or asr_cfg.get("model_size", "large-v3")
    device = args.device or asr_cfg.get("device", "cuda")
    needs_compute_device = (not args.skip_slicing) or (not args.skip_transcribe and not args.retag_only)
    if device == "cuda" and needs_compute_device:
        import torch

        if not torch.cuda.is_available():
            print("[!] Warning: CUDA is not available on this machine. Falling back to CPU.")
            device = "cpu"
    elif args.retag_only:
        device = "cpu"
    compute_type = asr_cfg.get("compute_type", "float16")
    language = asr_cfg.get("language", "ja")
    beam_size = int(asr_cfg.get("beam_size", 5))

    # Slicing and transcription can use different Whisper sizes: the slicer
    # only needs word timings, while Phase 3 needs the strongest ASR model.
    slicer_method = args.slicer or audio_cfg.get("slicer_method", "energy")
    slicer_model_size = args.slicer_model_size or audio_cfg.get("whisper_model_size", "base")

    # Isolate runs by profile/speaker so two profiles can no longer overwrite
    # or clean each other's slices. The no-profile layout stays backwards
    # compatible: datasets/processed/<slicer>/slices.
    profile_tag = ""
    if args.profile:
        profile_tag = Path(args.profile).stem
        profile_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in profile_tag)
    slicer_output_dir = output_dir / slicer_method
    if profile_tag:
        slicer_output_dir = slicer_output_dir / profile_tag
    slices_dir = slicer_output_dir / slices_subfolder
    existing_meta_path = slicer_output_dir / metadata_json_name
    old_by_name = _load_existing_metadata_map(existing_meta_path)
    has_any_asr_cache = any(
        isinstance(rec.get("asr_diagnostics"), dict) for rec in old_by_name.values()
    )
    if args.retag_only and not has_any_asr_cache:
        print(
            f"[!] Error: Manifest at {existing_meta_path} has not been ASR-enriched yet "
            "(no cached records contain 'asr_diagnostics'). Run without --retag-only first."
        )
        sys.exit(1)
    slices_dir.mkdir(parents=True, exist_ok=True)

    raw_files: List[Path] = []
    slicer = None
    if not args.skip_slicing:
        raw_files = sorted(
            list(raw_dir.glob("*.wav"))
            + list(raw_dir.glob("*.mp3"))
            + list(raw_dir.glob("*.flac"))
        )
        if not raw_files:
            print(f"[!] Error: No audio files found in {raw_dir}")
            sys.exit(1)

        from modules.slicers import get_slicer

        slicer_cfg = audio_cfg.copy()
        slicer_cfg.update({
            "device": device,
            "model_size": slicer_model_size,
            "compute_type": compute_type,
            "language": language,
        })
        slicer = get_slicer(slicer_method, slicer_cfg)

    # Semantic / Emotion Classifier
    classifier = CategoryClassifier(config.get("emotion_rules", {}))

    slicer_desc = f"{slicer_method.upper()} ({slicer.__class__.__name__})" if slicer is not None else f"{slicer_method.upper()} (skipped)"
    print("=" * 70)
    print("VOICE DATASET MINER - PHASE 3 ENRICHMENT PIPELINE INITIALIZED")
    print("=" * 70)
    print(f"Base Config:        {args.config}")
    if args.profile:
        print(f"Active Profile:     {args.profile}")
    print(f"Slicing Algorithm:  {slicer_desc}")
    print(f"Raw Input Dir:      {raw_dir}")
    print(f"Processed Dir:      {slicer_output_dir}")
    print(f"Speaker Tag:        {speaker_name} [{speaker_lang}]")
    print(f"Slice Prefix:       {file_prefix}_XXXXX.wav")
    print(f"Slicer ASR Model:   {slicer_model_size}")
    print(f"ASR Model:          {model_size} ({device}, {compute_type})")
    if args.retag_only:
        print("Execution Mode:     RETAG-ONLY (CPU cache reuse, Whisper skipped)")
    elif args.skip_transcribe:
        print("Execution Mode:     SKIP-TRANSCRIBE (ASR skipped)")
    if args.sample_seconds:
        print(f"Dry-run Mode:       First {args.sample_seconds} seconds only")
    print("=" * 70)

    metadata_records: List[Dict[str, Any]] = []
    has_asr_flags: List[bool] = []
    global_slice_idx = 0

    # --------------------------------------------------------------------------
    # STEP 1 & 2: SLICING / LOADING SLICES & PROSODY EXTRACTION
    # --------------------------------------------------------------------------
    if not args.skip_slicing:
        print(f"\n>>> STEP 1: Slicing audio files using [{slicer_method.upper()}] and normalizing volume...")
        for raw_path in raw_files:
            print(f"\n[*] Slicing source: {raw_path.name}")
            data, file_sampling_rate = sf.read(str(raw_path), dtype="float32")

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

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

            effective_norm = args.norm_method or audio_cfg.get("normalization_method", "peak")
            if effective_norm not in ("peak", "ebu_r128"):
                print(f"    [!] Warning: Unknown normalization method '{effective_norm}'. Falling back to 'peak'.")
                effective_norm = "peak"
            target_lufs = float(audio_cfg.get("target_lufs", -20.0))

            for start_idx, end_idx in tqdm(chunks, desc=f"    Writing {raw_path.stem}"):
                chunk_audio = data[start_idx:end_idx]
                faded_audio = apply_raised_cosine_fade(chunk_audio, sampling_rate, fade_ms=5.0)
                if effective_norm == "ebu_r128":
                    norm_audio = normalize_loudness_ebur128(
                        faded_audio, sampling_rate, target_lufs=target_lufs, max_gain_db=18.0
                    )
                else:
                    norm_audio = normalize_peak(
                        faded_audio, target_peak_db=target_peak_db, max_gain_db=18.0
                    )

                slice_filename = f"{file_prefix}_{global_slice_idx:05d}.wav"
                slice_path = slices_dir / slice_filename
                sf.write(str(slice_path), norm_audio, sampling_rate, subtype="PCM_16")

                # Final-file telemetry (after fade + normalization) and raw
                # boundary telemetry (before either) are different things.
                # The classifier must use the raw level; Phase 3 needs the gain.
                stats = compute_audio_stats(norm_audio, sampling_rate)
                raw_stats = compute_audio_stats(chunk_audio, sampling_rate)
                edge_metrics = compute_edge_metrics(chunk_audio, sampling_rate)
                applied_gain_db = (
                    round(stats["rms_db"] - raw_stats["rms_db"], 2)
                    if raw_stats["rms_db"] > -90.0 and stats["rms_db"] > -90.0
                    else 0.0
                )

                fresh_base = {
                    "id": global_slice_idx,
                    "filename": slice_filename,
                    "relative_path": str(slice_path.relative_to(output_dir)),
                    "absolute_path": str(slice_path),
                    "source_file": raw_path.name,
                    "start_time_sec": round(start_idx / sampling_rate, 3),
                    "end_time_sec": round(end_idx / sampling_rate, 3),
                    "slicer_method": slicer_method,
                    **stats,
                    **edge_metrics,
                    "raw_rms_db": raw_stats["rms_db"],
                    "raw_peak_db": raw_stats["peak_db"],
                    "applied_gain_db": applied_gain_db,
                    "text": "",
                    "chars_per_sec": 0.0,
                    "category": "",
                }
                rec, reused = _prepare_slice_record(
                    fresh_base,
                    old_by_name.get(slice_filename),
                    slice_path,
                    norm_audio,
                    sampling_rate,
                    speaker_name,
                    model_size,
                    args.retag_only,
                    args.model_size,
                    f0_min_hz,
                    f0_max_hz,
                )
                metadata_records.append(rec)
                has_asr_flags.append(reused)
                global_slice_idx += 1

        # Idempotency: remove only this run's stale prefix, never another
        # profile's files and never a dry-run's other slices.
        if not args.sample_seconds:
            written_filenames = {record["filename"] for record in metadata_records}
            stale_slices = [
                p for p in slices_dir.glob(f"{file_prefix}_*.wav")
                if p.name not in written_filenames
            ]
            for stale_path in stale_slices:
                stale_path.unlink()
            if stale_slices:
                print(f"    [i] Removed {len(stale_slices)} stale slice file(s) from a previous run.")
    else:
        print("\n[i] Slicing skipped (--skip-slicing). Reading existing slices & extracting prosody...")
        existing_slices = sorted(list(slices_dir.glob("*.wav")))
        if not existing_slices:
            print(f"[!] Error: No slice files found in {slices_dir}")
            sys.exit(1)

        for idx, s_path in enumerate(tqdm(existing_slices, desc="    Inspecting slices")):
            data, file_sampling_rate = sf.read(str(s_path), dtype="float32")
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            stats = compute_audio_stats(data, file_sampling_rate)
            old = old_by_name.get(s_path.name, {})
            audio_sha256 = compute_file_sha256(s_path)

            # Preserve Phase 2 pre-fade/pre-gain boundary telemetry when the WAV matches
            # or upgrades a legacy Phase 2 manifest; recompute/reset only if WAV changed.
            old_prov = old.get("provenance") if isinstance(old.get("provenance"), dict) else {}
            old_sha = old_prov.get("audio_sha256")
            can_keep_phase2_telemetry = not old_sha or old_sha == audio_sha256
            can_keep_phase2_edges = bool(old.get("status")) and can_keep_phase2_telemetry
            computed_edges = compute_edge_metrics(data, file_sampling_rate)
            if can_keep_phase2_edges:
                edge_metrics = {
                    key: old[key] if old.get(key) is not None else computed_edges[key]
                    for key in _EDGE_METRIC_KEYS
                }
            else:
                edge_metrics = {key: computed_edges[key] for key in _EDGE_METRIC_KEYS}

            fresh_base = {
                "id": old.get("id") if old.get("id") is not None else idx,
                "filename": s_path.name,
                "relative_path": old.get("relative_path") or str(s_path.relative_to(output_dir)),
                "absolute_path": str(s_path),
                "source_file": _select_phase2_field(old, "source_file", "existing", can_keep_phase2_telemetry),
                # Unknown offsets stay None; zero would falsely imply 0.000s.
                "start_time_sec": _select_phase2_field(old, "start_time_sec", None, can_keep_phase2_telemetry),
                "end_time_sec": _select_phase2_field(old, "end_time_sec", None, can_keep_phase2_telemetry),
                "slicer_method": old.get("slicer_method") or slicer_method,
                **stats,
                **edge_metrics,
                "raw_rms_db": _select_phase2_field(old, "raw_rms_db", stats["rms_db"], can_keep_phase2_telemetry),
                "raw_peak_db": _select_phase2_field(old, "raw_peak_db", stats["peak_db"], can_keep_phase2_telemetry),
                "applied_gain_db": _select_phase2_field(old, "applied_gain_db", 0.0, can_keep_phase2_telemetry),
                "text": "",
                "chars_per_sec": 0.0,
                "category": "",
            }
            rec, reused = _prepare_slice_record(
                fresh_base,
                old if old else None,
                s_path,
                data,
                file_sampling_rate,
                speaker_name,
                model_size,
                args.retag_only,
                args.model_size,
                f0_min_hz,
                f0_max_hz,
                audio_sha256=audio_sha256,
            )
            metadata_records.append(rec)
            has_asr_flags.append(reused)

    print(f"\n[+] Total dialogue slices prepared: {len(metadata_records)}")

    # Release the slicer model and full waveform before loading ASR. Without this,
    # whisper/hybrid keeps a Whisper model alive while a second ASR model loads,
    # and the full source waveform (~2 GB for a 3-hour 44.1 kHz file) stays in RAM.
    if not args.skip_slicing:
        data = None
        chunks = None
    if slicer is not None:
        slicer.release()
        slicer = None
    gc.collect()

    # --------------------------------------------------------------------------
    # STEP 3: HARDENED WHISPER ASR (INCREMENTAL & CACHE-AWARE)
    # --------------------------------------------------------------------------
    indices_to_transcribe = [
        i
        for i, reused in enumerate(has_asr_flags)
        if not args.skip_transcribe and not args.retag_only and not reused
    ]
    reused_count = sum(1 for flag in has_asr_flags if flag)
    if reused_count:
        print(f"[i] Reused cached ASR diagnostics for {reused_count} slice(s) matching SHA-256.")
    if args.retag_only and reused_count < len(metadata_records):
        uncached_count = len(metadata_records) - reused_count
        print(
            f"[!] Warning: --retag-only left {uncached_count} of {len(metadata_records)} "
            "slice(s) without reusable ASR cache (WAV changed or ASR model mismatch); "
            "uncached slices will be marked empty ASR."
        )

    if indices_to_transcribe and metadata_records:
        print(
            f"\n>>> STEP 2: Initializing HardenedWhisperTranscriber ({model_size}) "
            f"on '{device}' for {len(indices_to_transcribe)} slice(s)..."
        )
        transcriber = HardenedWhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
        )
        checkpoint_every = 50
        try:
            for step_num, idx in enumerate(
                tqdm(indices_to_transcribe, desc="    Transcribing"), start=1
            ):
                rec = metadata_records[idx]
                asr_diag = transcriber.transcribe_file(
                    Path(rec["absolute_path"]),
                    float(rec.get("duration_sec", 0.0)),
                )
                rec["asr_diagnostics"] = asr_diag
                rec.setdefault("provenance", {})["asr_model"] = model_size
                has_asr_flags[idx] = True

                if step_num % checkpoint_every == 0:
                    _atomic_write_json(existing_meta_path, metadata_records)
        except Exception:
            _atomic_write_json(existing_meta_path, metadata_records)
            raise
        finally:
            transcriber.release()

    # --------------------------------------------------------------------------
    # STEP 4: ASR GATES, SPEAKER PROSODY BINNING, TAGGING & 3-TIER QC
    # --------------------------------------------------------------------------
    if metadata_records:
        print("\n>>> STEP 3: Evaluating ASR gates, speaker prosody bins, vocal tags, and QC tiers...")
        for rec, has_asr in zip(metadata_records, has_asr_flags):
            if has_asr or args.retag_only:
                asr_diag = rec.get("asr_diagnostics")
                if not isinstance(asr_diag, dict):
                    asr_diag = _empty_asr_diagnostics(float(rec.get("duration_sec", 0.0)))
                    rec["asr_diagnostics"] = asr_diag
                elif asr_diag.get("canonical_text") is None:
                    raw_text = str(asr_diag.get("raw_text") or "")
                    canonical_text = canonicalize_japanese_text(raw_text)
                    speech_dur = float(asr_diag.get("speech_duration_sec") or rec.get("duration_sec") or 0.2)
                    rates = compute_speaking_rates(canonical_text, speech_dur)
                    asr_diag["canonical_text"] = canonical_text
                    asr_diag.update(rates)

                asr_diag["hallucination_flags"] = evaluate_asr_gates(
                    asr_diag,
                    float(rec.get("duration_sec", 0.0)),
                    cfg=config,
                )
                rec["qc"] = evaluate_slice_qc(rec, cfg=config)
            else:
                rec["qc"] = {
                    "tier": "PENDING_ASR",
                    "reasons": [],
                    "usable_for_training": False,
                    "usable_for_reference": False,
                }

        assign_speaker_prosody_bins(metadata_records)

        for rec, has_asr in zip(metadata_records, has_asr_flags):
            if has_asr or args.retag_only:
                canonical_text = str(rec["asr_diagnostics"].get("canonical_text") or "")
                rec["vocal_expression"] = classifier.classify_rich(
                    canonical_text,
                    rec,
                    prosody=rec.get("prosody"),
                )
                rec["qc"] = evaluate_slice_qc(rec, cfg=config)
            finalize_record_v2(rec)

    # --------------------------------------------------------------------------
    # STEP 5: EXPORT ENRICHED DATASET & REPORTS
    # --------------------------------------------------------------------------
    print("\n>>> STEP 4: Exporting Schema v2.0 metadata, splits, manifest, and reference prompts...")
    exported_paths = export_enriched_dataset(
        metadata_records,
        slicer_output_dir,
        speaker_name,
        speaker_lang,
        cfg=config,
    )
    mirror_custom_export_paths(exported_paths, slicer_output_dir, paths_cfg)

    for label, p in exported_paths.items():
        print(f"    -> {label:22s}: {p}")

    if metadata_records:
        tier_counts: Dict[str, int] = {}
        ref_count = 0
        category_counts: Dict[str, int] = {}
        for item in metadata_records:
            qc = item.get("qc") if isinstance(item.get("qc"), dict) else {}
            tier = str(qc.get("tier") or "UNKNOWN")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if qc.get("usable_for_reference") is True:
                ref_count += 1
            if qc.get("usable_for_training") is True or args.skip_transcribe:
                cat, _, _ = resolve_effective_emotion(item)
                category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n" + "=" * 56)
        print("PHASE 3 QC & EXPRESSION SUMMARY:")
        print("=" * 56)
        for tier_name in ("GOLD", "SILVER", "REJECT", "PENDING_ASR"):
            if tier_name in tier_counts:
                print(f"  * QC {tier_name:18s}: {tier_counts[tier_name]:4d} utterances")
        print(f"  * Reference Eligible     : {ref_count:4d} utterances")
        print("-" * 56)
        for cat, count in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  * {cat:25s}: {count:4d} utterances")
        print("=" * 56)

    print("\n[✓] Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
