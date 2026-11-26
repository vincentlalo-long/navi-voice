#!/usr/bin/env python3
"""
Phase 3 Enrichment Audit & Curation CLI.

Generates QC waterfall, prosody acoustic space, and emotion category reports
from an enriched processed directory, and optionally applies human curation
entries back into metadata.json and re-exports dataset artifacts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
VOICE_ROOT = SCRIPT_DIR.parent
if str(VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_ROOT))

from modules.enrichment import (
    apply_human_curation_file,
    export_enriched_dataset,
    mirror_custom_export_paths,
)
from modules.evaluation import generate_enrichment_report
from scripts.voice_dataset_miner import resolve_config


def _resolve_cli_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (VOICE_ROOT / candidate).resolve()


def _infer_speaker_and_lang(
    processed_dir: Path,
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    profile_arg: Optional[str],
    dataset_list_name: str,
) -> Tuple[str, str]:
    """Infer (speaker_name, speaker_lang) from profile, dataset.list, records, or config."""
    speaker_cfg = config.get("speaker", {}) if isinstance(config.get("speaker"), dict) else {}
    cfg_name = str(speaker_cfg.get("name") or "Speaker")
    cfg_lang = str(speaker_cfg.get("language") or "JA")

    if profile_arg:
        return cfg_name, cfg_lang

    list_path = processed_dir / dataset_list_name
    if list_path.is_file():
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4 and parts[1].strip() and parts[2].strip():
                return parts[1].strip(), parts[2].strip()

    for rec in records:
        if isinstance(rec, dict) and rec.get("speaker"):
            return str(rec["speaker"]), cfg_lang

    return cfg_name, cfg_lang


def main() -> None:
    default_config_path = VOICE_ROOT / "configs" / "default_config.json"

    parser = argparse.ArgumentParser(
        description="Phase 3 Enrichment Audit Report & Human Curation CLI"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="datasets/processed/energy",
        help="Path to processed slicer directory containing metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/enrichment",
        help="Directory where enrichment audit reports and PNG figures will be saved",
    )
    parser.add_argument(
        "--apply-curation",
        type=str,
        default=None,
        help="Optional path to edited reference_candidates.json or curation list JSON to apply before reporting",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile name or path to overlay configuration JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to base configuration JSON",
    )
    args = parser.parse_args()

    config = resolve_config(args.config, args.profile, VOICE_ROOT)
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    metadata_json_name = str(paths_cfg.get("metadata_json") or "metadata.json")
    dataset_list_name = str(paths_cfg.get("dataset_list") or "dataset.list")

    processed_dir = _resolve_cli_path(args.processed_dir)
    output_dir = _resolve_cli_path(args.output_dir)
    meta_path = processed_dir / metadata_json_name

    if not meta_path.is_file():
        print(f"[!] Error: Metadata file not found: {meta_path}")
        sys.exit(1)

    try:
        records = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[!] Error: Failed to read metadata file {meta_path}: {exc}")
        sys.exit(1)

    if not isinstance(records, list):
        print(f"[!] Error: Metadata payload in {meta_path} must be a JSON list.")
        sys.exit(1)

    speaker_name, speaker_lang = _infer_speaker_and_lang(
        processed_dir,
        records,
        config,
        args.profile,
        dataset_list_name,
    )

    curation_updated: Optional[int] = None
    curation_file_str: Optional[str] = None
    if args.apply_curation:
        curation_path = _resolve_cli_path(args.apply_curation)
        if not curation_path.is_file():
            print(f"[!] Error: Curation file not found: {curation_path}")
            sys.exit(1)
        curation_file_str = str(curation_path)
        curation_updated = apply_human_curation_file(records, curation_path, cfg=config)
        print(f"[+] Applied human curation to {curation_updated} record(s) from {curation_path}")
        exported_paths = export_enriched_dataset(
            records,
            processed_dir,
            speaker_name,
            speaker_lang,
            cfg=config,
        )
        mirror_custom_export_paths(exported_paths, processed_dir, paths_cfg)
        print(f"[+] Re-exported dataset artifacts in {processed_dir} ({speaker_name} [{speaker_lang}])")

    run_meta = {
        "processed_dir": str(processed_dir),
        "speaker": speaker_name,
        "language": speaker_lang,
        "profile": args.profile,
        "curation_file": curation_file_str,
        "curation_updated_count": curation_updated,
    }
    report_paths = generate_enrichment_report(records, output_dir, run_meta=run_meta)

    summary = json.loads(report_paths["enrichment_summary_json"].read_text(encoding="utf-8"))
    wf = summary["waterfall"]
    print("\n" + "=" * 60)
    print("PHASE 3 ENRICHMENT AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Processed Dir:      {processed_dir}")
    print(f"  Total Slices:       {wf['total_slices']}")
    print(f"  GOLD / SILVER / REJ:{wf['gold_count']} / {wf['silver_count']} / {wf['reject_count']}")
    print(f"  Reference Eligible: {wf['reference_eligible_count']}")
    print(f"  Human Reviewed:     {wf['human_reviewed_count']}")
    print(f"  Report Markdown:    {report_paths['enrichment_report_md']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
