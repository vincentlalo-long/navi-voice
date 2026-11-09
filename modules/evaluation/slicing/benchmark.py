"""
Phase 1: Audio Segmentation & Slicing Comparative Benchmark Suite.
Evaluates speech slicers (Energy, Silero, WebRTC, Whisper, Hybrid) against
boundary quality (CBR, EER, step jump), segmentation integrity, and RTF processing speed.
"""

from typing import Any, Dict, List, Optional, Tuple
import gc
import time
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
import numpy as np
import soundfile as sf

from modules.slicers.factory import get_slicer
from .boundary import evaluate_slice_boundary, aggregate_boundary_metrics
from .segmentation import compute_segmentation_integrity
from .visualize import (
    plot_benchmark_comparison_png,
    plot_boundary_acoustic_detail_png,
    plot_duration_distribution_png,
)


class SlicingBenchmarkSuite:
    """
    Executes a comprehensive, multi-algorithm benchmark comparing all available
    audio slicers against boundary quality, segmentation integrity, and RTF speed.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.config = config or {}
        self.output_dir = output_dir or Path("reports/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _package_version(name: str) -> str:
        """Return an installed distribution version without failing the report."""
        try:
            return importlib_metadata.version(name)
        except Exception:
            return "not-installed"

    def evaluate_slicer(
        self,
        slicer_name: str,
        audio_files: List[Path],
        sample_seconds: Optional[float] = None,
        target_sampling_rate: int = 44100,
    ) -> Dict[str, Any]:
        """Run benchmark on a single slicer strategy across test audio files."""
        slicer = get_slicer(slicer_name, self.config)

        total_audio_sec = 0.0
        start_wall_time = time.perf_counter()

        all_slice_metrics: List[Dict[str, Any]] = []
        all_chunks_info: List[Dict[str, Any]] = []

        total_raw_samples = 0
        slicing_time_sec = 0.0
        all_chunks_by_file: List[List[Tuple[int, int]]] = []

        for audio_path in audio_files:
            data, file_sr = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Polyphase resample if needed
            if file_sr != target_sampling_rate:
                import scipy.signal
                gcd = np.gcd(file_sr, target_sampling_rate)
                up = target_sampling_rate // gcd
                down = file_sr // gcd
                data = scipy.signal.resample_poly(data, up, down).astype(np.float32)

            if sample_seconds:
                max_samples = int(target_sampling_rate * sample_seconds)
                data = data[:max_samples]

            total_raw_samples += len(data)
            total_audio_sec += len(data) / target_sampling_rate

            # Execute slicing
            t0 = time.perf_counter()
            chunks = slicer.slice(data, target_sampling_rate)
            t1 = time.perf_counter()
            slicing_time_sec += t1 - t0
            all_chunks_by_file.append(chunks)

            # Evaluate each slice boundary directly on raw cut
            for s, e in chunks:
                chunk_audio = data[s:e]
                b_eval = evaluate_slice_boundary(chunk_audio, target_sampling_rate)
                all_slice_metrics.append(b_eval)
                all_chunks_info.append({"start": s, "end": e, "duration": (e - s) / target_sampling_rate})

        elapsed_wall_time = time.perf_counter() - start_wall_time
        # RTF must describe the algorithm, not the harness: wall time also covers
        # file I/O, resampling and per-slice boundary scoring.
        rtf = slicing_time_sec / (total_audio_sec + 1e-9)

        # Compute aggregate metrics
        seg_stats = compute_segmentation_integrity(
            all_chunks_by_file, total_raw_samples, target_sampling_rate
        )
        bnd_stats = aggregate_boundary_metrics(all_slice_metrics)

        result = {
            "slicer": slicer_name,
            "algorithm_class": slicer.__class__.__name__,
            "total_audio_sec": round(total_audio_sec, 2),
            "process_time_sec": round(elapsed_wall_time, 3),
            "slicing_time_sec": round(slicing_time_sec, 3),
            "rtf": round(rtf, 5),
            "segmentation": seg_stats,
            "boundary": bnd_stats,
            "durations": [round(c["duration"], 3) for c in all_chunks_info],
            "start_rms_list": [m.get("start_rms_db", -99.0) for m in all_slice_metrics],
            "end_rms_list": [m.get("end_rms_db", -99.0) for m in all_slice_metrics],
            "step_jumps": [m.get("step_jump", 0.0) for m in all_slice_metrics],
        }
        slicer.release()
        return result

    def run_full_benchmark(
        self,
        slicer_names: List[str],
        audio_files: List[Path],
        sample_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes comparative benchmark across all requested slicers,
        generates multi-panel PNG visualization charts, and outputs summary JSON and Markdown reports.
        """
        results: Dict[str, Any] = {}
        failures: List[Tuple[str, str]] = []

        for name in slicer_names:
            print(f"[*] Benchmarking slicer: '{name}' ...", flush=True)
            try:
                res = self.evaluate_slicer(name, audio_files, sample_seconds=sample_seconds)
                results[name] = res
                seg = res["segmentation"]
                bnd = res["boundary"]
                print(
                    f"    done: {seg['total_slices']} slices | "
                    f"CBR={bnd['cbr_pct']}% | RTF={res['rtf']}",
                    flush=True,
                )
            except Exception as e:
                failures.append((name, str(e)))
                print(f"[!] Warning: Slicer '{name}' failed with error: {e}")
            finally:
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if not results:
            details = "; ".join(f"{name}: {err}" for name, err in failures)
            raise RuntimeError(f"No slicer completed successfully. {details}")

        if failures:
            print(
                "[!] Benchmark completed with failed slicers: "
                + ", ".join(name for name, _ in failures)
            )

        # Provenance is part of the deliverable: a result without input identity,
        # device and dependency versions cannot be reproduced or compared later.
        total_audio_sec = next(iter(results.values())).get("total_audio_sec", 0.0) if results else 0.0
        self._run_meta = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluation_phase": "Phase 2: Audio Slicing & Boundary Segmentation",
            "input_files": [str(p) for p in audio_files],
            "input_file_count": len(audio_files),
            "sample_seconds": sample_seconds,
            "total_audio_sec": round(total_audio_sec, 2),
            "device": self.config.get("device", "unknown"),
            "whisper_model_size": self.config.get(
                "whisper_model_size", self.config.get("model_size", "base")
            ),
            "compute_type": self.config.get("compute_type", "unknown"),
            "language": self.config.get("language", "unknown"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "versions": {
                "numpy": self._package_version("numpy"),
                "scipy": self._package_version("scipy"),
                "soundfile": self._package_version("soundfile"),
                "faster-whisper": self._package_version("faster-whisper"),
                "torch": self._package_version("torch"),
                "silero-vad": self._package_version("silero-vad"),
                "webrtcvad-wheels": self._package_version("webrtcvad-wheels"),
            },
            "config": {
                key: value
                for key, value in self.config.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "failed_slicers": [{"slicer": name, "error": error} for name, error in failures],
        }

        # 1. Export High-Level Executive Summary Comparison (PNG)
        png_comparison = self.output_dir / "benchmark_comparison.png"
        try:
            plot_benchmark_comparison_png(results, png_comparison)
        except Exception as e:
            print(f"[!] Warning: Failed to render comparison PNG chart: {e}")

        # 2. Export Detailed Acoustic Boundary Inspection (PNG)
        png_boundary = self.output_dir / "boundary_acoustic_detail.png"
        try:
            plot_boundary_acoustic_detail_png(results, png_boundary)
        except Exception as e:
            print(f"[!] Warning: Failed to render boundary detail PNG chart: {e}")

        # 3. Export Detailed Duration & Segmentation Distribution (PNG)
        png_duration = self.output_dir / "duration_distribution.png"
        try:
            plot_duration_distribution_png(results, png_duration)
        except Exception as e:
            print(f"[!] Warning: Failed to render duration distribution PNG chart: {e}")

        # Export JSON report with comprehensive metadata
        json_path = self.output_dir / "benchmark_summary.json"
        export_payload = {
            "_metadata": {
                **self._run_meta,
                "note": "Phase 2 slicing uses the configured slicing model; Phase 3 ASR transcription uses large-v3 for maximum text accuracy.",
            }
        }
        export_payload.update(results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, ensure_ascii=False)

        # Export Markdown Table
        md_path = self.output_dir / "benchmark_report.md"
        markdown_report = self.generate_markdown_report(results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        return results

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for run_full_benchmark."""
        return self.run_full_benchmark(*args, **kwargs)

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Format benchmark results into a clean Github Flavored Markdown multi-dimensional comparison report."""
        lines = [
            "# Audio Slicer Comparative Benchmark Report",
            "",
            "> [!IMPORTANT]",            "> **PHASE 2.1 EVALUATION: AUDIO SLICING & BOUNDARY SEGMENTATION** ",
            "> This benchmark evaluates exclusively **Phase 2.1 (Audio Slicing & Boundary Segmentation)** of the speech pipeline on the input audio listed in the provenance table below. Downstream stages (Phase 3: Semantic Tagging & ASR, Phase 4: Acoustic Modeling & Downstream Synthesis Quality) are independent and evaluated in subsequent pipeline phases.",
            "",
            "---",
            "",
            "## 1. Executive Summary & Comparative Matrix",
            "![Benchmark Overview](benchmark_comparison.png)",
            "",
            "## 2. Acoustic Boundary Quality & Discontinuity Inspection",
            "![Acoustic Boundary Analysis](boundary_acoustic_detail.png)",
            "",
            "## 3. Utterance Duration Spread & TTS Window Compliance",
            "![Duration Distribution Analysis](duration_distribution.png)",
            "",
            "---",
            "",
            "## 4. Multi-Dimensional Benchmark Evaluation",
            "",
            "### Table 1: Acoustic Boundary Quality (Physical Safety & Click Prevention)",
            "Audited across the first and last 25ms of each extracted slice:",
            "- **`CLEAN`**: Edge RMS $\\le -35\\text{ dBFS}$, $\\text{EER} \\le 0.20$, and step jump $|x[0]| \\le 0.02$.",
            "- **`SAFE`**: Edge RMS $\\le -27\\text{ dBFS}$, $\\text{EER} \\le 0.45$, and step jump $|x[0]| \\le 0.05$.",
            "- **`WARN`**: Speech clipping or transient step discontinuity risk.",
            "",
            "Measured on the raw cut, before the raised-cosine fade and loudness normalization that",
            "`voice_dataset_miner.py` applies when writing files. This is deliberate: it grades the",
            "slicer's choice of boundary, and a faded file would report a step jump of ~0 by construction.",
            "",
            "| Slicer Algorithm | Total Slices | CBR % (Clean+Safe) | Clean % | Warn % | P95 Edge RMS | Max Step Jump | Consecutive Overlaps |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for s_name, res in results.items():
            if s_name.startswith("_"):
                continue
            bnd = res["boundary"]
            seg = res["segmentation"]
            p95_rms = f"{bnd.get('p95_edge_rms_db', 0.0):.1f} dBFS"
            max_jump = f"{bnd.get('max_step_jump', 0.0):.4f}"
            overlaps = f"{seg['overlap_count']} ({seg['max_overlap_ms']:.1f}ms)"
            line = (
                f"| **`{s_name}`** ({res.get('algorithm_class', '')}) | "
                f"{seg['total_slices']:,} | "
                f"**{bnd['cbr_pct']:.1f}%** | "
                f"{bnd['clean_pct']:.1f}% | "
                f"{bnd['warn_pct']:.1f}% | "
                f"{p95_rms} | "
                f"{max_jump} | "
                f"{overlaps} |"
            )
            lines.append(line)

        lines.extend([
            "",
            "---",
            "",
            "### Table 2: Segmentation Distribution & TTS Golden Range [2s – 10s] Compliance",
            "Evaluates conversational naturalness and attention window safety for autoregressive TTS models.",
            "`Audio Coverage %` is total slice duration / total audio duration; every slice is padded with",
            "silence, so this counts silence as retained and is not a speech-only coverage figure.",
            "",
            "| Slicer Algorithm | Audio Coverage % (incl. silence) | Mean Duration | Median Duration | P95 Duration | Golden Range [2s-10s] % | Short Phrases (< 2s) % | Run-on (> 10s) % | Over-Max (> 12s) % |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for s_name, res in results.items():
            if s_name.startswith("_"):
                continue
            seg = res["segmentation"]
            durs = res.get("durations", [])
            total_d = len(durs) if durs else 1
            golden_pct = (sum(1 for d in durs if 2.0 <= d <= 10.0) / total_d) * 100.0 if durs else 0.0
            short_pct = (sum(1 for d in durs if d < 2.0) / total_d) * 100.0 if durs else 0.0
            runon_pct = (sum(1 for d in durs if d > 10.0) / total_d) * 100.0 if durs else 0.0
            overmax_pct = (sum(1 for d in durs if d > 12.0) / total_d) * 100.0 if durs else 0.0

            line = (
                f"| **`{s_name}`** | "
                f"{seg['audio_coverage_pct']:.1f}% | "
                f"{seg['mean_duration_sec']:.2f}s | "
                f"{seg['median_duration_sec']:.2f}s | "
                f"{seg['p95_duration_sec']:.2f}s | "
                f"**{golden_pct:.1f}%** | "
                f"{short_pct:.1f}% | "
                f"{runon_pct:.1f}% | "
                f"{overmax_pct:.1f}% |"
            )
            lines.append(line)

        lines.extend([
            "",
            "---",
            "",
            "### Table 3: Computational Performance & Real-Time Factor (RTF)",
            "Evaluated on Intel CPU + NVIDIA GPU (Whisper/Hybrid evaluated with model_size='base').",
            "RTF = time spent inside `slicer.slice()` / audio duration. File I/O, resampling and",
            "boundary scoring are excluded, so this figure describes the algorithm alone.",
            "",
            "| Slicer Algorithm | Slicing Time | Real-Time Factor (RTF) | Processing Speedup |",
            "| :--- | :---: | :---: | :---: |",
        ])

        for s_name, res in results.items():
            if s_name.startswith("_"):
                continue
            rtf = res["rtf"]
            speedup = f"{1.0 / max(rtf, 1e-9):,.0f}× faster than realtime"
            p_time = f"{res.get('slicing_time_sec', res['process_time_sec']):.1f}s"
            line = f"| **`{s_name}`** | {p_time} | **{rtf:.5f}** | {speedup} |"
            lines.append(line)

        # ------------------------------------------------------------------
        # Dynamic conclusion: every number below is derived from this run.
        # ------------------------------------------------------------------
        valid = [(name, res) for name, res in results.items() if not str(name).startswith("_")]
        if not valid:
            lines.extend([
                "",
                "---",
                "",
                "## 5. Engineering Conclusion & Strategy Recommendation",
                "",
                "No slicer completed successfully, so no recommendation can be made from this run.",
                "",
            ])
            return "\n".join(lines)

        def _share(durations: List[float], predicate) -> float:
            if not durations:
                return 0.0
            return (sum(1 for duration in durations if predicate(duration)) / len(durations)) * 100.0

        rows = []
        for name, res in valid:
            durations = res.get("durations", [])
            seg = res["segmentation"]
            bnd = res["boundary"]
            rows.append({
                "name": name,
                "res": res,
                "golden": _share(durations, lambda d: 2.0 <= d <= 10.0),
                "short": _share(durations, lambda d: d < 2.0),
                "runon": _share(durations, lambda d: d > 10.0),
                "overmax": _share(durations, lambda d: d > 12.0),
                "cbr": bnd["cbr_pct"],
                "warn": bnd["warn_pct"],
                "coverage": seg["audio_coverage_pct"],
                "rtf": res["rtf"],
                "slices": seg["total_slices"],
                "mean": seg["mean_duration_sec"],
                "p95": seg["p95_duration_sec"],
            })

        # Prefer a high golden-range share, but never recommend a slicer whose
        # boundary safety is unacceptable or which still exceeds 12 s.
        eligible = [row for row in rows if row["cbr"] >= 80.0 and row["overmax"] <= 1.0]
        primary = max(eligible or rows, key=lambda row: (row["golden"], row["cbr"], -row["rtf"]))
        fastest = min(rows, key=lambda row: row["rtf"])
        best_cbr = max(rows, key=lambda row: row["cbr"])
        best_coverage = max(rows, key=lambda row: row["coverage"])
        disqualified = [row for row in rows if row["warn"] >= 30.0 or row["runon"] >= 20.0]

        def _metric_line(row) -> str:
            return (
                f"{row['golden']:.1f}% in [2s–10s], CBR {row['cbr']:.1f}%, "
                f"WARN {row['warn']:.1f}%, coverage {row['coverage']:.1f}% "
                f"(includes silence), mean {row['mean']:.2f}s, P95 {row['p95']:.2f}s, "
                f"RTF {row['rtf']:.5f} ({1.0 / max(row['rtf'], 1e-9):,.0f}× realtime)"
            )

        meta = getattr(self, "_run_meta", {})
        versions = meta.get("versions", {})
        lines.extend([
            "",
            "---",
            "",
            "## 4b. Benchmark Provenance & Reproducibility",
            "| Field | Value |",
            "| :--- | :--- |",
            f"| Generated (UTC) | {meta.get('generated_at_utc', 'unknown')} |",
            f"| Input files | {', '.join(Path(p).name for p in meta.get('input_files', [])) or 'unknown'} |",
            f"| Input duration | {float(meta.get('total_audio_sec', 0.0) or 0.0):.2f} s |",
            f"| Sample per file | {meta.get('sample_seconds') or 'FULL (100%)'} |",
            f"| Device | {meta.get('device', 'unknown')} |",
            f"| Slicing Whisper | {meta.get('whisper_model_size', 'unknown')} ({meta.get('compute_type', 'unknown')}) |",
            f"| Language | {meta.get('language', 'unknown')} |",
            f"| Failed slicers | {len(meta.get('failed_slicers', []))} |",
            "| Versions | "
            + ", ".join(f"{name} {version}" for name, version in versions.items())
            + " |",
            "",
            "---",
            "",
            "## 5. Engineering Conclusion & Strategy Recommendation",
            "",
            "> All figures in this section are computed from the current `results`; "
            "no performance constant is stored in the report generator.",
            "",
            f"### Primary Recommendation: `--slicer {primary['name']}`",
            f"- **Duration suitability:** {primary['golden']:.1f}% of {primary['slices']:,} slices "
            f"fall in the 2–10 s training window (mean {primary['mean']:.2f} s, P95 {primary['p95']:.2f} s).",
            f"- **Boundary safety:** CBR {primary['cbr']:.1f}% (CLEAN + SAFE), "
            f"WARN {primary['warn']:.1f}%, over-12 s share {primary['overmax']:.1f}%.",
            f"- **Audio coverage:** {primary['coverage']:.1f}% of the source duration is inside slices. "
            "This metric counts silence as retained; it is not a speech-only coverage figure.",
            f"- **Cost:** RTF {primary['rtf']:.5f}, i.e. {1.0 / max(primary['rtf'], 1e-9):,.0f}× faster than real time.",
            "",
            f"### Secondary Recommendation: `--slicer {fastest['name']}`",
            f"- Fastest measured strategy: RTF {fastest['rtf']:.5f} "
            f"({1.0 / max(fastest['rtf'], 1e-9):,.0f}× realtime).",
            f"- {_metric_line(fastest)}.",
            "",
            "### Per-Algorithm Observations",
        ])

        for row in rows:
            lines.append(f"- **`{row['name']}`**: {_metric_line(row)}.")

        lines.extend([
            "",
            "### Boundary and Coverage Highlights",
            f"- **Highest CBR:** `{best_cbr['name']}` at {best_cbr['cbr']:.1f}% "
            f"(WARN {best_cbr['warn']:.1f}%).",
            f"- **Highest audio coverage:** `{best_coverage['name']}` at "
            f"{best_coverage['coverage']:.1f}% (includes silence).",
        ])

        valid_names = {name for name, _ in valid}
        if "whisper" in valid_names and "hybrid" in valid_names:
            whisper_row = next(row for row in rows if row["name"] == "whisper")
            hybrid_row = next(row for row in rows if row["name"] == "hybrid")
            lines.append(
                f"- **Hybrid vs Whisper micro-refinement:** CBR "
                f"{hybrid_row['cbr'] - whisper_row['cbr']:+.1f} pp, WARN reduction "
                f"{whisper_row['warn'] - hybrid_row['warn']:.1f} pp."
            )

        if "energy" in valid_names:
            energy_row = next(row for row in rows if row["name"] == "energy")
            lines.append(
                f"- **Energy merge behaviour:** {energy_row['golden']:.1f}% in the golden range, "
                f"{energy_row['short']:.1f}% below 2 s, {energy_row['runon']:.1f}% above 10 s."
            )

        if disqualified:
            lines.extend(["", "### Disqualified for High-Quality TTS Training"])
            for row in disqualified:
                reasons = []
                if row["warn"] >= 30.0:
                    reasons.append(f"WARN {row['warn']:.1f}%")
                if row["runon"] >= 20.0:
                    reasons.append(f"run-on {row['runon']:.1f}% > 10 s")
                lines.append(f"- **`{row['name']}`**: " + ", ".join(reasons) + ".")
        else:
            lines.extend([
                "",
                "### No Disqualified Algorithms",
                "No strategy crossed the disqualification thresholds (WARN ≥ 30% or run-on ≥ 20%).",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 6. Boundary Quality Verification",
            "To mathematically inspect extracted audio slice boundaries:",
            "```bash",
            ".venv/bin/python scripts/verify_boundaries.py --slices-dir datasets/processed/energy/slices",
            "```",
            "Each slice is audited across its first and last 25ms:",
            "- **`[CLEAN]`**: Edge RMS <= -35 dBFS (quiet silence onset/offset).",
            "- **`[SAFE]`**: Edge RMS <= -27 dBFS with Edge Energy Ratio (EER) <= 0.45 and transient jump |x[0]| <= 0.05.",
            "- **`[WARN]`**: Edge energy exceeding thresholds, indicating potential phonetic clipping.",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def format_terminal_table(results: Dict[str, Any]) -> str:
        """Render a clean, aligned, compact summary table for terminal display."""
        header = f"{'Algorithm':<12} {'Slices':>8} {'CBR %':>8} {'Clean %':>9} {'Warn %':>8} {'Coverage %':>11} {'Dur (Mean/P95)':>16} {'RTF':>10}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for s_name, res in results.items():
            if s_name.startswith("_"):
                continue
            bnd = res["boundary"]
            seg = res["segmentation"]
            dur_str = f"{seg['mean_duration_sec']:.1f}s / {seg['p95_duration_sec']:.1f}s"
            row = (
                f"{s_name.upper():<12} "
                f"{seg['total_slices']:>8,d} "
                f"{bnd['cbr_pct']:>7.1f}% "
                f"{bnd['clean_pct']:>8.1f}% "
                f"{bnd['warn_pct']:>7.1f}% "
                f"{seg['audio_coverage_pct']:>11.1f}% "
                f"{dur_str:>16} "
                f"{res['rtf']:>10.4f}"
            )
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)


# Backward compatibility alias
BenchmarkSuite = SlicingBenchmarkSuite
