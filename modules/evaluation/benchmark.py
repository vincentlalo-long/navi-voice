"""
End-to-End Benchmark Runner & Comparative Evaluation Suite.
"""

from typing import Any, Dict, List, Optional
import time
import json
from pathlib import Path
import numpy as np
import soundfile as sf

from modules.slicers.factory import get_slicer, SLICER_DISPATCH
from .boundary import evaluate_slice_boundary, aggregate_boundary_metrics
from .segmentation import compute_segmentation_integrity
from .visualize import (
    plot_benchmark_comparison_png,
    plot_boundary_acoustic_detail_png,
    plot_duration_distribution_png,
)


class BenchmarkSuite:
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
            all_chunks_by_file.append(chunks)

            # Evaluate each slice boundary
            for s, e in chunks:
                chunk_audio = data[s:e]
                b_eval = evaluate_slice_boundary(chunk_audio, target_sampling_rate)
                all_slice_metrics.append(b_eval)
                all_chunks_info.append({"start": s, "end": e, "duration": (e - s) / target_sampling_rate})

        elapsed_wall_time = time.perf_counter() - start_wall_time
        rtf = elapsed_wall_time / (total_audio_sec + 1e-9)

        # Compute aggregate metrics
        seg_stats = compute_segmentation_integrity(
            all_chunks_by_file, total_raw_samples, target_sampling_rate
        )
        bnd_stats = aggregate_boundary_metrics(all_slice_metrics)

        return {
            "slicer": slicer_name,
            "algorithm_class": slicer.__class__.__name__,
            "total_audio_sec": round(total_audio_sec, 2),
            "process_time_sec": round(elapsed_wall_time, 3),
            "rtf": round(rtf, 5),
            "segmentation": seg_stats,
            "boundary": bnd_stats,
            "durations": [round(c["duration"], 3) for c in all_chunks_info],
            "start_rms_list": [m["start_rms_db"] for m in all_slice_metrics],
            "end_rms_list": [m["end_rms_db"] for m in all_slice_metrics],
            "step_jumps": [m["step_jump"] for m in all_slice_metrics],
        }

    def run_full_benchmark(
        self,
        slicer_names: List[str],
        audio_files: List[Path],
        sample_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute benchmark across all specified slicers and save structured reports."""
        results = {}
        for s_name in slicer_names:
            print(f"[*] Processing [{s_name.upper()}]...", end="", flush=True)
            res = self.evaluate_slicer(s_name, audio_files, sample_seconds=sample_seconds)
            results[s_name] = res
            print(
                f" done ({res['process_time_sec']:.1f}s, "
                f"{res['segmentation']['total_slices']:,} slices, "
                f"CBR={res['boundary']['cbr_pct']}%, "
                f"RTF={res['rtf']})"
            )
            # Reclaim system RAM and GPU VRAM between slicer runs
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


        # 1. Export Static Comparison Overview (PNG)
        png_overview = self.output_dir / "benchmark_comparison.png"
        try:
            plot_benchmark_comparison_png(results, png_overview)
        except Exception as e:
            print(f"[!] Warning: Failed to render overview PNG chart: {e}")

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

        # Export JSON report
        json_path = self.output_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Export Markdown Table
        md_path = self.output_dir / "benchmark_report.md"
        markdown_report = self.generate_markdown_report(results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        return results

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Format benchmark results into a clean Github Flavored Markdown comparison table."""
        lines = [
            "# Audio Slicer Comparative Benchmark Report",
            "",
            "Comprehensive multi-dimensional evaluation across Boundary Quality, Segmentation Integrity, and RTF performance.",
            "",
            "## 1. Executive Summary & Comparative Matrix",
            "![Benchmark Overview](benchmark_comparison.png)",
            "",
            "## 2. Acoustic Boundary Quality & Discontinuity",
            "![Acoustic Boundary Analysis](boundary_acoustic_detail.png)",
            "",
            "## 3. Utterance Duration Spread & TTS Window Compliance",
            "![Duration Distribution Analysis](duration_distribution.png)",
            "",
            "## 4. Quantitative Benchmark Metrics",
            "",
            "| Slicer Algorithm | Slices | CBR % (Clean+Safe) | Clean % | Warn % | Overlaps | Speech Coverage % | Mean Dur (s) | P95 Dur (s) | RTF (Speed) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for s_name, res in results.items():
            bnd = res["boundary"]
            seg = res["segmentation"]
            line = (
                f"| **{s_name.upper()}** ({res['algorithm_class']}) | "
                f"{seg['total_slices']} | "
                f"**{bnd['cbr_pct']}%** | "
                f"{bnd['clean_pct']}% | "
                f"{bnd['warn_pct']}% | "
                f"{seg['overlap_count']} ({seg['max_overlap_ms']}ms) | "
                f"{seg['speech_coverage_pct']}% | "
                f"{seg['mean_duration_sec']}s | "
                f"{seg['p95_duration_sec']}s | "
                f"**{res['rtf']}** |"
            )
            lines.append(line)

        lines.extend([
            "",
            "### Metric Definitions:",
            "- **CBR % (Clean Boundary Rate)**: Percentage of slices with `CLEAN` or `SAFE` acoustic borders (Edge RMS <= -27 dB, EER <= 0.45, StepJump <= 0.05).",
            "- **Overlaps**: Number of consecutive slice overlaps (Target: 0).",
            "- **Speech Coverage %**: Proportion of raw audio preserved as dialogue.",
            "- **RTF (Real-Time Factor)**: Processing time per second of audio (lower is faster).",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def format_terminal_table(results: Dict[str, Any]) -> str:
        """Render a clean, aligned, compact summary table for terminal display."""
        header = f"{'Algorithm':<12} {'Slices':>8} {'CBR %':>8} {'Clean %':>9} {'Warn %':>8} {'Coverage %':>12} {'Dur (Mean/P95)':>16} {'RTF':>10}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for s_name, res in results.items():
            bnd = res["boundary"]
            seg = res["segmentation"]
            dur_str = f"{seg['mean_duration_sec']:.1f}s / {seg['p95_duration_sec']:.1f}s"
            row = (
                f"{s_name.upper():<12} "
                f"{seg['total_slices']:>8,d} "
                f"{bnd['cbr_pct']:>7.1f}% "
                f"{bnd['clean_pct']:>8.1f}% "
                f"{bnd['warn_pct']:>7.1f}% "
                f"{seg['speech_coverage_pct']:>11.1f}% "
                f"{dur_str:>16} "
                f"{res['rtf']:>10.4f}"
            )
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)

