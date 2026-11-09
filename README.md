# Voice Dataset Miner & Segmentation Benchmarking

An experimental engineering toolkit for speech dataset preparation, multi-algorithm audio segmentation, loudness normalization, Whisper ASR transcription, and rigorous acoustic boundary evaluation.

---

## 1. Overview & Core Objectives

Preparing high-quality speech datasets for Text-to-Speech (TTS) models (e.g., GPT-SoVITS, CosyVoice, VITS) from long-form raw dialogue audio presents critical challenges:
- **Phoneme Clipping**: Truncating low-energy unvoiced consonants (such as `/s/`, `/sh/`, `/h/`, `/k/`) at utterance onset or offset.
- **Gibbs Transient Clicking**: Abrupt non-zero cuts causing step jumps $|x[0]| > 0.05$, generating high-frequency clicks $O(1/f)$.
- **Mid-sentence Over-segmentation**: Fragmenting natural sentences into incomplete sub-2s phrases during natural micro-pauses.
- **Run-on Utterances**: Long monologues exceeding 10–12 seconds, resulting in training attention collapse or OOM errors.

This project implements a modular Strategy Pattern supporting **5 distinct slicing algorithms**, pairs them with strict zero-crossing snapping, EBU R128 (-20 LUFS) or Peak normalization, and raised-cosine edge fading, and provides an end-to-end automated benchmark suite generating publication-ready comparison charts.

---

## 2. Documentation

Two companion documents describe the design and the second pipeline phase:

| Document | Scope |
| :--- | :--- |
| [Introduction](docs/introduction.pdf) | Problem statement, four-stage corpus pipeline, design principles, and related work. |
| [Phase 2 Slicing Whitepaper](docs/slice.pdf) | Constrained interval formulation, five segmentation strategies, signal conditioning, evaluation methodology, and the full 3.42-hour benchmark. |
| [Phase 2 Benchmark Report](reports/benchmarks/benchmark_report.md) | Full empirical multi-algorithm comparison tables, charts, CBR boundary metrics, and duration distributions. |

Phases 3 and 4 will each receive their own document; the introduction defines the shared data contract they consume.

---

## 3. Architecture & Modular Layout

```text
voice/
├── configs/
│   ├── default_config.json        # Base pipeline configuration (thresholds, ASR settings)
│   └── profiles/
│       ├── profile.example.json   # Template for speaker-specific categorization
│       └── *.json                 # Local custom profiles (e.g. my_speaker.json)
├── datasets/
│   ├── raw/                       # Source raw audio files (.wav, .mp3, .flac)
│   └── processed/                 # Separated output folders per slicer (energy/, silero/, ...)
├── docs/
│   ├── introduction.pdf           # Problem statement, pipeline, and related work
│   └── slice.pdf                  # Phase 2: slicing, conditioning, and benchmark
├── modules/
│   ├── audio/                     # [Phase 2 DSP] Normalization (Peak, EBU R128) & Fading
│   │   ├── utils.py               # Audio statistics, edge metrics, raised-cosine fade
│   │   └── __init__.py
│   ├── slicers/                   # [Phase 2 Slicing] Strategy Pattern for VAD / Segmentation
│   │   ├── base.py                # Abstract BaseSlicer interface
│   │   ├── energy_slicer.py       # DSP Energy-based RMS slicer (zero-copy strided)
│   │   ├── silero_slicer.py       # Neural Silero VAD slicer (LSTM recurrent network)
│   │   ├── webrtc_slicer.py       # WebRTC VAD GMM baseline slicer (CPU realtime)
│   │   ├── whisper_slicer.py      # Semantic word-timestamp & punctuation slicer
│   │   ├── hybrid_slicer.py       # Two-Stage Hybrid (Whisper macro + DSP micro-refine)
│   │   └── factory.py             # Slicer factory (get_slicer)
│   ├── tagging/                   # [Phase 3 Enrichment] Semantic Emotion & Prosody Classifier
│   │   ├── classifier.py          # Rule-based Emotion & prosody tagger
│   │   └── __init__.py
│   ├── evaluation/                # Evaluation Suite (Structured by Pipeline Phases)
│   │   ├── slicing/               # Phase 2: Speech Slicing & Boundary Evaluation
│   │   │   ├── boundary.py        # R-value (Räsänen 2009), non-greedy tolerance matching, CBR, EER
│   │   │   ├── segmentation.py    # Speech coverage %, durations (mean/p95), overlap detector
│   │   │   ├── benchmark.py       # SlicingBenchmarkSuite runner
│   │   │   └── visualize.py       # Multi-panel PNG comparison & distribution charts
│   │   ├── tts/                   # Phase 4: Voice Cloning & Synthesis Quality (MOS, SECS, CER)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── cuda_utils.py              # Dynamic CUDA/cuBLAS/cuDNN preloader
├── reports/
│   └── benchmarks/                # Benchmark reports (.md, .json) & PNG charts
├── scripts/
│   ├── voice_dataset_miner.py     # Main processing and transcription pipeline
│   └── run_benchmark.py           # Multi-slicer benchmark runner & chart generator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Slicing Algorithms

| Method | Core Mechanism | Strengths | Trade-offs |
| :--- | :--- | :--- | :--- |
| **`energy`** | Zero-copy SIMD strided RMS energy thresholding (`-38 dB`) + midpoint silence split + raised-cosine fade | Deterministic, ultra-fast, zero GPU/model dependency | Relies on silence gaps; sensitive to high-energy background music |
| **`silero`** | Pre-trained recurrent neural network (LSTM) on 64-band Log-Mel filterbanks | Detects high-frequency spectral footprints of unvoiced consonants even at low dB | Requires PyTorch inference runtime |
| **`webrtc`** | Gaussian Mixture Model (GMM) evaluating sub-band speech probabilities (16kHz) | Industry standard, lightweight CPU real-time processing | Lower resolution on noisy/BGM audio compared to neural VAD |
| **`whisper`** | Word-level timestamp mapping from Faster-Whisper cross-attention + Japanese punctuation restore | Groups into natural semantic clauses by construction; avoids mid-sentence pause splitting | Heaviest computation; requires GPU for acceptable latency |
| **`hybrid`** | Two-Stage: Whisper semantic macro-grouping + DSP short-time energy dip and zero-crossing snapping | Combines semantic clause integrity with sub-millisecond acoustic zero-crossing precision | Requires Faster-Whisper ASR pass |

---

## 5. Quick Start

### Environment Setup
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

*(Note: NVIDIA CUDA runtime libraries like `cublas` and `cudnn` are automatically preloaded dynamically into global process memory on Linux, eliminating the need for manual `LD_LIBRARY_PATH` configuration).*

---

### Running Dataset Creation (`voice_dataset_miner.py`)

Place your raw source audio files in `datasets/raw/`.

**1. Full Pipeline (Slicing + Peak Normalization + Whisper ASR + Category Tagging):**
```bash
.venv/bin/python scripts/voice_dataset_miner.py --slicer energy
```

**2. Slicing Only (Skip Whisper ASR):**
When you only want to extract audio slices and metadata without spending GPU time transcribing:
```bash
.venv/bin/python scripts/voice_dataset_miner.py --slicer energy --skip-transcribe
```

**3. Loudness Normalization Mode (EBU R128 vs Peak):**
To normalize to broadcast-standard -20 LUFS integrated loudness (via ITU-R BS.1770-4):
```bash
.venv/bin/python scripts/voice_dataset_miner.py --slicer energy --norm-method ebu_r128
```

**4. Two-Stage Hybrid Slicing (Semantic Macro + DSP Micro-Refine):**
```bash
.venv/bin/python scripts/voice_dataset_miner.py --slicer hybrid
```

**5. Dry-run Test (Process first 60 seconds):**
```bash
.venv/bin/python scripts/voice_dataset_miner.py --sample-seconds 60 --slicer silero
```

Generated slices and metadata (`metadata.json`, `metadata.csv`, `dataset.list`) are stored in isolated directories under `datasets/processed/<slicer>/`. When `--profile` is used, the run is isolated further under `datasets/processed/<slicer>/<profile>/`, so two characters cannot overwrite or clean each other's slices.

**6. Resume / Retag Existing Slices:**
```bash
.venv/bin/python scripts/voice_dataset_miner.py --slicer hybrid --skip-slicing --skip-transcribe
```
This reads the existing manifest instead of rebuilding it from WAV names, preserving `source_file`, source offsets, transcripts, categories and pre-normalization gain telemetry. Raw audio is not required.

---

## 6. Pipeline Phases & Current Progress

The end-to-end speech corpus construction pipeline consists of 4 decoupled, modular stages:

| Phase | Description | Status | Primary Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1: Ingest & Collection** | Raw audio ingest, speaker/language metadata, and format verification. | **Complete** | Continuous dialogue corpus ingest (12,301s / 3.42h). |
| **Phase 2: Audio Slicing & Boundary Segmentation** | Multi-algorithm segmentation (`energy`, `silero`, `webrtc`, `whisper`, `hybrid`), zero-crossing snapping, raised-cosine fading, and loudness normalization. | **Complete** (hardened 2026-11-09) | 5 slicer algorithms, automated benchmark suite, [slice.pdf](docs/slice.pdf), and [benchmark_report.md](reports/benchmarks/benchmark_report.md). |
| **Phase 3: Semantic Tagging & ASR Transcription** | Faster-Whisper ASR transcription, emotion/prosody classification, and dataset manifest export. | *In Progress* | Emotion/prosody classifier, Whisper transcription pipeline, dataset packaging. |
| **Phase 4: Acoustic Modeling & Downstream Synthesis Evaluation** | TTS model training (GPT-SoVITS, CosyVoice, VITS) and speech quality evaluation (MOS, SECS, CER/WER, MCD). | *Planned* | Downstream evaluation module (`modules/evaluation/tts`). |

---

## 7. Phase 2 Benchmarking & Recommendations Summary

The 5 slicing algorithms were benchmarked across 3.42 hours (12,301s) of continuous dialogue audio:

| Slicer Algorithm | CBR % (Clean+Safe) | Audio Coverage % (incl. silence) | Golden Range [2s–10s] % | Slicing RTF | Processing Speedup | Primary Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`energy`** | **99.3%** | **96.6%** | **95.6%** | **0.00138** | **725× faster** | 🏆 **Primary for TTS Training** (Clean silent cuts, best duration suitability) |
| **`hybrid`** | 88.1% | 92.1% | 92.9% | 0.02592 | 39× faster | Semantic option (Clause alignment, but lower coverage/CBR than Energy) |
| **`silero`** | 97.5% | 96.9% | 89.8% | 0.01966 | 51× faster | Strong neural VAD (High CBR, 9.3% run-on > 10s) |
| **`whisper`** | 78.4% | 94.2% | 94.6% | 0.02970 | 34× faster | Semantic baseline (Good durations, edges require DSP refinement) |
| **`webrtc`** | 88.8% | 98.3% | 88.0% | 0.00107 | 935× faster | ⚡ **Fastest CPU baseline** (11.2% WARN, use with downstream QC) |

> [!NOTE]
> These numbers are from the hardened Phase 2.1 run (2026-11-09). The earlier Silero/WebRTC run-on rates were inflated because their VAD intervals were merged *after* padding (~1.2s effective pause); the fix merges raw intervals first, then pads. Hybrid now drops long inter-sentence silences instead of retaining them as dead clips, so coverage is lower and no longer overstates speech preservation.

> [!TIP]
> **Detailed Benchmark Report & Visual Charts:**  
> For the complete multi-dimensional empirical findings, acoustic detail charts, duration distribution histograms, and mathematical derivations:
> - 📊 **Detailed Benchmark Report:** [reports/benchmarks/benchmark_report.md](reports/benchmarks/benchmark_report.md)
> - 📈 **Visual Comparison Charts:** [benchmark_comparison.png](reports/benchmarks/benchmark_comparison.png), [boundary_acoustic_detail.png](reports/benchmarks/boundary_acoustic_detail.png), [duration_distribution.png](reports/benchmarks/duration_distribution.png)
> - 📄 **Academic Whitepaper:** [docs/slice.pdf](docs/slice.pdf)

---

## 8. Verification & Benchmarking Commands

**1. Run Multi-Algorithm Benchmark Suite:**
```bash
.venv/bin/python scripts/run_benchmark.py --slicers energy silero webrtc whisper hybrid
```

**2. Audit Extracted Slice Boundaries (Phoneme Clipping Check):**
```bash
.venv/bin/python scripts/verify_boundaries.py --slices-dir datasets/processed/energy/slices
```
