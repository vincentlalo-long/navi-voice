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
| [Slicing — Phase 2](docs/slice.pdf) | Constrained interval formulation, five segmentation strategies, signal conditioning, evaluation methodology, and the full 3.42-hour benchmark. |

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

Generated slices and metadata (`metadata.json`, `metadata.csv`, `dataset.list`) are stored in isolated directories under `datasets/processed/<slicer>/`.

---

## 6. Multi-Algorithm Benchmarking & Empirical Findings

> [!IMPORTANT]
> **PHASE 2 EVALUATION ONLY: AUDIO SLICING & BOUNDARY SEGMENTATION**  
> This benchmark evaluates exclusively **Phase 2 (Speech Slicing & Boundary Segmentation)** on **3.42 hours (12,301 seconds)** of continuous Japanese in-game dialogue (`datasets/raw/kurisu_raw.wav`), extracted directly from video game visual novel voice lines: [YouTube Reference Source](https://youtu.be/31Ju6z28IIs?si=_gZqPTzEZL8xx-e7).  
> 
> A complete speech dataset preparation pipeline spans 4 distinct, decoupled phases:
> - **Phase 1: Collection:** Source acquisition, speaker and language metadata, and licensing.
> - **Phase 2: Audio Slicing & Boundary Segmentation (Evaluated Below):** Acoustically clean silence cuts (zero transient clicks, no phoneme clipping), optimal utterance duration distribution ([2s – 10s]), zero boundary overlaps, and maximal speech retention.
> - **Phase 3: Semantic Tagging & ASR Transcription:** Faster-Whisper text transcription, Emotion / Sentiment categorization, prosodic tone tagging, and speaker diarization.
> - **Phase 4: Acoustic Modeling & Downstream Synthesis Evaluation:** Model training (e.g., GPT-SoVITS, CosyVoice, F5-TTS, VITS) and end-to-end quality metrics on synthesized speech (MOS, SECS speaker similarity, CER/WER, Mel-Cepstral Distortion).

The benchmark suite evaluates all 5 candidate algorithms on the continuous dialogue dataset:

```bash
.venv/bin/python scripts/run_benchmark.py --slicers energy silero webrtc whisper hybrid
```

---

### Multi-Dimensional Empirical Results (Full 3.42-Hour Dataset: 12,301s)

#### Table 1: Acoustic Boundary Quality (Physical Safety & Click Prevention)
Audited across the first and last 25ms of each extracted slice:
- **`CLEAN`**: Edge RMS $\le -35\text{ dBFS}$, $\text{EER} \le 0.20$, and step jump $|x[0]| \le 0.02$.
- **`SAFE`**: Edge RMS $\le -27\text{ dBFS}$, $\text{EER} \le 0.45$, and step jump $|x[0]| \le 0.05$.
- **`WARN`**: Speech clipping or transient step discontinuity risk.

| Slicer Strategy | Slices Extracted | Clean Boundary Rate (CBR %) | Clean % | Warn % | P95 Edge RMS | Max Step Jump | Overlaps (Count / ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`energy`** (DSP Strided RMS) | **3,582** | **99.6%** | **98.6%** | **0.4%** | **-46.6 dBFS** | **0.0576** | **0 (0.0ms)** |
| **`hybrid`** (Two-Stage Macro+Micro) | **2,424** | **89.4%** | **86.3%** | **10.6%** | **-26.4 dBFS** | **0.1765** | **0 (0.0ms)** |
| **`silero`** (Neural VAD LSTM) | 1,465 | 96.2% | 94.4% | 3.8% | -47.6 dBFS | 0.5096 | **0 (0.0ms)** |
| **`whisper`** (Semantic ASR DTW) | 2,405 | 78.1% | 69.0% | 21.9% | -18.1 dBFS | 0.6871 | **0 (0.0ms)** |
| **`webrtc`** (GMM Sub-band VAD) | 1,303 | 63.0% | 54.1% | 37.0% | -14.3 dBFS | 0.6898 | **0 (0.0ms)** |

---

#### Table 2: Segmentation Distribution & TTS Golden Range [2s – 10s] Compliance
Evaluates conversational naturalness and attention window safety for autoregressive TTS models:

| Slicer Strategy | Speech Coverage % | Mean Duration | Median Duration | P95 Duration | Golden Range [2s-10s] % | Short Phrases (< 2s) % | Run-on (> 10s) % | Over-Max (> 12s) % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hybrid`** | **99.8%** | **5.07s** | **4.94s** | **9.65s** | **85.6%** | **10.3%** | **4.1%** | **0.1%** |
| **`whisper`** | 95.7% | 4.90s | 4.69s | 9.41s | 85.2% | 11.4% | 3.5% | 0.1% |
| **`energy`** | 93.2% | 3.20s | 2.62s | 7.21s | 69.2% | 30.1% (Over-segmented) | 0.7% | 0.0% |
| **`silero`** | 96.8% | 8.13s | 8.93s | 11.71s | 60.1% | 5.0% | 34.9% (Attention risk) | 0.0% |
| **`webrtc`** | 97.6% | 9.22s | 10.46s | 11.67s | 34.6% | 4.5% | 60.9% (Over-bridged) | 0.0% |

---

#### Table 3: Computational Performance & Real-Time Factor (RTF)
Evaluated across 12,301 seconds on Intel CPU + NVIDIA GeForce RTX 4060 Laptop GPU:  
*(Note: Whisper and Hybrid slicing runs were evaluated with `model_size="base"`. In the production mining pipeline, Phase 3 transcription defaults to `"large-v3"` for maximum text accuracy).*

| Slicer Strategy | Execution Time | Real-Time Factor (RTF) | Processing Speedup | Hardware Mode |
| :--- | :---: | :---: | :---: | :---: |
| **`energy`** | **6.8 seconds** | **0.00055** | **1,809× faster than realtime** | CPU (Zero-copy SIMD) |
| **`webrtc`** | 10.1 seconds | 0.00082 | 1,218× faster than realtime | CPU (C lightweight) |
| **`silero`** | 144.9 seconds (2.4m) | 0.01178 | 85× faster than realtime | GPU (PyTorch JIT, cuda) |
| **`whisper`** | 223.0 seconds (3.7m) | 0.01813 | 55× faster than realtime | GPU (CTranslate2 float16, base) |
| **`hybrid`** | 265.1 seconds (4.4m) | 0.02155 | 46× faster than realtime | GPU (Whisper base) + CPU DSP |

---

### Comparative Visualizations

#### 1. Executive Summary & Comparative Matrix
![Benchmark Overview](reports/benchmarks/benchmark_comparison.png)

#### 2. Acoustic Boundary Quality & Discontinuity Inspection
![Acoustic Boundary Analysis](reports/benchmarks/boundary_acoustic_detail.png)

#### 3. Utterance Duration Spread vs TTS Golden Range [2s – 10s]
![Duration Distribution Analysis](reports/benchmarks/duration_distribution.png)

---

## 7. Engineering Analysis & Slicer Recommendation

### 🏆 Primary Recommendation for TTS Dataset Creation: **`--slicer hybrid`**

For creating production-grade training corpora for modern autoregressive / diffusion TTS architectures (GPT-SoVITS, CosyVoice, VITS, F5-TTS):

1. **Syntactic & Prosodic Clause Alignment**:
   - Anchors boundaries to linguistic punctuation marks (`。！？`) by construction, ensuring utterances are grammatically complete clauses rather than arbitrary sound fragments.
2. **Superior Golden Range Compliance (85.6% in [2s – 10s])**:
   - Places **85.6%** of utterances directly into the ideal $2\text{s} - 10\text{s}$ attention training window (mean $5.07\text{s}$, P95 $9.65\text{s}$), with only $0.1\%$ exceeding $12\text{s}$.
3. **Maximal Speech Preservation (99.8% Speech Coverage)**:
   - Preserves virtually all dialogue content ($12,281.91\text{s}$ out of $12,301.43\text{s}$), achieving the highest audio retention of all tested strategies.
4. **DSP Micro-Refinement Minimizes Acoustic Discontinuity**:
   - The micro-refinement layer scans inter-utterance pauses for the lowest RMS silence dip and snaps to zero-crossings, lifting CBR from $78.1\%$ (vanilla Whisper) to **$89.4\%$** and cutting the clipping warning rate by more than half ($10.6\%$ vs $21.9\%$); the raised-cosine edge fade then smoothly attenuates residual boundary energy.

---

### ⚡ Secondary Recommendation for Bulk Data Mining: **`--slicer energy`**

When processing large-scale raw audio corpora where GPU resources are unavailable:
- **1,809× Faster than Real-Time**: Processes 3.42 hours in just **6.8 seconds** on standard CPU.
- **High Boundary Cleanliness & Run-on Avoidance (99.6% CBR, 0.4% Warn)**: P95 edge RMS at $-46.6\text{ dBFS}$ with only 0.7% exceeding 10s.
- **Known Trade-off**: Mid-sentence micro-pauses fragment $30.1\%$ of speech into short sub-2s phrases, which Energy cannot distinguish from clause endings.

---

### ⚠️ Disqualified for High-Quality TTS Training:
- **`webrtc`**: Unacceptable **37.0% WARN clipping rate**; GMM sub-band classification truncates Japanese unvoiced consonants ($/s/$, $/sh/$, $/h/$, $/k/$).
- **`silero`**: Aggressive pause bridging leads to **34.9% run-on utterances > 10s** (mean $8.13\text{s}$), increasing the risk of VRAM exhaustion and attention collapse in autoregressive TTS models.

---

## 8. Boundary Quality Verification

To mathematically inspect whether slice boundaries landed smoothly inside silence intervals or clipped active phonemes on extracted files:

```bash
.venv/bin/python scripts/verify_boundaries.py --slices-dir datasets/processed/energy/slices
```

Each slice is audited across its first and last 25ms:
- **`[CLEAN]`**: Edge RMS $\le -35\text{ dBFS}$ (quiet silence onset/offset).
- **`[SAFE]`**: Edge RMS $\le -27\text{ dBFS}$ with Edge Energy Ratio (EER) $\le 0.45$ and transient jump $|x[0]| \le 0.05$.
- **`[WARN]`**: Edge energy exceeding thresholds, indicating potential phonetic clipping.

