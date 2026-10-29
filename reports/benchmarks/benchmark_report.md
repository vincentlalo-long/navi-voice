# Audio Slicer Comparative Benchmark Report

> [!IMPORTANT]
> **PHASE 1 EVALUATION ONLY: AUDIO SLICING & BOUNDARY SEGMENTATION**  
> This benchmark evaluates exclusively **Phase 1 (Slicing)** of the speech pipeline on **3.42 hours (12,301 seconds)** of continuous Japanese in-game dialogue (`kurisu_raw.wav`), extracted directly from video game visual novel voice lines: [YouTube Reference Source](https://youtu.be/31Ju6z28IIs?si=_gZqPTzEZL8xx-e7). Downstream stages (Phase 2: Semantic Tagging & ASR, Phase 3: Acoustic Modeling / Voice Cloning, Phase 4: Downstream Synthesis Quality such as MOS, SECS, CER) are independent and evaluated in subsequent pipeline phases.

---

## 1. Executive Summary & Comparative Matrix
![Benchmark Overview](benchmark_comparison.png)

## 2. Acoustic Boundary Quality & Discontinuity Inspection
![Acoustic Boundary Analysis](boundary_acoustic_detail.png)

## 3. Utterance Duration Spread & TTS Window Compliance
![Duration Distribution Analysis](duration_distribution.png)

---

## 4. Multi-Dimensional Benchmark Evaluation

### Table 1: Acoustic Boundary Quality (Physical Safety & Click Prevention)
Audited across the first and last 25ms of each extracted slice:
- **`CLEAN`**: Edge RMS $\le -35\text{ dBFS}$, $\text{EER} \le 0.20$, and step jump $|x[0]| \le 0.02$.
- **`SAFE`**: Edge RMS $\le -27\text{ dBFS}$, $\text{EER} \le 0.45$, and step jump $|x[0]| \le 0.05$.
- **`WARN`**: Speech clipping or transient step discontinuity risk.

| Slicer Algorithm | Total Slices | CBR % (Clean+Safe) | Clean % | Warn % | P95 Edge RMS | Max Step Jump | Consecutive Overlaps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`energy`** (DSP Strided RMS) | **3,582** | **99.6%** | **98.6%** | **0.4%** | **-46.6 dBFS** | **0.0576** | **0 (0.0ms)** |
| **`hybrid`** (Two-Stage Macro+Micro) | **2,424** | **89.4%** | **86.3%** | **10.6%** | **-26.4 dBFS** | **0.1765** | **0 (0.0ms)** |
| **`silero`** (Neural VAD LSTM) | 1,465 | 96.2% | 94.4% | 3.8% | -47.6 dBFS | 0.5096 | **0 (0.0ms)** |
| **`whisper`** (Semantic ASR DTW) | 2,405 | 78.1% | 69.0% | 21.9% | -18.1 dBFS | 0.6871 | **0 (0.0ms)** |
| **`webrtc`** (GMM Subband VAD) | 1,303 | 63.0% | 54.1% | 37.0% | -14.3 dBFS | 0.6898 | **0 (0.0ms)** |

---

### Table 2: Segmentation Distribution & TTS Golden Range [2s – 10s] Compliance
Evaluates conversational naturalness and attention window safety for autoregressive TTS models:

| Slicer Algorithm | Speech Coverage % | Mean Duration | Median Duration | P95 Duration | Golden Range [2s-10s] % | Short Phrases (< 2s) % | Run-on (> 10s) % | Over-Max (> 12s) % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hybrid`** | **99.8%** | **5.07s** | **4.94s** | **9.65s** | **85.6%** | **10.3%** | **4.1%** | **0.1%** |
| **`whisper`** | 95.7% | 4.90s | 4.69s | 9.41s | 85.2% | 11.4% | 3.5% | 0.1% |
| **`energy`** | 93.2% | 3.20s | 2.62s | 7.21s | 69.2% | 30.1% (Over-segmented) | 0.7% | 0.0% |
| **`silero`** | 96.8% | 8.13s | 8.93s | 11.71s | 60.1% | 5.0% | 34.9% (Risk OOM) | 0.0% |
| **`webrtc`** | 97.6% | 9.22s | 10.46s | 11.67s | 34.6% | 4.5% | 60.9% (Severely bridged) | 0.0% |

---

### Table 3: Computational Performance & Real-Time Factor (RTF)
Evaluated on Intel CPU + NVIDIA GeForce RTX 4060 Laptop GPU across 12,301 seconds of studio audio:  
*(Note: Whisper and Hybrid slicing runs were evaluated with `model_size="base"`. In the production mining pipeline, Phase 2 transcription defaults to `"large-v3"` for maximum text accuracy).*

| Slicer Algorithm | Total Execution Time | Real-Time Factor (RTF) | Processing Speedup | Hardware Mode |
| :--- | :---: | :---: | :---: | :---: |
| **`energy`** | **6.8 seconds** | **0.00055** | **1,809× faster than realtime** | CPU (Zero-copy SIMD) |
| **`webrtc`** | 10.1 seconds | 0.00082 | 1,218× faster than realtime | CPU (C lightweight) |
| **`silero`** | 144.9 seconds (2.4m) | 0.01178 | 85× faster than realtime | GPU (PyTorch JIT, cuda) |
| **`whisper`** | 223.0 seconds (3.7m) | 0.01813 | 55× faster than realtime | GPU (CTranslate2 float16, base) |
| **`hybrid`** | 265.1 seconds (4.4m) | 0.02155 | 46× faster than realtime | GPU (Whisper base) + CPU DSP |

---

## 5. Engineering Conclusion & Strategy Recommendation

1. **Primary Recommendation for Downstream TTS Training: `--slicer hybrid`**
   - **Why:** In neural TTS (GPT-SoVITS, CosyVoice, VITS), syntactic and prosodic clause completeness is paramount. `hybrid` is designed by construction to group words into natural grammatical clauses (`。！？`), preserves **99.8%** of dialogue audio, and places **85.6%** of utterances in the ideal 2s–10s TTS window. The micro-refinement layer boosts boundary cleanliness from 78.1% (Whisper) to **89.4% CBR** while halving boundary clipping warnings (10.6% vs 21.9%).
2. **Secondary Recommendation for High-Throughput Bulk Mining: `--slicer energy`**
   - **Why:** If processing large-scale datasets where GPU compute is unavailable, `energy` processes 3.42 hours in **6.8 seconds** on standard CPU with a **99.6% CBR** and near-zero run-on risk (0.7% >10s). Note that 30.1% of sentences will be fragmented into sub-2s phrases during natural micro-pauses.
3. **Disqualified for High-Quality TTS: `webrtc` and `silero`**
   - `webrtc` incurs a severe **37.0% WARN rate**, damaging unvoiced consonants.
   - `silero` bridges natural pauses aggressively, yielding **34.9% of slices > 10s**, which increases the risk of attention collapse and VRAM exhaustion during training.
