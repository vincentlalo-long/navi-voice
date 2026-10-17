# Audio Slicer Comparative Benchmark Report

Comprehensive multi-dimensional evaluation across Boundary Quality, Segmentation Integrity, and RTF performance.

## 1. Executive Summary & Comparative Matrix
![Benchmark Overview](benchmark_comparison.png)

## 2. Acoustic Boundary Quality & Discontinuity
![Acoustic Boundary Analysis](boundary_acoustic_detail.png)

## 3. Utterance Duration Spread & TTS Window Compliance
![Duration Distribution Analysis](duration_distribution.png)

## 4. Quantitative Benchmark Metrics

| Slicer Algorithm | Slices | CBR % (Clean+Safe) | Clean % | Warn % | Overlaps | Speech Coverage % | Mean Dur (s) | P95 Dur (s) | RTF (Speed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ENERGY** (EnergySlicer) | 3582 | **99.6%** | 97.9% | 0.5% | 0 (0.0ms) | 93.2% | 3.2s | 7.21s | **0.00087** |
| **SILERO** (SileroSlicer) | 1465 | **96.4%** | 94.8% | 3.5% | 0 (0.0ms) | 96.8% | 8.13s | 11.71s | **0.01362** |
| **WEBRTC** (WebRTCSlicer) | 1303 | **63.7%** | 55.4% | 36.3% | 0 (0.0ms) | 97.6% | 9.22s | 11.67s | **0.00117** |
| **WHISPER** (WhisperXSlicer) | 2479 | **79.7%** | 69.8% | 20.3% | 0 (0.0ms) | 95.4% | 4.73s | 9.2s | **0.02205** |

### Metric Definitions:
- **CBR % (Clean Boundary Rate)**: Percentage of slices with `CLEAN` or `SAFE` acoustic borders (Edge RMS <= -27 dB, EER <= 0.45, StepJump <= 0.05).
- **Overlaps**: Number of consecutive slice overlaps (Target: 0).
- **Speech Coverage %**: Proportion of raw audio preserved as dialogue.
- **RTF (Real-Time Factor)**: Processing time per second of audio (lower is faster).
