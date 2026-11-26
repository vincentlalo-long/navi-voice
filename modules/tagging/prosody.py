"""
Acoustic Prosody DSP Extractor & Speaker-Adaptive Quantile Binning.

Extracts frame-level F0, voicing, spectral shape, and energy dynamics using
NumPy and SciPy, and assigns speaker-relative low/normal/high prosody bins.
"""

from functools import lru_cache
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sp_fft
from scipy import signal

PROSODY_EXTRACTOR_VERSION: str = "1.0.0"

_PROSODY_KEYS = (
    "f0_median_hz",
    "f0_iqr_hz",
    "f0_slope_hz_per_sec",
    "voiced_fraction",
    "spectral_centroid_hz",
    "spectral_flatness",
    "energy_std_db",
    "crest_factor_db",
)


def _empty_prosody_features() -> Dict[str, float]:
    """Return a fresh all-zero prosody feature dict."""
    return {key: 0.0 for key in _PROSODY_KEYS}


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert an arbitrary numeric array to a 1-D float32 mono waveform."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    # Collapse channel axis while preserving the longer time axis.
    if arr.ndim == 2:
        axis = 0 if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0] else 1
        return np.mean(arr, axis=axis, dtype=np.float32)
    time_axis = int(np.argmax(arr.shape))
    reduce_axes = tuple(i for i in range(arr.ndim) if i != time_axis)
    return np.mean(arr, axis=reduce_axes, dtype=np.float32)


@lru_cache(maxsize=16)
def _bandpass_sos(sampling_rate: int) -> Optional[np.ndarray]:
    """Cached 4th-order Butterworth bandpass SOS [60, min(1200, 0.45 * fs)] Hz."""
    low_hz = 60.0
    high_hz = min(1200.0, 0.45 * float(sampling_rate))
    if high_hz <= low_hz:
        return None
    return signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sampling_rate, output="sos"
    )


@lru_cache(maxsize=16)
def _hann_window(frame_len: int) -> np.ndarray:
    return np.hanning(frame_len).astype(np.float32)


@lru_cache(maxsize=16)
def _rfft_freqs_no_dc(frame_len: int, sampling_rate: int) -> np.ndarray:
    return np.fft.rfftfreq(frame_len, d=1.0 / float(sampling_rate))[1:].astype(
        np.float32
    )


def _bandpass_for_f0(arr: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Apply cached 4th-order Butterworth bandpass filter."""
    sos = _bandpass_sos(sampling_rate)
    if sos is None or len(arr) <= 12:
        return arr
    default_padlen = 3 * (2 * sos.shape[0] + 1)
    padlen = min(default_padlen, len(arr) - 1)
    return np.asarray(signal.sosfiltfilt(sos, arr, padlen=padlen), dtype=np.float32)


def _estimate_f0_nacf(
    bp_frames: np.ndarray,
    sampling_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-frame F0 (Hz) and voiced mask via FFT-accelerated NACF.

    Returns (f0_hz, voiced_mask) with shape (n_active_frames,).
    """
    n_frames, frame_len = bp_frames.shape
    f0_hz = np.zeros(n_frames, dtype=np.float64)
    voiced_mask = np.zeros(n_frames, dtype=bool)

    lag_min = max(1, int(sampling_rate / f0_max_hz))
    lag_max = min(frame_len - 2, int(sampling_rate / f0_min_hz))
    if lag_max < lag_min or n_frames == 0:
        return f0_hz, voiced_mask

    # Zero-mean each filtered frame before autocorrelation; n_fft >= frame_len + lag_max
    # prevents circular wrap-around for all searched lags [lag_min, lag_max].
    zm = bp_frames - np.mean(bp_frames, axis=1, keepdims=True)
    n_fft = sp_fft.next_fast_len(frame_len + lag_max)
    spec = sp_fft.rfft(zm, n=n_fft, axis=1)
    acf = sp_fft.irfft(spec * np.conj(spec), n=n_fft, axis=1)[:, : lag_max + 1]

    sq = zm.astype(np.float64)
    sq *= sq
    cs = np.cumsum(sq, axis=1)
    total_e = cs[:, -1:]

    lags = np.arange(lag_min, lag_max + 1)
    e1 = cs[:, frame_len - 1 - lags]
    e2 = total_e - cs[:, lags - 1]
    denom = np.sqrt(np.maximum(e1 * e2, 1e-24))

    nacf = acf[:, lag_min : lag_max + 1] / denom
    # Mild lag preference avoids subharmonic octave-down jumps on pure harmonics.
    lag_penalty = 1.0 - 0.02 * (lags.astype(np.float64) / float(frame_len))
    best_idx = np.argmax(nacf * lag_penalty[None, :], axis=1)
    row_idx = np.arange(n_frames)
    peak_r = nacf[row_idx, best_idx]
    voiced_mask = peak_r >= 0.45

    if np.any(voiced_mask):
        v_rows = np.flatnonzero(voiced_mask)
        v_idx = best_idx[v_rows]
        v_lags = lags[v_idx].astype(np.float64)

        interior = (v_idx > 0) & (v_idx < len(lags) - 1)
        if np.any(interior):
            ir = v_rows[interior]
            ii = v_idx[interior]
            y0 = nacf[ir, ii - 1]
            y1 = nacf[ir, ii]
            y2 = nacf[ir, ii + 1]
            denom_p = 2.0 * (y0 - 2.0 * y1 + y2)
            valid_p = np.abs(denom_p) > 1e-12
            delta = np.zeros_like(denom_p)
            delta[valid_p] = np.clip((y0[valid_p] - y2[valid_p]) / denom_p[valid_p], -0.5, 0.5)
            v_lags[interior] += delta

        f0_hz[v_rows] = float(sampling_rate) / np.maximum(v_lags, 1.0)

    return f0_hz, voiced_mask


def extract_prosody_features(
    audio: np.ndarray,
    sampling_rate: int,
    f0_min_hz: float = 75.0,
    f0_max_hz: float = 550.0,
) -> Dict[str, float]:
    """
    Extract acoustic prosody features from a waveform buffer.

    Guarded against empty, short (< 40 ms), non-finite, and silent/DC buffers;
    always returns finite Python floats for all 8 prosody keys.
    """
    if sampling_rate <= 0 or f0_min_hz <= 0.0 or f0_max_hz <= f0_min_hz:
        return _empty_prosody_features()

    arr = _to_mono_float32(audio)
    frame_len = int(0.040 * sampling_rate)
    hop_len = int(0.010 * sampling_rate)

    if arr.size == 0 or frame_len <= 0 or hop_len <= 0 or arr.size < frame_len:
        return _empty_prosody_features()

    if not np.all(np.isfinite(arr)):
        return _empty_prosody_features()

    max_abs = float(np.max(np.abs(arr)))
    if max_abs <= 1e-6 or float(np.std(arr)) <= 1e-6:
        return _empty_prosody_features()

    global_rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    peak_db = 20.0 * math.log10(max(max_abs, 1e-12))
    rms_db = 20.0 * math.log10(max(global_rms, 1e-12))
    crest_factor_db = float(max(0.0, peak_db - rms_db))

    frames_raw = sliding_window_view(arr, frame_len)[::hop_len]
    frames_zm = frames_raw - np.mean(frames_raw, axis=1, keepdims=True)
    frame_rms = np.sqrt(np.mean(frames_zm * frames_zm, axis=1))

    active_thresh = max(1e-5, 0.05 * global_rms)
    active_mask = frame_rms >= active_thresh
    if not np.any(active_mask):
        return _empty_prosody_features()

    active_frames = frames_zm[active_mask]
    active_rms = frame_rms[active_mask]
    active_times = np.flatnonzero(active_mask).astype(np.float64) * (
        float(hop_len) / float(sampling_rate)
    )

    # Energy std across active frames in dBFS.
    frame_db = 20.0 * np.log10(np.maximum(active_rms.astype(np.float64), 1e-12))
    energy_std_db = float(np.std(frame_db))

    # Hann-windowed rfft spectral centroid and spectral flatness on active frames.
    win_frames = active_frames * _hann_window(frame_len)[None, :]
    mag = np.abs(sp_fft.rfft(win_frames, axis=1))[:, 1:]
    freqs = _rfft_freqs_no_dc(frame_len, sampling_rate)

    if mag.shape[1] > 0:
        mag_sum = np.sum(mag, axis=1)
        valid_spec = mag_sum > 1e-12
        if np.any(valid_spec):
            valid_mag = mag[valid_spec]
            centroids = np.sum(valid_mag * freqs[None, :], axis=1) / mag_sum[valid_spec]
            spectral_centroid_hz = float(np.mean(centroids))

            # log(power) = 2 * log(mag), avoiding full-matrix squaring for geo mean.
            safe_mag = np.maximum(valid_mag, 1e-12)
            geo_mean = np.exp(2.0 * np.mean(np.log(safe_mag), axis=1))
            arith_mean = np.mean(safe_mag * safe_mag, axis=1)
            flatness = geo_mean / np.maximum(arith_mean, 1e-24)
            spectral_flatness = float(np.clip(np.mean(flatness), 0.0, 1.0))
        else:
            spectral_centroid_hz = 0.0
            spectral_flatness = 0.0
    else:
        spectral_centroid_hz = 0.0
        spectral_flatness = 0.0

    # Bandpass + NACF pitch tracking on active frames.
    bp_arr = _bandpass_for_f0(arr, sampling_rate)
    bp_frames = sliding_window_view(bp_arr, frame_len)[::hop_len][active_mask]
    f0_active, voiced_mask = _estimate_f0_nacf(
        bp_frames, sampling_rate, f0_min_hz, f0_max_hz
    )

    voiced_count = int(np.sum(voiced_mask))
    voiced_fraction = float(np.clip(voiced_count / float(len(active_frames)), 0.0, 1.0))

    if voiced_count >= 2:
        voiced_f0 = f0_active[voiced_mask]
        f0_median_hz = float(np.median(voiced_f0))
        p25, p75 = np.percentile(voiced_f0, [25.0, 75.0])
        f0_iqr_hz = float(p75 - p25)

        t_last = float(active_times[-1])
        tail_mask = voiced_mask & (active_times >= (t_last - 0.250 - 1e-9))
        if int(np.sum(tail_mask)) >= 2:
            t_tail = active_times[tail_mask]
            f_tail = f0_active[tail_mask]
            dt = t_tail - np.mean(t_tail)
            denom_t = float(np.sum(dt * dt))
            if denom_t > 1e-12:
                df = f_tail - np.mean(f_tail)
                f0_slope_hz_per_sec = float(np.sum(dt * df) / denom_t)
            else:
                f0_slope_hz_per_sec = 0.0
        else:
            f0_slope_hz_per_sec = 0.0
    else:
        f0_median_hz = 0.0
        f0_iqr_hz = 0.0
        f0_slope_hz_per_sec = 0.0

    result = {
        "f0_median_hz": f0_median_hz,
        "f0_iqr_hz": f0_iqr_hz,
        "f0_slope_hz_per_sec": f0_slope_hz_per_sec,
        "voiced_fraction": voiced_fraction,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_flatness": spectral_flatness,
        "energy_std_db": energy_std_db,
        "crest_factor_db": crest_factor_db,
    }
    return {k: (float(v) if math.isfinite(v) else 0.0) for k, v in result.items()}


def _default_eligible(record: Dict[str, Any]) -> bool:
    qc = record.get("qc")
    if isinstance(qc, dict):
        return qc.get("tier") != "REJECT"
    return True


def _finite_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _extract_speed(record: Dict[str, Any]) -> Optional[float]:
    asr = record.get("asr_diagnostics")
    if isinstance(asr, dict):
        for key in ("effective_mora_rate", "effective_cps"):
            val = _finite_float(asr.get(key))
            if val is not None:
                return val
    return _finite_float(record.get("chars_per_sec"))


def _extract_energy(record: Dict[str, Any]) -> Optional[float]:
    for key in ("raw_rms_db", "rms_db"):
        val = _finite_float(record.get(key))
        if val is not None:
            return val
    return None


def _compute_p20_p80(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    p20, p80 = np.percentile(np.asarray(values, dtype=np.float64), [20.0, 80.0])
    return float(p20), float(p80)


def _bin_value(
    value: Optional[float], p20: float, p80: float, has_quantiles: bool
) -> str:
    if value is None or not has_quantiles:
        return "normal"
    if value < p20:
        return "low"
    if value > p80:
        return "high"
    return "normal"


def assign_speaker_prosody_bins(
    records: List[Dict[str, Any]],
    eligible_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Fit speaker-relative P20/P80 quantiles on eligible records and assign
    ``record["prosody"]["bins"] = {"pitch", "speed", "energy"}`` in place.
    """
    predicate = eligible_predicate if eligible_predicate is not None else _default_eligible

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        speaker = record.get("speaker") or "default"
        grouped.setdefault(speaker, []).append(record)

    quantiles: Dict[str, Dict[str, float]] = {}
    for speaker, spk_records in grouped.items():
        pitch_vals: List[float] = []
        speed_vals: List[float] = []
        energy_vals: List[float] = []

        for rec in spk_records:
            if not predicate(rec):
                continue
            prosody = rec.get("prosody")
            if isinstance(prosody, dict):
                vf = _finite_float(prosody.get("voiced_fraction"))
                f0 = _finite_float(prosody.get("f0_median_hz"))
                if vf is not None and vf >= 0.15 and f0 is not None and f0 > 0.0:
                    pitch_vals.append(f0)

            spd = _extract_speed(rec)
            if spd is not None:
                speed_vals.append(spd)

            eng = _extract_energy(rec)
            if eng is not None:
                energy_vals.append(eng)

        pitch_p20, pitch_p80 = _compute_p20_p80(pitch_vals)
        speed_p20, speed_p80 = _compute_p20_p80(speed_vals)
        energy_p20, energy_p80 = _compute_p20_p80(energy_vals)

        quantiles[speaker] = {
            "pitch_p20": pitch_p20,
            "pitch_p80": pitch_p80,
            "speed_p20": speed_p20,
            "speed_p80": speed_p80,
            "energy_p20": energy_p20,
            "energy_p80": energy_p80,
        }

        has_pitch = len(pitch_vals) > 0
        has_speed = len(speed_vals) > 0
        has_energy = len(energy_vals) > 0

        for rec in spk_records:
            prosody = rec.get("prosody")
            if not isinstance(prosody, dict):
                prosody = {}
                rec["prosody"] = prosody

            vf = _finite_float(prosody.get("voiced_fraction"))
            f0 = _finite_float(prosody.get("f0_median_hz"))
            if vf is None or vf < 0.15 or f0 is None or f0 <= 0.0:
                pitch_bin = "unvoiced"
            else:
                pitch_bin = _bin_value(f0, pitch_p20, pitch_p80, has_pitch)

            speed_bin = _bin_value(
                _extract_speed(rec), speed_p20, speed_p80, has_speed
            )
            energy_bin = _bin_value(
                _extract_energy(rec), energy_p20, energy_p80, has_energy
            )

            prosody["bins"] = {
                "pitch": pitch_bin,
                "speed": speed_bin,
                "energy": energy_bin,
            }

    return quantiles
