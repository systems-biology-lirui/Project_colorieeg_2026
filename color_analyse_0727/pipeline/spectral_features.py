"""Spectrum-level feature extraction for the rebuilt analysis pipeline.

All functions are pure (no file I/O). Band features are computed with Welch
windowed-FFT power, so no additional per-epoch band-pass filtering is needed;
the analysis-window power is baseline z-scored at comparison time.

The 19 candidate bands step by 10 Hz from 5 to 195 Hz. The 45-55, 95-105 and
145-155 Hz bands are removed because they bracket the 50/100/150 Hz line-noise
harmonics already notched in preprocessing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

FS = 500.0

BANDS_19: tuple[tuple[float, float], ...] = tuple(
    (float(lo), float(lo + 10.0)) for lo in range(5, 195, 10)
)
NOISE_BANDS: frozenset[tuple[float, float]] = frozenset(
    {(45.0, 55.0), (95.0, 105.0), (145.0, 155.0)}
)
FEATURE_BANDS: tuple[tuple[float, float], ...] = tuple(
    band for band in BANDS_19 if band not in NOISE_BANDS
)
BAND_NAMES: tuple[str, ...] = tuple(
    f"{int(lo)}-{int(hi)}Hz" for lo, hi in FEATURE_BANDS
)


def padded_bandpass(
    data: np.ndarray,
    lo: float,
    hi: float,
    fs: float = FS,
    pad_samples: int = 500,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass with symmetric padding on the last axis.

    Filtering short epochs directly with ``sosfiltfilt`` creates edge
    transients that can reach 5-25% of the analysis-window RMS; padding with a
    symmetric reflection and cropping removes most of that artifact. The
    residual is a small (about 4-6% of window RMS for 1/f-like signals),
    zero-mean, condition-symmetric deviation from ideal continuous filtering;
    it adds a little noise to trial-wise statistics but does not bias the
    color-vs-gray comparison.
    """
    values = np.asarray(data, dtype=np.float64)
    low = max(float(lo), 1e-3)
    high = min(float(hi), fs / 2.0 - 1.0)
    if high <= low:
        raise ValueError(f"Invalid band [{lo}, {hi}] at fs={fs}")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    pad_width = [(0, 0)] * (values.ndim - 1) + [(pad_samples, pad_samples)]
    padded = np.pad(values, pad_width, mode="symmetric")
    filtered = signal.sosfiltfilt(sos, padded, axis=-1)
    index = [slice(None)] * (values.ndim - 1) + [
        slice(pad_samples, pad_samples + values.shape[-1])
    ]
    return np.asarray(filtered[tuple(index)], dtype=np.float64)


def welch_band_power(
    data: np.ndarray,
    fs: float = FS,
    bands: Sequence[tuple[float, float]] = FEATURE_BANDS,
    nperseg: int = 64,
    noverlap: int | None = None,
    detrend: str = "constant",
) -> np.ndarray:
    """Log band power with Welch's method on the last axis of ``data``.

    ``data`` is ``(..., time)`` and the return value is ``(..., n_bands)``.
    A frequency bin belongs to the band whose half-open interval contains it.
    """
    values = np.asarray(data, dtype=np.float64)
    if noverlap is None:
        noverlap = max(0, nperseg // 2)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty((flat.shape[0], len(bands)), dtype=np.float64)
    for i in range(flat.shape[0]):
        freqs, psd = signal.welch(
            flat[i],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
        )
        for j, (lo, hi) in enumerate(bands):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                out[i, j] = float(np.log(np.mean(psd[mask]) + 1e-12))
            else:
                out[i, j] = np.nan
    return out.reshape(values.shape[:-1] + (len(bands),))


def band_power_baseline_z(
    analysis_power: np.ndarray,
    baseline_power: np.ndarray,
) -> np.ndarray:
    """Z-score analysis-window log power against the baseline-trial distribution.

    Both arrays are ``(trials, channels, bands)``. The mean and standard
    deviation of the baseline power are estimated across the trial axis for
    each channel/band; a near-zero baseline SD is floored to 1 so the z-score
    is defined everywhere.
    """
    analysis = np.asarray(analysis_power, dtype=np.float64)
    baseline = np.asarray(baseline_power, dtype=np.float64)
    mu = np.nanmean(baseline, axis=0, keepdims=True)
    sd = np.nanstd(baseline, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    return (analysis - mu) / sd


def window_mask(time_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    """Boolean mask over the time axis for ``[start_ms, end_ms]``."""
    time_ms = np.asarray(time_ms, dtype=float)
    return (time_ms >= start_ms) & (time_ms <= end_ms)
