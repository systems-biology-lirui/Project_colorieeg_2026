"""Robust epoch QC used before all downstream analyses."""
from __future__ import annotations

import numpy as np


def robust_abs_z(values, axis=0):
    values = np.asarray(values, float)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    return 0.6745 * np.abs(values - median) / np.maximum(mad, np.finfo(float).eps)


def detect_bad_epochs(data, z_threshold=6.0, bad_channel_fraction=0.15):
    """Return trial keep mask and interpretable metrics for [trial,ch,time]."""
    peak_to_peak = np.ptp(data, axis=2)
    rms = np.sqrt(np.mean(np.square(data), axis=2))
    line_length = np.mean(np.abs(np.diff(data, axis=2)), axis=2)
    channel_bad = (
        (robust_abs_z(peak_to_peak, axis=0) > z_threshold)
        | (robust_abs_z(rms, axis=0) > z_threshold)
        | (robust_abs_z(line_length, axis=0) > z_threshold)
    )
    fraction = channel_bad.mean(axis=1)
    global_metrics = np.column_stack((np.median(peak_to_peak, axis=1), np.median(rms, axis=1), np.median(line_length, axis=1)))
    global_bad = (robust_abs_z(global_metrics, axis=0) > z_threshold).any(axis=1)
    bad = (fraction >= bad_channel_fraction) | global_bad
    return ~bad, {
        "bad_channel_fraction": fraction,
        "median_peak_to_peak": global_metrics[:, 0],
        "median_rms": global_metrics[:, 1],
        "median_line_length": global_metrics[:, 2],
    }
