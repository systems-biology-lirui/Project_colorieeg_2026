"""Shared continuous-signal preprocessing used by export and QC."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .config import BANDPASS_HZ, NOTCH_HZ, NOTCH_Q, TARGET_SFREQ


def filter_continuous(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Resample and apply the final zero-phase filter chain.

    The same function is used by the HDF5 exporter and the post-filter QC so
    that the QC spectrum reflects the signal that will enter the analysis.
    The input is never modified in place.
    """

    from scipy import signal

    values = np.asarray(data, dtype=np.float64)
    # Remove the channel-wise ADC/DC offset before zero-phase filtering. The
    # source recordings contain a very large constant offset; leaving it in
    # place creates an artificial edge transient in sosfiltfilt. This is not
    # event-wise baseline correction and does not remove evoked responses.
    center = np.nanmedian(values, axis=-1, keepdims=True)
    values = values - center
    if not np.isclose(sfreq, TARGET_SFREQ):
        ratio = Fraction(TARGET_SFREQ / sfreq).limit_denominator(1000)
        values = signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)

    high = min(BANDPASS_HZ[1], TARGET_SFREQ / 2.0 - 1.0)
    sos = signal.butter(
        4,
        [BANDPASS_HZ[0], high],
        btype="bandpass",
        fs=TARGET_SFREQ,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, values, axis=-1)
    for frequency in NOTCH_HZ:
        if frequency >= TARGET_SFREQ / 2.0:
            continue
        b, a = signal.iirnotch(frequency, NOTCH_Q, fs=TARGET_SFREQ)
        filtered = signal.filtfilt(b, a, filtered, axis=-1)
    return np.asarray(filtered, dtype=np.float64)
