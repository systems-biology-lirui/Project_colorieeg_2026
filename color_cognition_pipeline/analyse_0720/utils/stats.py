"""Small statistical utilities shared by notebooks."""
from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    return false_discovery_control(p, method="bh")


def contiguous_runs(mask: np.ndarray, min_length: int = 1):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e - s >= min_length]


def cluster_permutation_1d(observed, null, chance=0.5, alpha=0.05):
    """Positive cluster-mass correction for a decoding time course."""
    observed = np.asarray(observed, float) - chance
    null = np.asarray(null, float) - chance
    threshold = np.quantile(null, 1 - alpha, axis=0)
    null_max = np.zeros(null.shape[0])
    for pi, curve in enumerate(null):
        runs = contiguous_runs(curve > threshold)
        null_max[pi] = max((curve[s:e].sum() for s, e in runs), default=0.0)
    corrected = np.ones(observed.size)
    for start, stop in contiguous_runs(observed > threshold):
        mass = observed[start:stop].sum()
        p = (1 + np.sum(null_max >= mass)) / (1 + null.shape[0])
        corrected[start:stop] = p
    return corrected
