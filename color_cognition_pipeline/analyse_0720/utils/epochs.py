"""Epoch/feature I/O and baseline helpers."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def baseline_subtract(data: np.ndarray, times_ms: np.ndarray, interval_ms: tuple[float, float]):
    mask = (times_ms >= interval_ms[0]) & (times_ms <= interval_ms[1])
    if not mask.any():
        raise ValueError(f"No samples in baseline interval {interval_ms}")
    return data - data[..., mask].mean(axis=-1, keepdims=True)


def baseline_zscore(data: np.ndarray, times_ms: np.ndarray, interval_ms: tuple[float, float]):
    mask = (times_ms >= interval_ms[0]) & (times_ms <= interval_ms[1])
    if not mask.any():
        raise ValueError(f"No samples in baseline interval {interval_ms}")
    base = data[..., mask]
    mean = base.mean(axis=-1, keepdims=True)
    std = base.std(axis=-1, keepdims=True)
    return (data - mean) / np.maximum(std, np.finfo(float).eps)


def save_epochs(path: Path, data, times_ms, triggers, channel_names, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=data.astype(np.float32),
        times_ms=np.asarray(times_ms, dtype=np.float32),
        triggers=np.asarray(triggers, dtype=str),
        channel_names=np.asarray(channel_names, dtype=str),
    )
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_epochs(path: Path, prefer_clean: bool = True):
    path = Path(path)
    clean_path = path.with_name(f"{path.stem}_clean{path.suffix}")
    if prefer_clean and not path.stem.endswith("_clean") and clean_path.exists():
        path = clean_path
    x = np.load(path, allow_pickle=False)
    return {key: x[key] for key in x.files}
