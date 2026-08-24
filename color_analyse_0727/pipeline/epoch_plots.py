"""Reusable epoch-average plots with a mean line and a shaded uncertainty band."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _resolve_channels(
    channel_labels: Sequence[str],
    channels: Iterable[str | int] | None,
) -> list[tuple[int, str]]:
    labels = [str(label) for label in channel_labels]
    lookup = {label.upper(): index for index, label in enumerate(labels)}
    requested = list(channels) if channels is not None else list(range(len(labels)))
    resolved: list[tuple[int, str]] = []
    for channel in requested:
        if isinstance(channel, (int, np.integer)):
            index = int(channel)
            if not 0 <= index < len(labels):
                raise IndexError(f"Channel index out of range: {index}")
        else:
            key = str(channel).upper()
            if key not in lookup:
                raise KeyError(f"Channel not found: {channel}")
            index = lookup[key]
        resolved.append((index, labels[index]))
    if not resolved:
        raise ValueError("At least one channel is required")
    return resolved


def _mean_and_band(epochs: np.ndarray, shade: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(epochs, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected (trials, time) data, got {values.shape}")
    mean = np.nanmean(values, axis=0)
    if shade == "sd":
        band = np.nanstd(values, axis=0, ddof=1)
    elif shade == "sem":
        band = np.nanstd(values, axis=0, ddof=1) / np.sqrt(max(values.shape[0], 1))
    elif shade == "none":
        band = np.zeros_like(mean)
    else:
        raise ValueError("shade must be 'sem', 'sd', or 'none'")
    return mean, band


def plot_epoch_mean_shading(
    epochs: np.ndarray,
    time_ms: Sequence[float],
    channel_labels: Sequence[str],
    channels: Iterable[str | int],
    *,
    condition_label: str | None = None,
    shade: str = "sem",
    color: str | None = None,
    axes=None,
    figsize: tuple[float, float] | None = None,
    title_prefix: str = "",
):
    """Plot selected channels as trial means with SEM/SD shading.

    Parameters
    ----------
    epochs:
        Array shaped ``(trials, channels, time)``.
    time_ms:
        Time axis with one value per time point.
    channel_labels:
        Labels corresponding to the channel dimension.
    channels:
        Channel labels or integer indices to draw.
    shade:
        ``"sem"`` (default), ``"sd"``, or ``"none"``.
    """

    values = np.asarray(epochs, dtype=float)
    time = np.asarray(time_ms, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected (trials, channels, time), got {values.shape}")
    if values.shape[2] != time.size:
        raise ValueError("Time axis length does not match epoch data")
    resolved = _resolve_channels(channel_labels, channels)
    if axes is None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            len(resolved),
            1,
            sharex=True,
            figsize=figsize or (10.0, max(2.6 * len(resolved), 3.2)),
            squeeze=False,
        )
        axes = list(axes[:, 0])
    else:
        figure = axes[0].figure if isinstance(axes, (list, tuple)) else axes.figure
        axes = list(np.atleast_1d(axes).ravel())
        if len(axes) != len(resolved):
            raise ValueError("Number of supplied axes must match selected channels")

    for axis, (index, label) in zip(axes, resolved):
        mean, band = _mean_and_band(values[:, index, :], shade)
        line = axis.plot(time, mean, color=color, linewidth=1.8, label=condition_label)
        line_color = line[0].get_color()
        if shade != "none":
            axis.fill_between(time, mean - band, mean + band, color=line_color, alpha=0.20)
        axis.axvline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axhline(0.0, color="0.75", linewidth=0.6)
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        if title_prefix or condition_label:
            title = " - ".join(part for part in (title_prefix, condition_label, label) if part)
            axis.set_title(title, loc="left", fontsize=10)

    axes[-1].set_xlabel("Time (ms)")
    return figure, axes


def plot_hdf5_conditions(
    h5_path: Path,
    conditions: Iterable[str],
    channels: Iterable[str | int],
    *,
    shade: str = "sem",
    colors: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Overlay condition means and shaded uncertainty bands from one HDF5 file."""

    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as h5:
        time_ms = np.asarray(h5["time_ms"][()], dtype=float)
        labels = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["labels"][()]
        ]
        selected = _resolve_channels(labels, channels)
        condition_list = list(conditions)
        if not condition_list:
            raise ValueError("At least one condition is required")
        epoch_arrays = {
            condition: np.asarray(h5["epochs"][condition][()], dtype=float)
            for condition in condition_list
        }

    figure, axes = plt.subplots(
        len(selected),
        1,
        sharex=True,
        figsize=figsize or (10.0, max(2.8 * len(selected), 3.5)),
        squeeze=False,
    )
    axes = list(axes[:, 0])
    for condition_index, condition in enumerate(condition_list):
        plot_epoch_mean_shading(
            epoch_arrays[condition],
            time_ms,
            labels,
            [index for index, _ in selected],
            condition_label=condition,
            shade=shade,
            color=(colors or {}).get(condition),
            axes=axes,
        )
    for axis, (_, label) in zip(axes, selected):
        axis.set_title(label, loc="left", fontsize=10)
        axis.legend(frameon=False, ncol=min(len(condition_list), 4))
    if title:
        figure.suptitle(title, fontsize=13, y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        figure.tight_layout()
    return figure, axes

