"""Shared plotting primitives; every function can display and save."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, output=None):
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
    return fig


def plot_decoding(times, accuracy, p_cluster=None, title="Decoding", output=None):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(times, accuracy, lw=2.2, color="#2457a6", label="Accuracy")
    ax.axhline(0.5, color="0.35", ls="--", lw=1.2, label="Chance")
    ax.axvline(0, color="0.65", lw=1)
    if p_cluster is not None:
        sig = np.asarray(p_cluster) < 0.05
        if sig.any():
            y = min(np.nanmin(accuracy), 0.49)
            ax.scatter(np.asarray(times)[sig], np.full(sig.sum(), y), s=12, color="#d62728", label="cluster p<.05")
    ax.set(xlabel="Time (ms)", ylabel="Accuracy", title=title)
    ax.legend(frameon=False)
    return finish(fig, output)


def plot_tgm(matrix, times, title="Temporal generalization", output=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=0.35, vmax=0.65,
                      extent=[times[0], times[-1], times[0], times[-1]])
    ax.plot([times[0], times[-1]], [times[0], times[-1]], color="k", lw=.7, alpha=.5)
    ax.set(xlabel="Test time (ms)", ylabel="Train time (ms)", title=title)
    fig.colorbar(image, ax=ax, label="Accuracy")
    return finish(fig, output)


def plot_condition_summary(table, title, output=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(table["trigger"].astype(str), table["mean"], yerr=table.get("sem"), color="#4c78a8", alpha=.85)
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel="Trigger", ylabel="Mean response", title=title)
    return finish(fig, output)
