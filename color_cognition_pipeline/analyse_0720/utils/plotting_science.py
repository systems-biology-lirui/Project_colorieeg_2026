"""Compact, publication-oriented figures shared by the 20-mm batch."""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "blue": "#3B6FB6", "orange": "#E07A2D", "green": "#2A9D6F",
    "red": "#C94C4C", "purple": "#7A5AA6", "gray": "#687078",
}


def set_science_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 600, "savefig.transparent": False,
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.titleweight": "semibold", "axes.linewidth": .8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.width": .7, "ytick.major.width": .7,
        "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, pad=2)


def save_figure(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return fig


def decoding_curve(times, accuracy, title, path, null=None, p_cluster=None, color=None):
    set_science_style(); color = color or COLORS["blue"]
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    times=np.asarray(times); accuracy=np.asarray(accuracy)
    if null is not None:
        lo,hi=np.quantile(null,[.025,.975],axis=0)
        ax.fill_between(times,lo,hi,color=COLORS["gray"],alpha=.16,lw=0,label="95% null")
    ax.plot(times,accuracy,color=color,label="Observed")
    ax.axhline(.5,color="black",ls=(0,(3,2)),lw=.8); ax.axvline(0,color=".72",lw=.7)
    if p_cluster is not None:
        sig=np.asarray(p_cluster)<.05
        if sig.any(): ax.scatter(times[sig],np.full(sig.sum(),ax.get_ylim()[0]),s=5,color=COLORS["red"],clip_on=False)
    ax.set(xlabel="Time from stimulus onset (ms)",ylabel="Accuracy",title=title)
    ax.legend(loc="upper right"); _clean(ax); return save_figure(fig,path)


def group_decoding(times, curves, labels, title, path, chance=.5):
    set_science_style(); curves=np.asarray(curves,float); times=np.asarray(times)
    fig,ax=plt.subplots(figsize=(3.5,2.35),constrained_layout=True)
    for curve,label in zip(curves,labels): ax.plot(times,curve,color=COLORS["gray"],alpha=.42,lw=.8)
    mean=np.nanmean(curves,axis=0)
    if len(curves)>1:
        sem=np.nanstd(curves,axis=0,ddof=1)/np.sqrt(len(curves)); ax.fill_between(times,mean-sem,mean+sem,color=COLORS["blue"],alpha=.20,lw=0)
    ax.plot(times,mean,color=COLORS["blue"],lw=1.8,label=f"Mean (n={len(curves)})")
    ax.axhline(chance,color="black",ls=(0,(3,2)),lw=.8); ax.axvline(0,color=".72",lw=.7)
    ax.set(xlabel="Time from stimulus onset (ms)",ylabel="Accuracy",title=title); ax.legend(); _clean(ax)
    return save_figure(fig,path)


def electrode_distance_plot(table, subject, path):
    set_science_style(); d=table.sort_values("target_distance_mm")
    fig,ax=plt.subplots(figsize=(3.2,max(2.0,.12*len(d)+.8)),constrained_layout=True)
    colors=[COLORS["blue"] if x=="left" else COLORS["orange"] for x in d.target_side]
    ax.scatter(d.target_distance_mm,np.arange(len(d)),c=colors,s=17,edgecolor="white",linewidth=.35)
    ax.set_yticks(np.arange(len(d)),d.channel); ax.axvline(20,color="black",ls=(0,(3,2)),lw=.8)
    ax.set(xlabel="Distance to nearest fMRI peak (mm)",title=f"{subject} · 20-mm coverage"); _clean(ax)
    return save_figure(fig,path)


def effect_forest(table, title, path):
    set_science_style(); d=table.sort_values("effect")
    fig,ax=plt.subplots(figsize=(3.4,max(2.0,.16*len(d)+.8)),constrained_layout=True)
    ax.errorbar(d.effect,np.arange(len(d)),xerr=d.se,fmt="o",ms=3,color=COLORS["blue"],ecolor=".5",elinewidth=.8,capsize=2)
    ax.axvline(0,color="black",lw=.8); ax.set_yticks(np.arange(len(d)),d.label); ax.set(xlabel="Color − gray response",title=title); _clean(ax)
    return save_figure(fig,path)
