from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
if str(MODULE) not in sys.path:
    sys.path.insert(0, str(MODULE))
from pipeline.spectral_features import padded_bandpass

RESULT_DIR: Path | None = None
VARIANT = "100-400_lf30"
CSC_PATH: Path | None = None
CSC_COLUMN = "CSC"


def csc_table() -> pd.DataFrame:
    global CSC_PATH
    if CSC_PATH is None:
        if RESULT_DIR is None:
            raise ValueError("RESULT_DIR is not configured")
        CSC_PATH = (
            RESULT_DIR
            / "stage01_selection"
            / f"electrode_sets_and_csc_{VARIANT}.csv"
        )
    return pd.read_csv(CSC_PATH).query(CSC_COLUMN)[["subject", "channel"]]


def process_root() -> Path:
    return MODULE / "process_data"


def figures_root() -> Path:
    if RESULT_DIR is None:
        raise ValueError("RESULT_DIR is not configured")
    return RESULT_DIR / "stage02_amplitude_spectral" / "figures"

CONDITIONS = [
    ("face_color", "face", "color"),
    ("face_gray", "face", "gray"),
    ("object_color", "object", "color"),
    ("object_gray", "object", "gray"),
    ("body_color", "body", "color"),
    ("body_gray", "body", "gray"),
    ("place_color", "place", "color"),
    ("place_gray", "place", "gray"),
]


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def collect():
    csc = csc_table()
    subjects = []
    all_values = []
    global_min = np.inf
    global_max = -np.inf
    for subject, group in csc.groupby("subject", sort=True):
        path = process_root() / subject / "task1_epoched_1_200Hz.h5"
        with h5py.File(path, "r") as h5:
            labels = [decode(v).strip().upper() for v in h5["labels"][()]]
            time = np.asarray(h5["time_ms"][()], dtype=float)
            keep = np.arange(0, len(time), 2)
            baseline = (time >= -200) & (time <= 0)
            for channel in group["channel"].astype(str):
                idx = labels.index(channel.upper())
                series = []
                for condition, category, color in CONDITIONS:
                    data = np.asarray(h5["epochs"][condition][:, idx, :], dtype=np.float32)
                    data = padded_bandpass(data, 1.0, 150.0)
                    data = data - np.nanmean(data[:, baseline], axis=1, keepdims=True)
                    mean = np.nanmean(data, axis=0)[keep]
                    sem = np.nanstd(data, axis=0, ddof=1)[keep] / np.sqrt(max(data.shape[0], 1))
                    global_min = min(global_min, float(np.nanmin(mean - sem)))
                    global_max = max(global_max, float(np.nanmax(mean + sem)))
                    series.append({
                        "condition": condition,
                        "category": category,
                        "color": color,
                        "mean": mean.astype(float).tolist(),
                        "sem": sem.astype(float).tolist(),
                        "n_trials": int(data.shape[0]),
                    })
                subjects.append({
                    "subject": subject,
                    "channel": channel,
                    "time": time[keep].astype(float).tolist(),
                    "series": series,
                })
    pad = max((global_max - global_min) * 0.08, 1e-6)
    return {"panels": subjects, "y_min": global_min - pad, "y_max": global_max + pad}


def save_png(payload):
    figures = figures_root()
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"face": "#e45756", "object": "#4c78a8", "body": "#59a14f", "place": "#b279a2"}
    fig, axes = plt.subplots(4, 2, figsize=(18, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, panel in zip(axes, payload["panels"]):
        time = np.asarray(panel["time"], dtype=float)
        for series in panel["series"]:
            mean = np.asarray(series["mean"], dtype=float)
            sem = np.asarray(series["sem"], dtype=float)
            color = colors[series["category"]]
            alpha = 0.12 if series["color"] == "color" else 0.07
            ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=alpha, linewidth=0)
            ax.plot(
                time,
                mean,
                color=color,
                linewidth=1.45 if series["color"] == "color" else 1.05,
                linestyle="-" if series["color"] == "color" else "--",
                alpha=0.95 if series["color"] == "color" else 0.62,
            )
        ax.axvspan(100, 400, color="#dff3f1", alpha=0.55, zorder=0)
        ax.axvline(0, color="#52606d", linewidth=1.2)
        ax.axvline(100, color="#8ea3b5", linewidth=0.8, linestyle=":")
        ax.axvline(400, color="#8ea3b5", linewidth=0.8, linestyle=":")
        ax.set_title(f"{panel['subject']} · {panel['channel']}", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlim(-500, 1000)
        ax.set_ylim(payload["y_min"], payload["y_max"])
        ax.set_xticks([-500, 0, 500, 1000])
        ax.grid(axis="y", color="#d9e2ec", linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color="#e45756", lw=2, label="face"),
        plt.Line2D([0], [0], color="#4c78a8", lw=2, label="object"),
        plt.Line2D([0], [0], color="#59a14f", lw=2, label="body"),
        plt.Line2D([0], [0], color="#b279a2", lw=2, label="place"),
        plt.Line2D([0], [0], color="#627d98", lw=1.5, linestyle="--", label="gray condition"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        "Task 1 signals on the 8 CSC electrodes\nBaseline-corrected mean ± SEM; color = solid, gray = dashed; shaded window = 100–400 ms",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.supxlabel("Time (ms)", fontsize=13)
    fig.supylabel("Amplitude (a.u.)", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    out = figures / f"task1_csc_signals_{CSC_COLUMN}_{VARIANT}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'''<div id="task1-csc-signals">
  <style>
    #task1-csc-signals {{
      --task1-bg: #f7fafc;
      --task1-panel: #ffffff;
      --task1-ink: #102a43;
      --task1-muted: #627d98;
      --task1-grid: #d9e2ec;
      --task1-face: #e45756;
      --task1-object: #4c78a8;
      --task1-body: #59a14f;
      --task1-place: #b279a2;
      font-family: Arial, sans-serif;
      color: var(--task1-ink);
      background: var(--task1-bg);
      width: 100%;
      max-width: 1200px;
      margin: 0 auto;
      padding: 12px;
      box-sizing: border-box;
    }}
    #task1-csc-signals .task1-head {{ margin: 0 0 10px 0; }}
    #task1-csc-signals .task1-title {{ font-size: 22px; font-weight: 700; margin-bottom: 3px; }}
    #task1-csc-signals .task1-subtitle {{ font-size: 13px; color: var(--task1-muted); }}
    #task1-csc-signals .task1-legend {{ display: flex; flex-wrap: wrap; gap: 10px 18px; font-size: 12px; margin: 8px 0 8px 0; }}
    #task1-csc-signals .task1-legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
    #task1-csc-signals .task1-swatch {{ width: 20px; height: 3px; display: inline-block; }}
    #task1-csc-signals .task1-dash {{ border-top: 2px dashed currentColor; height: 0; }}
    #task1-csc-signals svg {{ display: block; width: 100%; height: auto; background: var(--task1-panel); border: 1px solid var(--task1-grid); border-radius: 8px; }}
    #task1-csc-signals .task1-note {{ font-size: 12px; color: var(--task1-muted); margin-top: 7px; }}
  </style>
  <div class="task1-head">
    <div class="task1-title">Task 1 signals on the 8 CSC electrodes</div>
    <div class="task1-subtitle">Baseline-corrected mean ± SEM · four stimulus categories · color = solid, gray = dashed · shaded window: 100–400 ms</div>
  </div>
  <div class="task1-legend">
    <span class="task1-legend-item"><span class="task1-swatch" style="background:var(--task1-face)"></span>face</span>
    <span class="task1-legend-item"><span class="task1-swatch" style="background:var(--task1-object)"></span>object</span>
    <span class="task1-legend-item"><span class="task1-swatch" style="background:var(--task1-body)"></span>body</span>
    <span class="task1-legend-item"><span class="task1-swatch" style="background:var(--task1-place)"></span>place</span>
    <span class="task1-legend-item"><span class="task1-swatch task1-dash" style="color:#627d98"></span>gray condition</span>
  </div>
  <svg id="task1-csc-svg" viewBox="0 0 1200 920" role="img" aria-label="Eight small-multiple plots showing Task 1 color and gray evoked signals for CSC electrodes"></svg>
  <div class="task1-note">The vertical solid line marks stimulus onset (0 ms); dotted lines mark the 100–400 ms selection window. The signal scale is shared across panels.</div>
  <script>
    (() => {{
      const root = document.getElementById('task1-csc-signals');
      const svg = document.getElementById('task1-csc-svg');
      const payload = {data_json};
      const NS = 'http://www.w3.org/2000/svg';
      const colors = {{face: '#e45756', object: '#4c78a8', body: '#59a14f', place: '#b279a2'}};
      const W = 1200, H = 920, cols = 2, rows = 4;
      const gapX = 26, gapY = 22, panelW = (W - 86 - gapX) / cols, panelH = (H - 56 - gapY * (rows - 1)) / rows;
      const yMin = payload.y_min, yMax = payload.y_max;
      const xMin = -500, xMax = 1000;
      const el = (tag, attrs) => {{ const n = document.createElementNS(NS, tag); Object.entries(attrs || {{}}).forEach(([k,v]) => n.setAttribute(k, v)); return n; }};
      const addText = (parent, x, y, value, attrs={{}}) => {{ const n = el('text', {{x, y, ...attrs}}); n.textContent = value; parent.appendChild(n); }};
      const xMap = (x, left, width) => left + (x - xMin) / (xMax - xMin) * width;
      const yMap = (y, top, height) => top + (yMax - y) / (yMax - yMin) * height;
      const linePath = (xs, ys, left, top, width, height) => xs.map((x,i) => `${{i ? 'L' : 'M'}} ${{xMap(x,left,width).toFixed(2)}} ${{yMap(ys[i],top,height).toFixed(2)}}`).join(' ');
      const areaPath = (xs, mean, sem, left, top, width, height) => {{
        const upper = xs.map((x,i) => `${{i ? 'L' : 'M'}} ${{xMap(x,left,width).toFixed(2)}} ${{yMap(mean[i] + sem[i],top,height).toFixed(2)}}`).join(' ');
        const lower = xs.slice().reverse().map((x,j) => {{ const i = xs.length - 1 - j; return `L ${{xMap(x,left,width).toFixed(2)}} ${{yMap(mean[i] - sem[i],top,height).toFixed(2)}}`; }}).join(' ');
        return upper + ' ' + lower + ' Z';
      }};
      const fmt = (v) => Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
      payload.panels.forEach((panel, idx) => {{
        const col = idx % cols, row = Math.floor(idx / cols);
        const px = 38 + col * (panelW + gapX), py = 18 + row * (panelH + gapY);
        const left = px + 42, top = py + 28, width = panelW - 56, height = panelH - 54;
        svg.appendChild(el('rect', {{x:px, y:py, width:panelW, height:panelH, rx:8, fill:'#ffffff', stroke:'#d9e2ec'}}));
        addText(svg, px + 12, py + 18, `${{panel.subject}} · ${{panel.channel}}`, {{'font-size':'14', 'font-weight':'700', fill:'#102a43'}});
        [0, 100, 400].forEach((v) => {{
          const xx = xMap(v, left, width);
          svg.appendChild(el('rect', {{x:xx, y:top, width: v === 100 ? xMap(400,left,width)-xx : 0, height, fill:'#e6f4f3', opacity:v===100 ? 0.65 : 0}}));
          svg.appendChild(el('line', {{x1:xx, x2:xx, y1:top, y2:top+height, stroke:v===0 ? '#52606d' : '#8ea3b5', 'stroke-width':v===0 ? 1.5 : 1, 'stroke-dasharray':v===0 ? '' : '3 3'}}));
        }});
        [-500, 0, 500, 1000].forEach((v) => {{ const xx=xMap(v,left,width); svg.appendChild(el('line', {{x1:xx,x2:xx,y1:top+height,y2:top+height+4,stroke:'#829ab1'}})); addText(svg,xx,top+height+17,String(v),{{'font-size':'9','text-anchor':'middle',fill:'#627d98'}}); }});
        const yTicks = [yMin, (yMin+yMax)/2, yMax];
        yTicks.forEach((v) => {{ const yy=yMap(v,top,height); svg.appendChild(el('line', {{x1:left-4,x2:left,y1:yy,y2:yy,stroke:'#829ab1'}})); if (col === 0) addText(svg,left-8,yy+3,fmt(v),{{'font-size':'9','text-anchor':'end',fill:'#627d98'}}); }});
        panel.series.forEach((series) => {{
          const c = colors[series.category];
          svg.appendChild(el('path', {{d:areaPath(panel.time,series.mean,series.sem,left,top,width,height), fill:c, opacity:0.08, stroke:'none'}}));
        }});
        panel.series.forEach((series) => {{
          const c = colors[series.category];
          svg.appendChild(el('path', {{d:linePath(panel.time,series.mean,left,top,width,height), fill:'none', stroke:c, 'stroke-width':series.color==='color' ? 1.5 : 1.1, 'stroke-dasharray':series.color==='gray' ? '5 3' : '', opacity:series.color==='color' ? 0.95 : 0.6}}));
        }});
        if (col === 0) addText(svg, px + 8, py + panelH/2, 'a.u.', {{'font-size':'9', fill:'#627d98', transform:`rotate(-90 ${{px+8}} ${{py+panelH/2}})`, 'text-anchor':'middle'}});
      }});
    }})();
  </script>
</div>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("task1_csc_signals.html"),
        help="Where to write the interactive HTML panel.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Result directory; defaults to the newest final_analysis_* folder.",
    )
    parser.add_argument("--variant", default="100-400_lf30")
    parser.add_argument(
        "--csc-column",
        default="CSC",
        help="Column of the electrode_sets_and_csc CSV to use (CSC or CSC_merged).",
    )
    parser.add_argument(
        "--csc",
        type=Path,
        default=None,
        help="Explicit CSC table (electrode_sets_and_csc_*.csv).",
    )
    args = parser.parse_args()
    if args.result_dir is not None:
        RESULT_DIR = args.result_dir.resolve()
    else:
        candidates = sorted((MODULE / "result").glob("final_analysis_*"))
        if not candidates:
            raise SystemExit("No final_analysis_* result folder found")
        RESULT_DIR = candidates[-1]
    VARIANT = args.variant
    CSC_PATH = args.csc
    CSC_COLUMN = args.csc_column
    OUT = args.out_html
    OUT.mkdir(parents=True, exist_ok=True)
    payload = collect()
    html = build_html(payload)
    path = OUT / "task1-csc-signals.html"
    path.write_text(html, encoding="utf-8")
    print(path)
    print(save_png(payload))
