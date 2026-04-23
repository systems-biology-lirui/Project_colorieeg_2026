import html
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
from newanalyse_paths import get_report_dir, project_root

# =========================
# 绘制ROI下不同电极的时间信号差异-批处理
# =========================
BASE_PATH = project_root()
SCRIPT_PATH = BASE_PATH / "newanalyse" / "Sec3_s1_roi_electrode_condition_erp_stats.py"

# =========================
# User Config
# =========================
SUBJECT = "test001"
TASK = "task1"
FEATURE_KINDS = ["erp"]

COMPARISONS = [
    {"id": "1_vs_2", "group_a": [1], "group_b": [2], "label_a": "Cond 1", "label_b": "Cond 2"},
    {"id": "3_vs_4", "group_a": [3], "group_b": [4], "label_a": "Cond 3", "label_b": "Cond 4"},
    {"id": "5_vs_6", "group_a": [5], "group_b": [6], "label_a": "Cond 5", "label_b": "Cond 6"},
    {"id": "7_vs_8", "group_a": [7], "group_b": [8], "label_a": "Cond 7", "label_b": "Cond 8"},
    {"id": "137_vs_248", "group_a": [1, 3, 7], "group_b": [2, 4, 8], "label_a": "Cond 1,3,7", "label_b": "Cond 2,4,8"},
]

ALPHA = 0.05
FS = 500.0
TMIN_MS = -100.0
BASELINE_START_MS = -100.0
BASELINE_END_MS = 0.0
ROI = None
ROI_PATTERN = "Color_with*.mat"
DPI = 220

OUTPUT_ROOT = get_report_dir(BASE_PATH, 'roi_electrode_condition_stats_batch_colorpatch1') / SUBJECT


def load_single_run_module():
    spec = importlib.util.spec_from_file_location("roi_condition_stats_single", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one_comparison(feature_kind, comparison):
    module = load_single_run_module()

    module.SUBJECT = SUBJECT
    module.FEATURE_KIND = feature_kind
    module.TASK = TASK
    module.GROUP_A = comparison["group_a"]
    module.GROUP_B = comparison["group_b"]
    module.GROUP_A_LABEL = comparison["label_a"]
    module.GROUP_B_LABEL = comparison["label_b"]
    module.ALPHA = ALPHA
    module.FS = FS
    module.TMIN_MS = TMIN_MS
    module.BASELINE_START_MS = BASELINE_START_MS
    module.BASELINE_END_MS = BASELINE_END_MS
    module.ROI = ROI
    module.ROI_PATTERN = ROI_PATTERN
    module.DPI = DPI
    module.OUTPUT_DIR = OUTPUT_ROOT / feature_kind / comparison["id"]

    print(f"Running {feature_kind} | {comparison['id']}")
    module.main()
    return module.OUTPUT_DIR


def collect_run_summaries(run_dir, feature_kind, comparison):
    roi_summary_path = run_dir / "roi_summary.csv"
    if not roi_summary_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    roi_df = pd.read_csv(roi_summary_path)
    roi_df["subject"] = SUBJECT
    roi_df["task"] = TASK
    roi_df["feature_kind"] = feature_kind
    roi_df["comparison_id"] = comparison["id"]
    roi_df["group_a"] = ",".join(str(v) for v in comparison["group_a"])
    roi_df["group_b"] = ",".join(str(v) for v in comparison["group_b"])
    roi_df["group_a_label"] = comparison["label_a"]
    roi_df["group_b_label"] = comparison["label_b"]
    roi_df["roi_summary_csv"] = str(roi_summary_path)

    channel_rows = []
    for _, row in roi_df.iterrows():
        stats_path = Path(row["stats_csv"])
        if not stats_path.exists():
            continue
        stats_df = pd.read_csv(stats_path)
        stats_df["subject"] = SUBJECT
        stats_df["task"] = TASK
        stats_df["feature_kind"] = feature_kind
        stats_df["comparison_id"] = comparison["id"]
        stats_df["group_a"] = ",".join(str(v) for v in comparison["group_a"])
        stats_df["group_b"] = ",".join(str(v) for v in comparison["group_b"])
        stats_df["group_a_label"] = comparison["label_a"]
        stats_df["group_b_label"] = comparison["label_b"]
        stats_df["figure"] = row["figure"]
        stats_df["stats_csv"] = row["stats_csv"]
        channel_rows.append(stats_df)

    channel_df = pd.concat(channel_rows, ignore_index=True) if channel_rows else pd.DataFrame()
    return roi_df, channel_df


def build_aggregates(roi_df, channel_df):
    if roi_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    roi_agg = (
        roi_df.groupby(["subject", "task", "feature_kind", "comparison_id", "group_a", "group_b"], as_index=False)
        .agg(
            n_roi=("roi", "count"),
            total_channels=("n_channels", "sum"),
            total_sig_channels=("n_channels_with_sig", "sum"),
            max_sig_points_any_roi=("max_sig_points_one_channel", "max"),
        )
    )

    if channel_df.empty:
        channel_agg = pd.DataFrame()
    else:
        channel_agg = (
            channel_df.groupby(
                ["subject", "task", "feature_kind", "comparison_id", "group_a", "group_b", "roi"],
                as_index=False,
            )
            .agg(
                n_electrodes=("channel", "count"),
                n_sig_electrodes=("n_sig_points", lambda x: int((x > 0).sum())),
                max_sig_points=("n_sig_points", "max"),
                min_p=("min_p", "min"),
            )
        )
    return roi_agg, channel_agg


def to_rel(path_str):
    path = Path(path_str)
    return path.relative_to(OUTPUT_ROOT.parent.parent.parent.parent)


def write_html_report(roi_df, roi_agg, channel_agg, out_path):
    if roi_df.empty:
        out_path.write_text("<html><body><p>No results found.</p></body></html>", encoding="utf-8")
        return

    overview_records = roi_agg.to_dict(orient="records")
    roi_records = []
    for _, row in roi_df.sort_values(["feature_kind", "comparison_id", "roi"]).iterrows():
        rec = row.to_dict()
        rec["figure_rel"] = str(Path(rec["figure"]).relative_to(out_path.parent))
        rec["stats_csv_rel"] = str(Path(rec["stats_csv"]).relative_to(out_path.parent))
        roi_records.append(rec)

    channel_records = channel_agg.to_dict(orient="records") if not channel_agg.empty else []

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ROI Condition Stats Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #f7f7f9;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .meta {{
      margin-bottom: 20px;
      color: #555;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 24px;
      background: #fff;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #eef1f6;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .card h3 {{
      margin: 0 0 8px 0;
      font-size: 16px;
    }}
    .card p {{
      margin: 4px 0;
      font-size: 13px;
      color: #444;
    }}
    .thumb {{
      width: 100%;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin-top: 10px;
      background: #fff;
    }}
    .small {{
      font-size: 12px;
      color: #666;
    }}
  </style>
</head>
<body>
  <h1>ROI Electrode Condition Stats</h1>
  <div class="meta">
    Subject: {html.escape(SUBJECT)} |
    Task: {html.escape(TASK)} |
    Features: {html.escape(", ".join(FEATURE_KINDS))} |
    Baseline: {BASELINE_START_MS:g} to {BASELINE_END_MS:g} ms
  </div>

  <div class="controls">
    <label>Feature
      <select id="featureFilter">
        <option value="">All</option>
        {''.join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in FEATURE_KINDS)}
      </select>
    </label>
    <label>Comparison
      <select id="comparisonFilter">
        <option value="">All</option>
        {''.join(f'<option value="{html.escape(c["id"])}">{html.escape(c["id"])}</option>' for c in COMPARISONS)}
      </select>
    </label>
    <label>ROI
      <input id="roiFilter" type="text" placeholder="Type ROI name">
    </label>
  </div>

  <h2>Overview</h2>
  <table id="overviewTable">
    <thead>
      <tr>
        <th>Feature</th>
        <th>Comparison</th>
        <th>Group A</th>
        <th>Group B</th>
        <th>ROI Count</th>
        <th>Total Channels</th>
        <th>Significant Channels</th>
        <th>Max Sig Points</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <h2>ROI Figures</h2>
  <div id="cardGrid" class="card-grid"></div>

  <h2>ROI Aggregate Table</h2>
  <table id="roiTable">
    <thead>
      <tr>
        <th>Feature</th>
        <th>Comparison</th>
        <th>ROI</th>
        <th>Channels</th>
        <th>Sig Channels</th>
        <th>Max Sig Points</th>
        <th>Figure</th>
        <th>Stats CSV</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <script>
    const overviewData = {json.dumps(overview_records, ensure_ascii=False)};
    const roiData = {json.dumps(roi_records, ensure_ascii=False)};
    const channelData = {json.dumps(channel_records, ensure_ascii=False)};

    function q(id) {{
      return document.getElementById(id);
    }}

    function filteredRoiData() {{
      const feature = q('featureFilter').value.trim();
      const comparison = q('comparisonFilter').value.trim();
      const roiText = q('roiFilter').value.trim().toLowerCase();
      return roiData.filter(row =>
        (!feature || row.feature_kind === feature) &&
        (!comparison || row.comparison_id === comparison) &&
        (!roiText || row.roi.toLowerCase().includes(roiText))
      );
    }}

    function renderOverview(filteredRows) {{
      const allowed = new Set(filteredRows.map(r => `${{r.feature_kind}}|${{r.comparison_id}}|${{r.group_a}}|${{r.group_b}}`));
      const rows = overviewData.filter(r => allowed.size === 0 || allowed.has(`${{r.feature_kind}}|${{r.comparison_id}}|${{r.group_a}}|${{r.group_b}}`));
      q('overviewTable').querySelector('tbody').innerHTML = rows.map(r => `
        <tr>
          <td>${{r.feature_kind}}</td>
          <td>${{r.comparison_id}}</td>
          <td>${{r.group_a}}</td>
          <td>${{r.group_b}}</td>
          <td>${{r.n_roi}}</td>
          <td>${{r.total_channels}}</td>
          <td>${{r.total_sig_channels}}</td>
          <td>${{r.max_sig_points_any_roi}}</td>
        </tr>
      `).join('');
    }}

    function renderRoiTable(filteredRows) {{
      q('roiTable').querySelector('tbody').innerHTML = filteredRows.map(r => `
        <tr>
          <td>${{r.feature_kind}}</td>
          <td>${{r.comparison_id}}</td>
          <td>${{r.roi}}</td>
          <td>${{r.n_channels}}</td>
          <td>${{r.n_channels_with_sig}}</td>
          <td>${{r.max_sig_points_one_channel}}</td>
          <td><a href="${{r.figure_rel}}" target="_blank">figure</a></td>
          <td><a href="${{r.stats_csv_rel}}" target="_blank">csv</a></td>
        </tr>
      `).join('');
    }}

    function renderCards(filteredRows) {{
      q('cardGrid').innerHTML = filteredRows.map(r => `
        <div class="card">
          <h3>${{r.roi}}</h3>
          <p><strong>${{r.feature_kind}}</strong> | ${{r.comparison_id}}</p>
          <p>sig channels: ${{r.n_channels_with_sig}} / ${{r.n_channels}}</p>
          <p class="small">
            group A: ${{r.group_a_label}} (${{r.group_a}})
            <br>
            group B: ${{r.group_b_label}} (${{r.group_b}})
          </p>
          <p><a href="${{r.figure_rel}}" target="_blank">Open figure</a> |
             <a href="${{r.stats_csv_rel}}" target="_blank">Open csv</a></p>
          <img class="thumb" src="${{r.figure_rel}}" alt="${{r.roi}}">
        </div>
      `).join('');
    }}

    function renderAll() {{
      const filteredRows = filteredRoiData();
      renderOverview(filteredRows);
      renderRoiTable(filteredRows);
      renderCards(filteredRows);
    }}

    q('featureFilter').addEventListener('change', renderAll);
    q('comparisonFilter').addEventListener('change', renderAll);
    q('roiFilter').addEventListener('input', renderAll);
    renderAll();
  </script>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_roi_dfs = []
    all_channel_dfs = []
    run_rows = []

    for feature_kind in FEATURE_KINDS:
        for comparison in COMPARISONS:
            run_dir = run_one_comparison(feature_kind, comparison)
            roi_df, channel_df = collect_run_summaries(run_dir, feature_kind, comparison)
            if not roi_df.empty:
                all_roi_dfs.append(roi_df)
                run_rows.append(
                    {
                        "subject": SUBJECT,
                        "task": TASK,
                        "feature_kind": feature_kind,
                        "comparison_id": comparison["id"],
                        "output_dir": str(run_dir),
                        "roi_summary_csv": str(run_dir / "roi_summary.csv"),
                        "n_roi": int(len(roi_df)),
                    }
                )
            if not channel_df.empty:
                all_channel_dfs.append(channel_df)

    run_df = pd.DataFrame(run_rows)
    roi_df = pd.concat(all_roi_dfs, ignore_index=True) if all_roi_dfs else pd.DataFrame()
    channel_df = pd.concat(all_channel_dfs, ignore_index=True) if all_channel_dfs else pd.DataFrame()
    roi_agg, channel_agg = build_aggregates(roi_df, channel_df)

    run_csv = OUTPUT_ROOT / "batch_runs.csv"
    roi_csv = OUTPUT_ROOT / "batch_roi_summary.csv"
    channel_csv = OUTPUT_ROOT / "batch_channel_summary.csv"
    overview_csv = OUTPUT_ROOT / "batch_overview_summary.csv"
    roi_agg_csv = OUTPUT_ROOT / "batch_roi_aggregate.csv"
    html_path = OUTPUT_ROOT / "report.html"

    run_df.to_csv(run_csv, index=False)
    roi_df.to_csv(roi_csv, index=False)
    channel_df.to_csv(channel_csv, index=False)
    roi_agg.to_csv(overview_csv, index=False)
    channel_agg.to_csv(roi_agg_csv, index=False)
    write_html_report(roi_df, roi_agg, channel_agg, html_path)

    print(f"Saved runs: {run_csv}")
    print(f"Saved ROI summary: {roi_csv}")
    print(f"Saved channel summary: {channel_csv}")
    print(f"Saved overview summary: {overview_csv}")
    print(f"Saved ROI aggregate: {roi_agg_csv}")
    print(f"Saved HTML report: {html_path}")


if __name__ == "__main__":
  _script_start_time = time.time()
  try:
    main()
  finally:
    print(f"Total runtime: {time.time() - _script_start_time:.2f} s")
