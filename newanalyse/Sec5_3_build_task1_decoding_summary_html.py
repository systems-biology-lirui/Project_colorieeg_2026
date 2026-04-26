import html
from pathlib import Path

import numpy as np
import pandas as pd

from newanalyse_paths import get_report_dir, project_root, result_root


BASE_PATH = project_root()
SUBJECTS = []
FEATURE_KINDS = ['erp', 'highgamma', 'lowgamma', 'tfa', 'gamma', 'gamma_multiband']
BATCH_NAME = None
PERM_TAGS = ['perm100', 'real_only']
ONLY_SIGNIFICANT = False

OUTPUT_HTML = get_report_dir(BASE_PATH, 'task1_decoding_summary') / 'task1_decoding_summary.html'
OUTPUT_DETAIL_CSV = OUTPUT_HTML.with_suffix('.details.csv')
OUTPUT_SIGNIFICANT_CSV = OUTPUT_HTML.with_suffix('.significant.csv')
OUTPUT_SUMMARY_CSV = OUTPUT_HTML.with_suffix('.summary.csv')
OUTPUT_COMPARISON_CSV = OUTPUT_HTML.with_suffix('.comparison.csv')

TASK_SCHEMES = {
    'task1_color_vs_gray_per_category': 'Paired Centered Per-Category Decoding',
    'task1_color_vs_gray_per_category_multiband': 'Paired Centered Per-Category Decoding',
    'task1_color_vs_gray_cross_category_average': 'Cross-Category Trial-Averaged Decoding',
}

T_START = -100.0
T_END = 1000.0
N_POINTS = 550
BASE_TIMES = np.linspace(T_START, T_END, N_POINTS)


def normalize_scalar(value, default=None):
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode('utf-8')
    if isinstance(scalar, np.generic):
        return scalar.item()
    return scalar


def safe_float(value, default=np.nan):
    scalar = normalize_scalar(value, default)
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return float(default)


def safe_bool(value, default=False):
    scalar = normalize_scalar(value, default)
    if isinstance(scalar, str):
        return scalar.strip().lower() in {'1', 'true', 'yes'}
    return bool(scalar)


def infer_path_context(npz_path):
    parts = npz_path.parts
    decoding_idx = parts.index('decoding')
    task_id = parts[decoding_idx + 1]
    feature_kind = parts[decoding_idx + 2]
    subject = parts[decoding_idx + 3]
    batch_parts = parts[decoding_idx + 4:-4]
    batch_name = '/'.join(batch_parts) if batch_parts else ''
    perm_tag = parts[-4]
    variant = parts[-3]
    return {
        'task_id_from_path': task_id,
        'feature_kind_from_path': feature_kind,
        'subject_from_path': subject,
        'batch_name': batch_name,
        'perm_tag': perm_tag,
        'variant': variant,
    }


def find_curve_figure(npz_path, task_id, roi_name):
    roi_dir = npz_path.parent.parent / 'roi_curves'
    candidates = []
    if task_id in {'task1_color_vs_gray_per_category', 'task1_color_vs_gray_per_category_multiband'}:
        candidates.append(roi_dir / f'{roi_name}_mean_curve.png')
    candidates.append(roi_dir / f'{roi_name}_curve.png')
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def extract_record(npz_path):
    path_info = infer_path_context(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        task_id = str(normalize_scalar(data.get('task_id'), path_info['task_id_from_path']))
        if task_id not in TASK_SCHEMES:
            return None

        subject = str(path_info['subject_from_path'])
        feature_kind = str(path_info['feature_kind_from_path'])
        if SUBJECTS and subject not in SUBJECTS:
            return None
        if FEATURE_KINDS and feature_kind not in FEATURE_KINDS:
            return None
        if BATCH_NAME is not None and path_info['batch_name'] != str(BATCH_NAME):
            return None
        if PERM_TAGS and path_info['perm_tag'] not in PERM_TAGS:
            return None

        mean_auc = np.asarray(data['mean_auc'], dtype=float)
        sig_mask = np.asarray(data['sig_indices'], dtype=bool)
        roi_name = npz_path.stem.replace('_results', '')
        stride = max(1, int(round(N_POINTS / max(mean_auc.size, 1))))
        plot_times = BASE_TIMES[::stride][:mean_auc.size]
        time_step_ms = float(np.median(np.diff(plot_times))) if plot_times.size > 1 else np.nan
        curve_figure = find_curve_figure(npz_path, task_id, roi_name)

        analysis_mode = normalize_scalar(data.get('analysis_mode'), None)
        if analysis_mode is None:
            analysis_mode = 'center' if safe_bool(data.get('use_groupeddata_pair_centering'), False) else 'raw'

        return {
            'subject': subject,
            'feature_kind': feature_kind,
            'scheme_label': TASK_SCHEMES[task_id],
            'task_id': task_id,
            'roi_name': roi_name,
            'batch_name': path_info['batch_name'],
            'perm_tag': path_info['perm_tag'],
            'variant': path_info['variant'],
            'analysis_mode': str(analysis_mode),
            'use_groupeddata_pairing': safe_bool(data.get('use_groupeddata_pairing'), False),
            'use_groupeddata_pair_centering': safe_bool(data.get('use_groupeddata_pair_centering'), False),
            'groupeddata_mat': str(normalize_scalar(data.get('groupeddata_mat'), '')),
            'peak_auc_all': float(np.nanmax(mean_auc)) if mean_auc.size else np.nan,
            'peak_auc_sig': float(np.nanmax(mean_auc[sig_mask])) if np.any(sig_mask) else np.nan,
            'earliest_sig_ms': safe_float(data.get('latency_earliest')),
            'half_height_ms': safe_float(data.get('latency_half_height')),
            'peak_sig_ms': safe_float(data.get('latency_peak')),
            'n_sig_points': int(np.sum(sig_mask)),
            'sig_duration_ms': float(np.sum(sig_mask) * time_step_ms) if np.any(sig_mask) and np.isfinite(time_step_ms) else 0.0,
            'is_significant': bool(np.any(sig_mask)),
            'n_real': int(normalize_scalar(data.get('n_real'), 0) or 0),
            'n_repeats_real': int(normalize_scalar(data.get('n_repeats_real'), 0) or 0),
            'n_repeats_perm': int(normalize_scalar(data.get('n_repeats_perm'), 0) or 0),
            'n_perm': int(normalize_scalar(data.get('n_perm'), 0) or 0),
            'npz_path': str(npz_path),
            'curve_figure': str(curve_figure) if curve_figure else '',
        }


def collect_records():
    decoding_root = result_root(BASE_PATH) / 'decoding'
    records = []
    for npz_path in sorted(decoding_root.glob('**/computed_results/*_results.npz')):
        record = extract_record(npz_path)
        if record is None:
            continue
        if ONLY_SIGNIFICANT and not record['is_significant']:
            continue
        records.append(record)
    return pd.DataFrame(records)


def build_summary_frames(all_df):
    if all_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_df = (
        all_df.groupby(['subject', 'feature_kind', 'scheme_label', 'analysis_mode', 'perm_tag'], as_index=False)
        .agg(
            n_rois=('roi_name', 'count'),
            n_significant_rois=('is_significant', 'sum'),
            best_peak_auc=('peak_auc_all', 'max'),
            earliest_sig_ms=('earliest_sig_ms', 'min'),
        )
        .sort_values(['subject', 'feature_kind', 'scheme_label'])
        .reset_index(drop=True)
    )

    significant_df = all_df[all_df['is_significant']].copy().reset_index(drop=True)
    if significant_df.empty:
        comparison_df = pd.DataFrame()
    else:
        comparison_df = (
            significant_df.groupby(['subject', 'scheme_label', 'roi_name', 'feature_kind'], as_index=False)
            .agg(
                peak_auc_all=('peak_auc_all', 'max'),
                peak_auc_sig=('peak_auc_sig', 'max'),
                earliest_sig_ms=('earliest_sig_ms', 'min'),
                peak_sig_ms=('peak_sig_ms', 'min'),
                sig_duration_ms=('sig_duration_ms', 'max'),
            )
            .sort_values(['subject', 'scheme_label', 'roi_name', 'feature_kind'])
            .reset_index(drop=True)
        )
    return summary_df, significant_df, comparison_df


def make_relative_path(path_text):
    if not path_text:
        return ''
    return str(Path(path_text).resolve().relative_to(BASE_PATH.resolve())) if Path(path_text).resolve().is_relative_to(BASE_PATH.resolve()) else str(Path(path_text).resolve())


def dataframe_to_html_table(df, link_cols=None):
    if df.empty:
        return '<p>No rows.</p>'

    link_cols = set(link_cols or [])
    headers = ''.join(f'<th>{html.escape(str(col))}</th>' for col in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            text = '' if pd.isna(row[col]) else str(row[col])
            if col in link_cols and text:
                label = html.escape(Path(text).name)
                href = html.escape(text)
                cells.append(f'<td><a href="{href}">{label}</a></td>')
            else:
                cells.append(f'<td>{html.escape(text)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def write_outputs(all_df, summary_df, significant_df, comparison_df):
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    all_df.to_csv(OUTPUT_DETAIL_CSV, index=False)
    significant_df.to_csv(OUTPUT_SIGNIFICANT_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    render_all = all_df.copy()
    render_sig = significant_df.copy()
    for frame in (render_all, render_sig):
        if not frame.empty:
            frame['npz_path'] = frame['npz_path'].map(make_relative_path)
            frame['curve_figure'] = frame['curve_figure'].map(make_relative_path)

    html_text = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <title>Task1 Decoding Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #f7f7f9; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef1f6; }}
    .section {{ margin-top: 28px; }}
    a {{ color: #005a9c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Task1 Color/Gray Decoding Summary</h1>
  <div class=\"meta\">Subjects: {html.escape(', '.join(SUBJECTS) if SUBJECTS else 'all')} | Features: {html.escape(', '.join(FEATURE_KINDS))} | Batch filter: {html.escape(str(BATCH_NAME))}</div>

  <div class=\"section\">
    <h2>Run Summary</h2>
    {dataframe_to_html_table(summary_df)}
  </div>

  <div class=\"section\">
    <h2>Significant ROIs</h2>
    {dataframe_to_html_table(render_sig, link_cols=['curve_figure', 'npz_path'])}
  </div>

  <div class=\"section\">
    <h2>Cross-Feature Comparison</h2>
    {dataframe_to_html_table(comparison_df)}
  </div>

  <div class=\"section\">
    <h2>All ROI Results</h2>
    {dataframe_to_html_table(render_all, link_cols=['curve_figure', 'npz_path'])}
  </div>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html_text, encoding='utf-8')

    print(f'Summary HTML saved: {OUTPUT_HTML}')
    if OUTPUT_DETAIL_CSV.exists():
        print(f'Detail CSV saved: {OUTPUT_DETAIL_CSV}')
    if OUTPUT_SIGNIFICANT_CSV.exists():
        print(f'Significant CSV saved: {OUTPUT_SIGNIFICANT_CSV}')
    if OUTPUT_SUMMARY_CSV.exists():
        print(f'Summary CSV saved: {OUTPUT_SUMMARY_CSV}')
    if OUTPUT_COMPARISON_CSV.exists():
        print(f'Comparison CSV saved: {OUTPUT_COMPARISON_CSV}')


def main():
    all_df = collect_records().sort_values(['subject', 'feature_kind', 'scheme_label', 'roi_name']).reset_index(drop=True)
    summary_df, significant_df, comparison_df = build_summary_frames(all_df)
    write_outputs(all_df, summary_df, significant_df, comparison_df)


if __name__ == '__main__':
    main()
