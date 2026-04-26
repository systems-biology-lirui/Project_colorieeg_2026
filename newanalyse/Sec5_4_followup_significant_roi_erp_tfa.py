import html
from pathlib import Path

import pandas as pd

from newanalyse_paths import get_report_dir, project_root

import Sec3_s1_roi_electrode_condition_erp_stats as erp_followup
import Sec3_s2_roi_condition_tfa as tfa_followup


BASE_PATH = project_root()
TASK = 'task1'
GROUP_A = [1, 3, 5, 7]
GROUP_B = [2, 4, 6, 8]
GROUP_A_LABEL = 'Color'
GROUP_B_LABEL = 'Gray'
SUBJECTS = []
SIGNIFICANT_CSV = get_report_dir(BASE_PATH, 'task1_decoding_summary') / 'task1_decoding_summary.significant.csv'
OUTPUT_ROOT = get_report_dir(BASE_PATH, 'task1_significant_roi_followup')
OUTPUT_HTML = OUTPUT_ROOT / 'task1_significant_roi_followup.html'
OUTPUT_CSV = OUTPUT_ROOT / 'task1_significant_roi_followup.csv'


def comparison_id():
    left = '-'.join(str(v) for v in GROUP_A)
    right = '-'.join(str(v) for v in GROUP_B)
    return f'{TASK}_A_{left}_vs_B_{right}'


def make_relative_path(path_text):
    if not path_text:
        return ''
    path = Path(path_text).resolve()
    if path.is_relative_to(OUTPUT_HTML.parent.resolve()):
        return str(path.relative_to(OUTPUT_HTML.parent.resolve()))
    if path.is_relative_to(BASE_PATH.resolve()):
        return str(path.relative_to(BASE_PATH.resolve()))
    return str(path)


def load_targets():
    if not SIGNIFICANT_CSV.is_file():
        raise FileNotFoundError(f'Significant CSV not found: {SIGNIFICANT_CSV}')

    df = pd.read_csv(SIGNIFICANT_CSV)
    if df.empty:
        return pd.DataFrame(columns=['subject', 'roi_name', 'source_features', 'source_schemes', 'best_peak_auc'])

    if SUBJECTS:
        df = df[df['subject'].astype(str).isin([str(item) for item in SUBJECTS])].copy()

    if df.empty:
        return pd.DataFrame(columns=['subject', 'roi_name', 'source_features', 'source_schemes', 'best_peak_auc'])

    grouped = (
        df.groupby(['subject', 'roi_name'], as_index=False)
        .agg(
            source_features=('feature_kind', lambda values: ';'.join(sorted({str(v) for v in values if pd.notna(v)}))),
            source_schemes=('scheme_label', lambda values: ';'.join(sorted({str(v) for v in values if pd.notna(v)}))),
            best_peak_auc=('peak_auc_all', 'max'),
        )
        .sort_values(['subject', 'roi_name'])
        .reset_index(drop=True)
    )
    return grouped


def run_erp(subject, roi_name):
    output_dir = OUTPUT_ROOT / 'erp' / subject / roi_name
    erp_followup.SUBJECT = subject
    erp_followup.FEATURE_KIND = 'erp'
    erp_followup.TASK = TASK
    erp_followup.GROUP_A = list(GROUP_A)
    erp_followup.GROUP_B = list(GROUP_B)
    erp_followup.GROUP_A_LABEL = GROUP_A_LABEL
    erp_followup.GROUP_B_LABEL = GROUP_B_LABEL
    erp_followup.ROI = roi_name
    erp_followup.ROI_PATTERN = '*.mat'
    erp_followup.OUTPUT_DIR = output_dir
    erp_followup.main()
    return {
        'erp_figure': str(output_dir / 'figures' / f'{roi_name}.png'),
        'erp_stats_csv': str(output_dir / 'stats' / f'{roi_name}.csv'),
    }


def run_tfa(subject, roi_name):
    output_root = OUTPUT_ROOT / 'tfa' / subject / roi_name
    run_id = comparison_id()
    tfa_followup.SUBJECT = subject
    tfa_followup.TASK = TASK
    tfa_followup.GROUP_A = list(GROUP_A)
    tfa_followup.GROUP_B = list(GROUP_B)
    tfa_followup.GROUP_A_LABEL = GROUP_A_LABEL
    tfa_followup.GROUP_B_LABEL = GROUP_B_LABEL
    tfa_followup.ROI = roi_name
    tfa_followup.ROI_PATTERN = '*.mat'
    tfa_followup.OUTPUT_ROOT = output_root
    tfa_followup.main()
    base_dir = output_root / run_id
    return {
        'tfa_panel_figure': str(base_dir / 'figures' / 'roi_panel' / f'{roi_name}.png'),
        'tfa_diff_figure': str(base_dir / 'figures' / 'roi_diff' / f'{roi_name}.png'),
        'tfa_mat': str(base_dir / 'mat' / f'{roi_name}.mat'),
    }


def run_followup(targets):
    rows = []
    for row in targets.itertuples(index=False):
        result = {
            'subject': str(row.subject),
            'roi_name': str(row.roi_name),
            'source_features': str(row.source_features),
            'source_schemes': str(row.source_schemes),
            'best_peak_auc': float(row.best_peak_auc),
            'erp_status': 'not-run',
            'erp_figure': '',
            'erp_stats_csv': '',
            'tfa_status': 'not-run',
            'tfa_panel_figure': '',
            'tfa_diff_figure': '',
            'tfa_mat': '',
        }

        try:
            result.update(run_erp(result['subject'], result['roi_name']))
            result['erp_status'] = 'ok'
        except Exception as exc:
            result['erp_status'] = f'error: {type(exc).__name__}: {exc}'

        try:
            result.update(run_tfa(result['subject'], result['roi_name']))
            result['tfa_status'] = 'ok'
        except Exception as exc:
            result['tfa_status'] = f'error: {type(exc).__name__}: {exc}'

        rows.append(result)

    return pd.DataFrame(rows)


def dataframe_to_html_table(df, link_cols=None):
    if df.empty:
        return '<p>No rows.</p>'

    link_cols = set(link_cols or [])
    headers = ''.join(f'<th>{html.escape(str(col))}</th>' for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = '' if pd.isna(row[col]) else str(row[col])
            if col in link_cols and value:
                cells.append(
                    f'<td><a href="{html.escape(value)}">{html.escape(Path(value).name)}</a></td>'
                )
            else:
                cells.append(f'<td>{html.escape(value)}</td>')
        body_rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def write_outputs(result_df):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    render_df = result_df.copy()
    for col in ['erp_figure', 'erp_stats_csv', 'tfa_panel_figure', 'tfa_diff_figure', 'tfa_mat']:
        if col in render_df.columns:
            render_df[col] = render_df[col].map(make_relative_path)

    html_text = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <title>Task1 Significant ROI Follow-up</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #f7f7f9; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef1f6; }}
    a {{ color: #005a9c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Task1 Significant ROI ERP/TFA Follow-up</h1>
  <div class=\"meta\">Input: {html.escape(str(SIGNIFICANT_CSV))} | Task: {html.escape(TASK)} | Group A: {html.escape(str(GROUP_A))} | Group B: {html.escape(str(GROUP_B))}</div>
  <h2>ROI Follow-up Results</h2>
  {dataframe_to_html_table(render_df, link_cols=['erp_figure', 'erp_stats_csv', 'tfa_panel_figure', 'tfa_diff_figure', 'tfa_mat'])}
</body>
</html>
"""
    OUTPUT_HTML.write_text(html_text, encoding='utf-8')
    print(f'Follow-up HTML saved: {OUTPUT_HTML}')
    print(f'Follow-up CSV saved: {OUTPUT_CSV}')


def main():
    targets = load_targets()
    result_df = run_followup(targets)
    write_outputs(result_df)


if __name__ == '__main__':
    main()
