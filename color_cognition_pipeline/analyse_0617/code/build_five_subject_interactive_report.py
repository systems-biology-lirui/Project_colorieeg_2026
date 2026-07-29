"""Build a single interactive HTML report from the existing 0617 outputs.

No subject-level decoding is performed here. Existing test001--003 tables are
read from analyse_0617/doc and test005--006 tables from run_5subjects_original.
Only descriptive five-subject means are calculated from saved accuracy curves.
"""
from pathlib import Path
import html
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

ROOT = Path('/home/lirui/liulab_project/ieeg/Project_colorieeg_2026')
ANALYSE = ROOT / 'color_cognition_pipeline' / 'analyse_0617'
OLD = ANALYSE / 'doc'
NEW = ANALYSE / 'run_5subjects_original' / 'doc'
OUT = ANALYSE / 'run_5subjects_original' / 'report'
FINAL_OUT = ANALYSE / 'result' / 'final_report'
SUBJECTS = ['test001', 'test002', 'test003', 'test005', 'test006']
COLORS = {'test001':'#E69F00', 'test002':'#009E73', 'test003':'#0072B2',
          'test005':'#CC79A7', 'test006':'#D55E00'}


def read_combined(name):
    new = NEW / name
    old = OLD / name
    target = new if new.exists() else old
    df = pd.read_csv(target)
    acc_cols = [c for c in df.columns if re.fullmatch(r'test\d+_Acc', c)]
    if acc_cols and 'Group_Mean_Acc' not in df.columns:
        df['Group_Mean_Acc'] = df[acc_cols].mean(axis=1)
    return df


def fig_counts(summary):
    rows = []
    for s, d in summary.groupby('Subject'):
        rows += [{'Subject':s, 'Evidence':'ERP selected', 'Count':int(d['ERP_Selected'].fillna(False).astype(bool).sum())},
                 {'Subject':s, 'Evidence':'HG selected', 'Count':int(d['HG_Selected'].fillna(False).astype(bool).sum())}]
    return px.bar(pd.DataFrame(rows), x='Subject', y='Count', color='Evidence', barmode='group',
                  color_discrete_map={'ERP selected':'#0072B2','HG selected':'#D55E00'},
                  title='Color-selective electrode counts', template='plotly_white')


def fig_mni(summary):
    d = summary.copy()
    d['ERP_Selected'] = d['ERP_Selected'].fillna(False).astype(bool)
    d['HG_Selected'] = d['HG_Selected'].fillna(False).astype(bool)
    d['Evidence'] = np.select([d.ERP_Selected & d.HG_Selected, d.ERP_Selected, d.HG_Selected],
                               ['ERP + HG','ERP only','HG only'], default='Not selected')
    d['label'] = d.Subject + ' · ' + d.Electrode + ' · ' + d.Evidence
    return px.scatter_3d(d, x='MNI_X', y='MNI_Y', z='MNI_Z', color='Subject', symbol='Evidence',
                         hover_name='label', hover_data=['AAL3','AAL3_ROI'], size_max=8,
                         color_discrete_map=COLORS, title='All localized electrodes and functional evidence',
                         template='plotly_white')


def fig_decoding(df, modality, scheme):
    fig = go.Figure()
    for s in SUBJECTS:
        c = f'{s}_Acc'
        if c in df:
            fig.add_trace(go.Scatter(x=df.Time_ms, y=df[c], mode='lines', name=s,
                                     line={'color':COLORS[s], 'width':1.3}, opacity=.7))
    fig.add_trace(go.Scatter(x=df.Time_ms, y=df.Group_Mean_Acc, mode='lines', name='Five-subject mean',
                             line={'color':'#111111','width':3.2}))
    fig.add_hline(y=.5, line_dash='dot', line_color='#777777', annotation_text='chance = 0.5')
    fig.update_layout(title=f'{modality} · {scheme}', xaxis_title='Time (ms)', yaxis_title='Accuracy',
                      xaxis_range=[-200,800], yaxis_range=[.35,.8], template='plotly_white',
                      hovermode='x unified', legend_title='Trace')
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(NEW / 'select_channel_summary.csv')
    summary = summary[summary.Subject.isin(SUBJECTS)].copy()
    figures = [('counts', fig_counts(summary)), ('mni', fig_mni(summary))]
    for modality in ('ERP','HG'):
        for scheme in ('strategy4','union','memorysig'):
            name = f'decoding_data_{modality.lower()}_{scheme}.csv'
            p = NEW / name
            if p.exists():
                figures.append((f'{modality.lower()}_{scheme}', fig_decoding(read_combined(name), modality, scheme)))
    blocks = []
    for i, (key, fig) in enumerate(figures):
        blocks.append(f'<section><h2>{html.escape(key.replace("_", " ").title())}</h2>' +
                      to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False,
                              config={'displaylogo':False, 'responsive':True}) + '</section>')
    erp_old = pd.read_csv(OLD/'select_channel_memory_decoding_estp_erp.csv')
    erp_new = pd.read_csv(NEW/'select_channel_memory_decoding_estp_erp.csv')
    hg_old = pd.read_csv(OLD/'select_channel_memory_decoding_estp_hg.csv')
    hg_new = pd.read_csv(NEW/'select_channel_memory_decoding_estp_hg.csv')
    estp = pd.concat([erp_old.assign(Modality='ERP'), erp_new.assign(Modality='ERP'),
                      hg_old.assign(Modality='HG'), hg_new.assign(Modality='HG')], ignore_index=True)
    estp['Subject'] = pd.Categorical(estp.Subject, SUBJECTS, ordered=True)
    estp_html = estp.sort_values(['Modality','Subject']).to_html(index=False, classes='data', border=0)
    summary_html = summary.to_html(index=False, classes='data', border=0)
    report = f'''<!doctype html><html><head><meta charset="utf-8"><title>Color iEEG · five-subject 0617 report</title>
<style>body{{font-family:Inter,Arial,sans-serif;background:#f5f7fa;color:#17202a;margin:0}}main{{max-width:1450px;margin:auto;padding:32px}}h1{{font-size:32px;margin-bottom:4px}}h2{{margin-top:28px;text-transform:capitalize}}.subtitle{{color:#52606d}}section{{background:#fff;border-radius:14px;padding:18px;margin:20px 0;box-shadow:0 3px 18px #17202a12}}.data{{width:100%;border-collapse:collapse;font-size:12px;display:block;overflow:auto;max-height:390px}}.data th,.data td{{padding:6px 9px;border-bottom:1px solid #e8edf2;white-space:nowrap}}.data th{{position:sticky;top:0;background:#eef2f6;text-align:left}}.note{{background:#fff8e6;border-left:4px solid #e69f00;padding:12px 16px;border-radius:5px}}</style></head><body><main>
<h1>Color iEEG · analyse_0617 unified report</h1><p class="subtitle">Interactive five-subject summary · generated from existing legacy outputs</p>
<div class="note"><b>Inclusion:</b> test001, test002, test003, test005, test006 (N=5). <b>test004 is excluded</b> because electrode localization failed. This report does not rerun test001–003; curves are combined from saved subject-level results. Five-subject curves are descriptive means of saved accuracies.</div>
{''.join(blocks)}
<section><h2>Electrode manifest</h2>{summary_html}</section><section><h2>Memory decoding ESTP</h2>{estp_html}</section>
</main></body></html>'''
    path = OUT / 'color_ieeg_0617_five_subject_interactive_report.html'
    path.write_text(report, encoding='utf-8')
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    final_path = FINAL_OUT / path.name
    final_path.write_text(report, encoding='utf-8')
    print(path)
    print(final_path)


if __name__ == '__main__':
    main()
