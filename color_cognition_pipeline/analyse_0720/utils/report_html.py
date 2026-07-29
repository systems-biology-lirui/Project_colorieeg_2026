"""Generate a portable HTML work report from completed all-channel results."""
from __future__ import annotations

from html import escape
from pathlib import Path
import pandas as pd
import config


def _table(path, columns=None, n=20):
    if not path.exists():
        return '<p class="muted">结果尚未生成。</p>'
    d = pd.read_csv(path)
    if columns:
        d = d[[c for c in columns if c in d.columns]]
    return d.head(n).to_html(index=False, classes="data", border=0)


def generate():
    root = config.ALL_RESULT_ROOT
    out = root / "work_report.html"
    fig = root / "group" / "figures"
    img = lambda p, alt: f'<figure><img src="{escape(p)}" alt="{escape(alt)}"><figcaption>{escape(alt)}</figcaption></figure>'
    sections = []
    sections.append("<h1>Color iEEG 全通道分析工作报告</h1><p class='muted'>自动生成；test004 因定位失败排除。筛选使用 100 次 permutation，未进行被试内 FDR。</p>")
    sections.append("<h2>分析流程</h2><p>全通道预处理 → Task1 color–gray 功能筛选 → 单被试 decoding → 被试平均汇总 → 虚拟被试与空间分组分析。</p>")
    summary = root / "group" / "tables" / "color_select_summary.csv"
    sections.append("<h2>Color-select 电极</h2>" + _table(summary))
    sections.append(img("group/figures/nilearn_all_localized_electrodes.png", "Nilearn：所有已定位电极的 MNI 叠加图"))
    sections.append(img("group/figures/nilearn_color_select_electrodes.png", "Nilearn：color-select 电极的 MNI 叠加图"))
    sections.append(img("group/figures/subject_accuracy_mean_union.png", "单被试准确率平均及 95% 区间"))
    sections.append(img("group/figures/color_select_native_mni.png", "功能筛选电极的 MNI 分布"))
    sections.append("<h2>被试层面 decoding 平均</h2>" + _table(root / "group" / "tables" / "subject_accuracy_mean.csv", n=30))
    sections.append("<h2>虚拟被试与空间分组</h2>" + _table(root / "group" / "spatial_groups" / "spatial_group_summary.csv"))
    sections.append("<h2>解释边界</h2><p>被试平均结果以被试为统计单位；虚拟被试结果表示跨被试电极模式，不替代传统组水平推断。当前无 FDR 筛选属于探索性分析，应在正式结论中明确多重比较风险。</p>")
    html = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Color iEEG analysis report</title>
<style>body{font-family:Arial,'Noto Sans CJK SC',sans-serif;max-width:1200px;margin:36px auto;color:#20242a;line-height:1.55}h1{border-bottom:2px solid #20242a;padding-bottom:10px}h2{margin-top:34px;border-left:4px solid #2166AC;padding-left:10px}figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid #e0e3e6;background:#fff}figcaption{font-size:13px;color:#5c6670;margin-top:5px}.muted{color:#68717a}.data{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}.data th,.data td{border:1px solid #dfe3e8;padding:5px 8px;white-space:nowrap}.data th{background:#f1f4f7}</style></head><body>""" + "".join(sections) + "</body></html>"
    out.write_text(html, encoding="utf-8")
    return out
