"""Combine legacy test001--003 tables with newly computed test005--006 tables.

This is a bookkeeping utility, not a new analysis: it retains all per-subject
columns and recomputes only the descriptive five-subject mean from those saved
accuracy curves.  Legacy GLMM values are intentionally not carried forward,
because their trial-level inputs are not present in the exported CSV files.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


ROOT = Path('/home/lirui/liulab_project/ieeg/Project_colorieeg_2026')
ANALYSE = ROOT / 'color_cognition_pipeline' / 'analyse_0617'
LEGACY = ANALYSE / 'doc'
NEW = ANALYSE / 'run_5subjects_original' / 'doc'
OUT = NEW / 'five_subject_combined'


def merge_decoding_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for legacy_path in sorted(LEGACY.glob('decoding_data_*.csv')):
        new_path = NEW / legacy_path.name
        if not new_path.exists():
            continue
        old = pd.read_csv(legacy_path)
        new = pd.read_csv(new_path)
        old_subject_columns = [c for c in old if re.fullmatch(r'test\d+_Acc', c)]
        new_subject_columns = [c for c in new if re.fullmatch(r'test\d+_Acc', c)]
        if not old_subject_columns or not new_subject_columns:
            continue
        merged = old[['Time_ms', *old_subject_columns]].merge(
            new[['Time_ms', *new_subject_columns]], on='Time_ms', validate='one_to_one'
        )
        subject_columns = [*old_subject_columns, *new_subject_columns]
        merged.insert(1, 'N_Subjects', merged[subject_columns].notna().sum(axis=1))
        merged.insert(2, 'Group_Mean_Acc', merged[subject_columns].mean(axis=1))
        stem = legacy_path.stem.replace('decoding_data_', '')
        merged.to_csv(OUT / f'decoding_data_{stem}_five_subject_mean.csv', index=False)
        merged.to_excel(OUT / f'decoding_data_{stem}_five_subject_mean.xlsx', index=False)


if __name__ == '__main__':
    merge_decoding_tables()
