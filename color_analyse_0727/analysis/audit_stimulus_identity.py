"""Audit Task 3 stimulus-file counts against the actual HDF5 trial counts.

This audit deliberately does not infer a trial-to-image mapping.  The current
HDF5 contract stores condition-level epochs but no image name per epoch, so the
report distinguishes observed file/trial counts from inferred average reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd

from analysis.common import PROCESS_ROOT, STIMULI_ROOT, SUBJECTS


CONDITIONS = {
    "red": "Red_Color",
    "yellow": "Yellow_Color",
    "blue": "Blue_Color",
    "green": "Green_Color",
    "black": "Black_Color",
    "white": "White_Color",
}


def audit_subject(subject: str) -> list[dict[str, object]]:
    h5_path = PROCESS_ROOT / subject / "task3_epoched_1_200Hz.h5"
    stimulus_dir = STIMULI_ROOT / "Stimuli_Task3"
    rows: list[dict[str, object]] = []
    with h5py.File(h5_path, "r") as h5:
        datasets = set(h5.keys())
        identity_keys = sorted(
            key
            for key in datasets
            if any(token in key.lower() for token in ("image", "stimulus", "filename", "trial_info"))
        )
        for condition, stem in CONDITIONS.items():
            files = sorted(stimulus_dir.glob(f"{stem}_*.bmp"))
            n_trials = int(h5["epochs"][condition].shape[0])
            n_files = len(files)
            rows.append(
                {
                    "subject": subject,
                    "task": 3,
                    "condition": condition,
                    "stimulus_stem": stem,
                    "stimulus_files": n_files,
                    "stimulus_filenames": ";".join(path.name for path in files),
                    "actual_hdf5_trials": n_trials,
                    "mean_trials_per_file": round(n_trials / n_files, 3) if n_files else None,
                    "trial_image_identity_in_hdf5": bool(identity_keys),
                    "identity_keys_found": ";".join(identity_keys),
                    "exact_repeat_count_proven": False,
                    "interpretation": (
                        "observed files and epochs; mapping missing"
                        if not identity_keys
                        else "identity dataset found; inspect before using"
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result/final_analysis_seeg_20260806_corrected/stage06_exploration"),
    )
    args = parser.parse_args()
    rows = [row for subject in SUBJECTS for row in audit_subject(subject)]
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "exemplar_identity_audit_actual_trials.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(path)
    print(table[["subject", "condition", "stimulus_files", "actual_hdf5_trials", "mean_trials_per_file", "trial_image_identity_in_hdf5"]].to_string(index=False))


if __name__ == "__main__":
    main()
