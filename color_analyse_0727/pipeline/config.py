"""Paths and conservative preprocessing defaults for the rebuilt pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "COLOR_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()
MODULE_ROOT = PROJECT_ROOT / "color_analyse_0727"
PIPELINE_ROOT = MODULE_ROOT / "pipeline"
SCRIPTS_ROOT = MODULE_ROOT / "scripts"

SEEG_ROOT = PROJECT_ROOT / "seegdata"
SOURCE_METADATA_ROOT = PROJECT_ROOT / "processed_data"
PROCESS_DATA_ROOT = MODULE_ROOT / "process_data"
QC_ROOT = MODULE_ROOT / "qc"
QC_PLOT_ROOT = QC_ROOT / "channel_diagnostics"
METADATA_ROOT = MODULE_ROOT / "metadata"
MANUAL_DECISIONS_PATH = METADATA_ROOT / "manual_channel_decisions.csv"

SUBJECTS = tuple(f"test00{i}" for i in range(1, 8))
TASKS = (1, 2, 3)

SOURCE_SFREQ = 1000.0
TARGET_SFREQ = 500.0
EPOCH_TMIN_MS = -500.0
EPOCH_TMAX_MS = 1000.0

BANDPASS_HZ = (1.0, 200.0)
NOTCH_HZ = (50.0, 100.0, 150.0)
NOTCH_Q = 30.0

# These are proposals from the previous project and will be shown in the
# review table. They are not applied until a manual decision is recorded.
KNOWN_BAD_CHANNELS = {
    "test001": ("F15",),
    "test002": ("A8", "G7"),
    "test003": ("C13", "D5", "H7", "I1", "I2", "I3"),
    "test005": ("C13", "C14", "F7", "I10"),
}

PROTECTED_CHANNELS = {
    "test001": ("D3",),
    "test002": ("D1", "D2", "D3"),
}


def subject_seeg_dir(subject: str) -> Path:
    number = subject.replace("test", "").lstrip("0")
    return SEEG_ROOT / f"test{number}"


def recording_paths(subject: str, task_num: int) -> tuple[Path, Path]:
    directory = subject_seeg_dir(subject)
    stem = directory / f"erp{int(task_num)}"
    return stem.with_suffix(".set"), stem.with_suffix(".fdt")


def ensure_output_dirs() -> None:
    for directory in (PROCESS_DATA_ROOT, QC_ROOT, QC_PLOT_ROOT, METADATA_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
