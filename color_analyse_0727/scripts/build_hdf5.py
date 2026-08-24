"""Build reviewed HDF5 data after manual channel decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import METADATA_ROOT, PROCESS_DATA_ROOT, QC_ROOT, SUBJECTS, TASKS, ensure_output_dirs
from pipeline.io_seeg import load_set_metadata
from pipeline.preprocess import export_recording


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", action="append", choices=SUBJECTS)
    parser.add_argument("--task", action="append", type=int, choices=TASKS)
    parser.add_argument(
        "--review",
        type=Path,
        default=QC_ROOT / "bad_channel_candidates.csv",
        help="CSV edited by the user; only explicit manual exclusions are applied.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=METADATA_ROOT / "electrode_manifest_by_task.csv",
        help="Signal/localization intersection manifest for bipolar centers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    subjects = tuple(args.subject or SUBJECTS)
    tasks = tuple(args.task or TASKS)
    for subject in subjects:
        for task_num in tasks:
            metadata = load_set_metadata(subject, task_num)
            output = export_recording(
                metadata,
                args.review,
                PROCESS_DATA_ROOT,
                manifest_csv=args.manifest,
            )
            print(f"Exported {output}")


if __name__ == "__main__":
    main()
