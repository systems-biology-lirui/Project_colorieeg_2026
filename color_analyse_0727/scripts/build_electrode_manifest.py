"""Build the signal/localization intersection manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import METADATA_ROOT, QC_ROOT, SUBJECTS, TASKS
from pipeline.electrode_manifest import write_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=METADATA_ROOT)
    args = parser.parse_args()
    subject_path, task_path = write_manifests(
        args.output,
        subjects=SUBJECTS,
        tasks=TASKS,
        review_csv=QC_ROOT / "bad_channel_candidates.csv",
    )
    print(f"Wrote {subject_path}")
    print(f"Wrote {task_path}")


if __name__ == "__main__":
    main()
