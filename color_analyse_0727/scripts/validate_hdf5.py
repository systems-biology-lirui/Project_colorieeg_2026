"""Validate all generated HDF5 files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import METADATA_ROOT, PROCESS_DATA_ROOT
from pipeline.hdf5_io import inspect_hdf5, validate_hdf5_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROCESS_DATA_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=METADATA_ROOT / "hdf5_validation_report.csv",
    )
    args = parser.parse_args()
    paths = sorted(args.root.glob("test*/task*_epoched_1_200Hz.h5"))
    if not paths:
        raise SystemExit(f"No HDF5 files found under {args.root}")
    failed = 0
    report_rows: list[dict[str, object]] = []
    for path in paths:
        errors = validate_hdf5_file(path)
        info = inspect_hdf5(path)
        report_rows.append(
            {
                "path": str(path),
                "subject": info["subject"],
                "task_num": info["task_num"],
                "n_channels": info["n_channels"],
                "n_times": info["n_times"],
                "conditions": len(info["condition_names"]),
                "trial_counts": json.dumps(
                    dict(zip(info["condition_names"], info["trial_counts"])),
                    ensure_ascii=False,
                ),
                "status": "PASS" if not errors else "FAIL",
                "errors": ";".join(errors),
            }
        )
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            counts = ", ".join(
                f"{name}={count}" for name, count in zip(info["condition_names"], info["trial_counts"])
            )
            print(f"PASS {path} | channels={info['n_channels']} | {counts}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote validation report: {args.report}")
    if failed:
        raise SystemExit(f"{failed} HDF5 files failed validation")


if __name__ == "__main__":
    main()
