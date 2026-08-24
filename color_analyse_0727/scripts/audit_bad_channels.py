"""Audit event codes and channel quality without excluding any channel."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.condition_registry import event_inventory
from pipeline.config import QC_ROOT, SUBJECTS, TASKS, ensure_output_dirs
from pipeline.io_seeg import load_set_metadata
from pipeline.quality_audit import (
    aggregate_channel_rows,
    audit_recording,
    write_aggregated_review_table,
    write_review_table,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", action="append", choices=SUBJECTS)
    parser.add_argument("--task", action="append", type=int, choices=TASKS)
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument(
        "--stage",
        choices=("filtered", "raw"),
        default="filtered",
        help="Signal used for QC; filtered applies the final 1-200 Hz plus 50/100/150 Hz notch chain.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of channels filtered at once (lower this if memory is limited).",
    )
    parser.add_argument("--output", type=Path, default=QC_ROOT / "bad_channel_candidates.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_step < 1:
        raise SystemExit("--sample-step must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    subjects = tuple(args.subject or SUBJECTS)
    tasks = tuple(args.task or TASKS)
    ensure_output_dirs()

    all_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for subject in subjects:
        for task_num in tasks:
            print(f"Auditing {subject} task{task_num} ...", flush=True)
            metadata = load_set_metadata(subject, task_num)
            inventory = event_inventory(task_num, [event.trigger for event in metadata.events])
            event_rows.append(
                {
                    "subject": subject,
                    "task_num": task_num,
                    "set_path": str(metadata.set_path),
                    "sfreq": metadata.sfreq,
                    "n_channels": metadata.n_channels,
                    "n_points": metadata.n_points,
                    "event_total": len(metadata.events),
                    "recognized_total": inventory["recognized_total"],
                    "unknown_total": inventory["unknown_total"],
                    "recognized_counts": json.dumps(inventory["recognized_counts"], ensure_ascii=False),
                    "unknown_counts": json.dumps(inventory["unknown_counts"], ensure_ascii=False),
                }
            )
            all_rows.extend(
                audit_recording(
                    metadata,
                    sample_step=args.sample_step,
                    apply_filter=args.stage == "filtered",
                    batch_size=args.batch_size,
                )
            )

    detail_path = QC_ROOT / "bad_channel_candidates_by_recording.csv"
    write_review_table(all_rows, detail_path)
    aggregate_rows = aggregate_channel_rows(all_rows)
    write_aggregated_review_table(aggregate_rows, args.output)
    review_only_path = QC_ROOT / "bad_channel_candidates_to_review_filtered.csv"
    write_aggregated_review_table(
        [row for row in aggregate_rows if row["candidate_level"] != "normal"],
        review_only_path,
    )
    write_summary(all_rows, QC_ROOT / "bad_channel_candidate_summary.csv")
    with (QC_ROOT / "event_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)

    print(f"Wrote channel review table: {args.output}")
    print(f"Wrote candidate-only review table: {review_only_path}")
    print(f"Wrote event inventory: {QC_ROOT / 'event_inventory.csv'}")
    print("No channel was excluded or modified.")


if __name__ == "__main__":
    main()
