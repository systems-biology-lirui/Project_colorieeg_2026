"""Create waveform/PSD panels for non-normal channels in the review table."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import QC_PLOT_ROOT, QC_ROOT, TASKS
from pipeline.quality_plots import plot_channel_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=QC_ROOT / "bad_channel_candidates.csv")
    parser.add_argument("--output", type=Path, default=QC_PLOT_ROOT)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    selected = [row for row in rows if row.get("candidate_level") != "normal"]
    for index, row in enumerate(selected, start=1):
        subject = row["subject"]
        channel = row["channel"]
        safe_channel = re.sub(r"[^A-Za-z0-9_.-]+", "_", channel)
        output = args.output / f"{subject}_{safe_channel}.png"
        plot_channel_diagnostics(subject, channel, output, tasks=TASKS)
        print(f"[{index}/{len(selected)}] {output}", flush=True)

    print(f"Generated {len(selected)} diagnostic panels in {args.output}")


if __name__ == "__main__":
    main()

