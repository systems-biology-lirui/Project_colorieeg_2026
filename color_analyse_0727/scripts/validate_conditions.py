"""Validate the canonical condition registry without reading signal data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.condition_registry import registry_rows


def main() -> None:
    output = ROOT / "qc" / "condition_registry.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = registry_rows()
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

