"""Run the stimulus luminance/contrast/colorfulness audit standalone."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import STIMULI_ROOT  # noqa: E402
from analysis.luminance import run_luminance_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "result" / f"final_analysis_{date.today():%Y%m%d}",
        help="Result directory; stage04_luminance will be written there.",
    )
    parser.add_argument(
        "--stim-root",
        type=Path,
        default=STIMULI_ROOT,
        help="Root folder containing Stimuli_Task1/2/3.",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tables = run_luminance_audit(out, stim_root=args.stim_root.resolve())
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")
    print(f"Audit written to {out / 'stage04_luminance'}")


if __name__ == "__main__":
    main()
