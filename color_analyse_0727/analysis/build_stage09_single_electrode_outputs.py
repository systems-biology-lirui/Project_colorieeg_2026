"""Build explicit single-electrode indexes from completed stage09 RSA outputs.

The underlying CSVs already contain subject/channel-level results.  This
utility does not rerun neural analysis; it adds a compact electrode index and
one-row-per-electrode/time-bin wide RDM table for every current stage09 branch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "color_analyse_0727"
RESULT_ROOT = MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected"

STAGES = (
    "stage09_task1_condition_rsa_raw200",
    "stage09_1_task2_grayfruit_rsa_raw200",
    "stage09_2_task3_purecolor_rsa_raw200",
    "stage09_3_task2_task3_cross_rsa_raw200",
)
CURVE_FILES = {
    "stage09_1_task2_grayfruit_rsa_raw200": "memory_color_distance_curves.csv",
    "stage09_2_task3_purecolor_rsa_raw200": "memory_color_distance_curves.csv",
    "stage09_3_task2_task3_cross_rsa_raw200": "cross_task_memory_color_distance_curves.csv",
}
KEY_COLUMNS = ["subject", "channel", "time_bin_index"]
RDM_INDEX_COLUMNS = KEY_COLUMNS + ["bin_start_ms", "bin_end_ms"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figure_for(stage_dir: Path, subject: str, channel: str) -> str:
    matches = sorted(stage_dir.joinpath("electrode_figures").glob(f"{subject}_{channel}_*.png"))
    if not matches:
        return ""
    return matches[0].relative_to(stage_dir).as_posix()


def _build_stage(stage_dir: Path) -> dict[str, object]:
    rdm_path = stage_dir / "condition_rdm_long.csv"
    sets_path = stage_dir / "electrode_sets_used.csv"
    if not rdm_path.exists() or not sets_path.exists():
        raise FileNotFoundError(f"Missing completed RSA inputs under {stage_dir}")

    rdm = pd.read_csv(rdm_path)
    sets = pd.read_csv(sets_path)
    required = set(KEY_COLUMNS + ["condition_i", "condition_j", "distance", "bin_start_ms", "bin_end_ms"])
    missing = required.difference(rdm.columns)
    if missing:
        raise ValueError(f"{rdm_path} is missing columns: {sorted(missing)}")

    rdm["pair"] = rdm["condition_i"].astype(str) + "__" + rdm["condition_j"].astype(str)
    wide = (
        rdm.pivot_table(
            index=RDM_INDEX_COLUMNS,
            columns="pair",
            values="distance",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [str(column) for column in wide.columns]
    wide["stage"] = str(rdm["stage"].iloc[0]) if "stage" in rdm.columns and not rdm.empty else stage_dir.name

    curve_name = CURVE_FILES.get(stage_dir.name)
    if curve_name:
        curve_path = stage_dir / curve_name
        curve = pd.read_csv(curve_path)
        curve_keep = KEY_COLUMNS + [
            column
            for column in (
                "same_memory_color_distance",
                "different_memory_color_distance",
                "different_minus_same",
                "within_memory_color_distance",
                "between_memory_color_distance",
                "between_minus_within",
                "red_green_distance",
            )
            if column in curve.columns
        ]
        wide = wide.merge(curve[curve_keep], on=KEY_COLUMNS, how="left", validate="one_to_one")

    wide_path = stage_dir / "single_electrode_rdm_summary.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    set_columns = [
        column
        for column in ("subject", "channel", "mni_x", "mni_y", "mni_z", "roi", "electrode_sets", "S1", "S2", "CSC")
        if column in sets.columns
    ]
    electrode_index = sets[set_columns].drop_duplicates(["subject", "channel"]).copy()
    counts = (
        wide.groupby(["subject", "channel"], as_index=False)
        .agg(
            n_time_bins=("time_bin_index", "nunique"),
            n_rdm_rows=("time_bin_index", "size"),
        )
    )
    electrode_index = electrode_index.merge(counts, on=["subject", "channel"], how="left", validate="one_to_one")
    electrode_index["figure_path"] = [
        _figure_for(stage_dir, str(subject), str(channel))
        for subject, channel in zip(electrode_index["subject"], electrode_index["channel"])
    ]
    electrode_index["numeric_result_path"] = wide_path.name
    electrode_index = electrode_index.sort_values(["subject", "channel"]).reset_index(drop=True)
    index_path = stage_dir / "single_electrode_index.csv"
    electrode_index.to_csv(index_path, index=False, encoding="utf-8-sig")

    return {
        "stage": stage_dir.name,
        "unique_electrodes": int(len(electrode_index)),
        "rdm_summary_rows": int(len(wide)),
        "rdm_cells_per_row": int(len([column for column in wide.columns if "__" in column])),
        "outputs": [str(index_path), str(wide_path)],
    }


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_stage09_single_electrode_outputs"
    run_dir.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc)
    (run_dir / "start_time.txt").write_text(start.isoformat() + "\n", encoding="utf-8")
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    summaries = []
    manifest = []
    try:
        for stage_name in STAGES:
            stage_dir = RESULT_ROOT / stage_name
            summaries.append(_build_stage(stage_dir))
            for path in sorted(stage_dir.glob("condition_rdm_long.csv")):
                manifest.append({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
        end = datetime.now(timezone.utc)
        run_summary = {
            "status": "completed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": end.isoformat(),
            "stages": summaries,
            "input_manifest": manifest,
            "run_dir": str(run_dir),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "output_paths.txt").write_text(
            "\n".join(path for summary in summaries for path in summary["outputs"]) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        (run_dir / "finish_time.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
