#!/usr/bin/env python3
"""Build a read-only inventory and lightweight QC report for seegdata.

This script never edits the EEGLAB ``.set/.fdt`` inputs.  It supports both
classic MATLAB files and MATLAB v7.3/HDF5 ``.set`` files used in this project.
The continuous-signal checks are deliberately lightweight and sampled; their
purpose is to flag runs/channels for review before formal preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


EXPECTED_TRIGGERS = {
    "erp1": {str(code): 70 for code in (11, 12, 21, 22, 31, 32, 41, 42)},
    "erp2": {str(code): 60 for code in range(101, 134) if code % 10 in (1, 2, 3)},
    "erp3": {str(code): 60 for code in (51, 52, 53, 54, 55, 56)},
}
CONTROL_TRIGGER = "99"
EVENT_PREFIX = "Trigger-In:"


@dataclass
class SetMetadata:
    nbchan: int
    pnts: int
    trials: int
    srate: float
    xmin: float
    xmax: float
    channel_labels: list[str]
    event_types: list[str]
    event_latencies: list[float]


def _matlab_h5_value(handle: h5py.File, ref: Any) -> Any:
    obj = handle[ref]
    value = obj[()]
    matlab_class = obj.attrs.get("MATLAB_class", b"")
    if matlab_class == b"char":
        return "".join(chr(int(x)) for x in np.ravel(value, order="F") if x)
    flat = np.ravel(value)
    if flat.size == 1:
        return float(flat[0])
    return np.asarray(value).squeeze().tolist()


def _read_hdf5_set(path: Path) -> SetMetadata:
    with h5py.File(path, "r") as handle:
        scalar = lambda name: float(np.ravel(handle[name][()])[0])
        labels = [_matlab_h5_value(handle, ref) for ref in handle["chanlocs/labels"][0]]
        types = [_matlab_h5_value(handle, ref) for ref in handle["event/type"][:, 0]]
        latencies = [_matlab_h5_value(handle, ref) for ref in handle["event/latency"][:, 0]]
        return SetMetadata(
            nbchan=int(scalar("nbchan")),
            pnts=int(scalar("pnts")),
            trials=int(scalar("trials")),
            srate=scalar("srate"),
            xmin=scalar("xmin"),
            xmax=scalar("xmax"),
            channel_labels=[str(x).strip() for x in labels],
            event_types=[str(x).strip() for x in types],
            event_latencies=[float(x) for x in latencies],
        )


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    return [dict(x) if not isinstance(x, dict) else x for x in np.atleast_1d(value)]


def _read_classic_set(path: Path) -> SetMetadata:
    loaded = loadmat(path, simplify_cells=True)
    eeg = loaded.get("EEG", loaded)
    channels = _as_records(eeg["chanlocs"])
    events = _as_records(eeg["event"])
    return SetMetadata(
        nbchan=int(eeg["nbchan"]),
        pnts=int(eeg["pnts"]),
        trials=int(eeg["trials"]),
        srate=float(eeg["srate"]),
        xmin=float(eeg["xmin"]),
        xmax=float(eeg["xmax"]),
        channel_labels=[str(x.get("labels", "")).strip() for x in channels],
        event_types=[str(x.get("type", "")).strip() for x in events],
        event_latencies=[float(x.get("latency", math.nan)) for x in events],
    )


def read_set_metadata(path: Path) -> SetMetadata:
    try:
        return _read_hdf5_set(path)
    except OSError:
        return _read_classic_set(path)


def normalize_trigger(event_type: str) -> str:
    return event_type.removeprefix(EVENT_PREFIX).strip()


def subject_id(raw_subject_dir: str) -> str:
    match = re.fullmatch(r"test0*(\d+)", raw_subject_dir, flags=re.IGNORECASE)
    if not match:
        return raw_subject_dir
    return f"test{int(match.group(1)):03d}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sampled_channel_qc(fdt_path: Path, metadata: SetMetadata, target_samples: int) -> list[dict[str, Any]]:
    if not fdt_path.exists() or metadata.trials != 1:
        return []
    continuous = np.memmap(
        fdt_path,
        dtype="<f4",
        mode="r",
        shape=(metadata.pnts, metadata.nbchan),
    )
    stride = max(1, metadata.pnts // target_samples)
    sampled = np.asarray(continuous[::stride], dtype=np.float64)
    finite_fraction = np.mean(np.isfinite(sampled), axis=0)
    sampled[~np.isfinite(sampled)] = np.nan
    channel_sd = np.nanstd(sampled, axis=0)
    channel_p2p = np.nanmax(sampled, axis=0) - np.nanmin(sampled, axis=0)
    positive_sd = channel_sd[np.isfinite(channel_sd) & (channel_sd > 0)]
    median_sd = float(np.median(positive_sd)) if positive_sd.size else math.nan
    rows = []
    for index, label in enumerate(metadata.channel_labels):
        sd_ratio = channel_sd[index] / median_sd if median_sd > 0 else math.nan
        flags = []
        if finite_fraction[index] < 1.0:
            flags.append("nonfinite")
        if np.isfinite(sd_ratio) and sd_ratio < 0.01:
            flags.append("near_flat")
        if np.isfinite(sd_ratio) and sd_ratio > 10.0:
            flags.append("high_variance")
        rows.append(
            {
                "channel_index_1based": index + 1,
                "channel": label,
                "sample_count": sampled.shape[0],
                "finite_fraction": round(float(finite_fraction[index]), 8),
                "sampled_sd_native_unit": round(float(channel_sd[index]), 8),
                "sampled_p2p_native_unit": round(float(channel_p2p[index]), 8),
                "sd_over_run_median": round(float(sd_ratio), 8),
                "qc_flags": ";".join(flags),
            }
        )
    return rows


def discover_metadata(processed_root: Path, subject: str) -> dict[str, bool]:
    subject_dir = processed_root / subject
    return {
        "subject_dir_exists": subject_dir.is_dir(),
        "ieegloc_xlsx_exists": (subject_dir / f"{subject}_ieegloc.xlsx").is_file(),
        "any_tsv_exists": any(subject_dir.glob("*.tsv")) if subject_dir.is_dir() else False,
        "groupeddata_exists": (subject_dir / "groupedData.mat").is_file(),
        "task3_groupeddata_exists": (subject_dir / "task3groupedData.mat").is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-samples", type=int, default=2000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    seeg_root = project_root / "seegdata"
    processed_root = project_root / "processed_data"
    output_dir = (args.output_dir or project_root / "result" / "preprocessing_qc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    set_paths = sorted(seeg_root.glob("test*/erp*.set"))
    for set_path in set_paths:
        raw_subject = set_path.parent.name
        subject = subject_id(raw_subject)
        run = set_path.stem
        fdt_path = set_path.with_suffix(".fdt")
        metadata = read_set_metadata(set_path)
        expected_fdt_bytes = metadata.nbchan * metadata.pnts * metadata.trials * 4
        actual_fdt_bytes = fdt_path.stat().st_size if fdt_path.exists() else -1
        trigger_counts = Counter(normalize_trigger(x) for x in metadata.event_types)
        expected = EXPECTED_TRIGGERS.get(run, {})
        metadata_files = discover_metadata(processed_root, subject)

        duplicate_channels = len(metadata.channel_labels) - len(set(metadata.channel_labels))
        latencies = np.asarray(metadata.event_latencies, dtype=float)
        latency_monotonic = bool(np.all(np.diff(latencies[np.isfinite(latencies)]) >= 0))
        latency_in_bounds = bool(np.all((latencies[np.isfinite(latencies)] >= 1) & (latencies[np.isfinite(latencies)] <= metadata.pnts)))
        boundary_count = sum(t.lower() == "boundary" for t in metadata.event_types)

        missing_trials = sum(max(expected_count - trigger_counts.get(code, 0), 0) for code, expected_count in expected.items())
        excess_trials = sum(max(trigger_counts.get(code, 0) - expected_count, 0) for code, expected_count in expected.items())
        unexpected_task_events = sorted(
            code for code in trigger_counts if code not in expected and code not in (CONTROL_TRIGGER, "boundary")
        )
        run_flags = []
        if actual_fdt_bytes != expected_fdt_bytes:
            run_flags.append("fdt_size_mismatch")
        if duplicate_channels:
            run_flags.append("duplicate_channels")
        if not latency_monotonic:
            run_flags.append("nonmonotonic_events")
        if not latency_in_bounds:
            run_flags.append("event_out_of_bounds")
        if boundary_count:
            run_flags.append("boundary_event")
        if missing_trials:
            run_flags.append("missing_expected_trials")
        if excess_trials:
            run_flags.append("excess_vs_nominal_trials")
        if unexpected_task_events:
            run_flags.append("unexpected_event_type")

        run_row = {
            "subject": subject,
            "raw_subject_dir": raw_subject,
            "run": run,
            "set_path": str(set_path.relative_to(project_root)),
            "fdt_path": str(fdt_path.relative_to(project_root)),
            "srate_hz": metadata.srate,
            "channels": metadata.nbchan,
            "points": metadata.pnts,
            "duration_seconds": round(metadata.pnts / metadata.srate, 3),
            "events": len(metadata.event_types),
            "boundary_events": boundary_count,
            "missing_expected_trials": missing_trials,
            "excess_vs_nominal_trials": excess_trials,
            "duplicate_channels": duplicate_channels,
            "fdt_bytes_expected": expected_fdt_bytes,
            "fdt_bytes_actual": actual_fdt_bytes,
            "fdt_size_ok": actual_fdt_bytes == expected_fdt_bytes,
            "event_latencies_monotonic": latency_monotonic,
            "event_latencies_in_bounds": latency_in_bounds,
            "run_qc_flags": ";".join(run_flags),
            **metadata_files,
        }
        run_rows.append(run_row)

        for code in sorted(set(trigger_counts) | set(expected), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            observed = trigger_counts.get(code, 0)
            expected_count = expected.get(code, "")
            event_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "event_type": code,
                    "observed_count": observed,
                    "expected_count": expected_count,
                    "count_difference": observed - expected_count if isinstance(expected_count, int) else "",
                }
            )

        sampled_rows = sampled_channel_qc(fdt_path, metadata, args.target_samples)
        for row in sampled_rows:
            row.update({"subject": subject, "run": run})
        channel_rows.extend(sampled_rows)

        for flag in run_flags:
            issue_rows.append({"subject": subject, "run": run, "scope": "run", "item": "", "issue": flag})
        for row in sampled_rows:
            for flag in filter(None, row["qc_flags"].split(";")):
                issue_rows.append(
                    {"subject": subject, "run": run, "scope": "channel_sampled", "item": row["channel"], "issue": flag}
                )

    write_csv(output_dir / "runs.csv", run_rows, list(run_rows[0]) if run_rows else [])
    write_csv(
        output_dir / "event_counts.csv",
        event_rows,
        ["subject", "run", "event_type", "observed_count", "expected_count", "count_difference"],
    )
    write_csv(
        output_dir / "channel_sampled_qc.csv",
        channel_rows,
        [
            "subject", "run", "channel_index_1based", "channel", "sample_count", "finite_fraction",
            "sampled_sd_native_unit", "sampled_p2p_native_unit", "sd_over_run_median", "qc_flags",
        ],
    )
    write_csv(output_dir / "issues.csv", issue_rows, ["subject", "run", "scope", "item", "issue"])

    subjects = sorted({row["subject"] for row in run_rows})
    summary = {
        "project_root": str(project_root),
        "source_data": str(seeg_root),
        "raw_inputs_modified": False,
        "subjects": subjects,
        "subject_count": len(subjects),
        "erp_run_count": len(run_rows),
        "runs_with_flags": sum(bool(row["run_qc_flags"]) for row in run_rows),
        "sampled_channels_with_flags": sum(bool(row["qc_flags"]) for row in channel_rows),
        "notes": [
            "Continuous channel QC is sampled and is a screening step, not an automatic rejection decision.",
            "Expected trial counts encode the dominant protocol; count deviations must be reviewed against acquisition logs.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    flagged_runs = [row for row in run_rows if row["run_qc_flags"]]
    repeated_channel_flags = Counter(
        (row["subject"], row["channel"])
        for row in channel_rows
        if row["qc_flags"]
    )
    missing_loc = sorted({row["subject"] for row in run_rows if not row["ieegloc_xlsx_exists"]})
    missing_pairing = sorted({row["subject"] for row in run_rows if not row["groupeddata_exists"]})
    report_lines = [
        "# SEEG raw inventory and lightweight QC",
        "",
        f"- Subjects: {len(subjects)} ({', '.join(subjects)})",
        f"- ERP runs: {len(run_rows)}",
        f"- Runs with flags: {len(flagged_runs)}",
        f"- Sampled channels with flags: {summary['sampled_channels_with_flags']}",
        "- Raw inputs modified: no",
        "",
        "## Flagged runs",
        "",
    ]
    if flagged_runs:
        report_lines += ["| Subject | Run | Flags |", "|---|---|---|"]
        report_lines += [f"| {row['subject']} | {row['run']} | {row['run_qc_flags']} |" for row in flagged_runs]
    else:
        report_lines.append("No run-level flags.")
    report_lines += ["", "## Repeated sampled channel flags", ""]
    repeated = [(subject, channel, count) for (subject, channel), count in sorted(repeated_channel_flags.items()) if count >= 2]
    if repeated:
        report_lines += ["| Subject | Channel | Flagged runs |", "|---|---|---:|"]
        report_lines += [f"| {subject} | {channel} | {count} |" for subject, channel, count in repeated]
    else:
        report_lines.append("No channel was flagged in two or more runs.")
    report_lines += [
        "",
        "## Metadata gaps",
        "",
        f"- Missing standardized `*_ieegloc.xlsx`: {', '.join(missing_loc) if missing_loc else 'none'}",
        f"- Missing task1 groupedData: {', '.join(missing_pairing) if missing_pairing else 'none'}",
        "",
        "## Interpretation",
        "",
        "`channel_sampled_qc.csv` uses regularly sampled continuous values. Flags are candidates for visual/full-resolution review only.",
        "No channel or epoch is rejected by this inventory stage.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
