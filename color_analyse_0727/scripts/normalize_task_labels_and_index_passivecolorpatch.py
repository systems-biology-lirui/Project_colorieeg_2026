"""Normalize Task2/Task3 labels and build a trial-level shape index.

The index is deliberately trial-level and does not split or rewrite any EEG
data.  It joins the MATLAB presentation log (results/stimData) to the actual
Task3 event stream in seegdata/erp3.set using the trigger-code sequence.
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Data"
SEEG_ROOT = ROOT.parent / "seegdata"
METADATA_ROOT = ROOT / "metadata"
DEFAULT_INDEX = METADATA_ROOT / "passivecolorpatch_shape_trial_index.csv"
DEFAULT_AUDIT = METADATA_ROOT / "passivecolorpatch_alignment_audit.csv"
DEFAULT_RENAMES = METADATA_ROOT / "task_label_rename_manifest.csv"

SWAPPED_SUBJECTS = {f"test{i:03d}" for i in range(1, 5)}
COLOR_FILE_RE = re.compile(r"^(?P<prefix>.*)_Task(?P<task>[23])Passive(?P<kind>ColorPatches|FruitFull)(?P<suffix>_.*\.mat)$")
SHAPE_RE = re.compile(r"_(?P<shape>[0-9]+)\.[^.]+$")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if arr.dtype.kind in "US":
        return "".join(arr.astype(str).ravel().tolist()).strip()
    if arr.dtype.kind in "iu":
        return "".join(chr(int(x)) for x in arr.ravel()).strip()
    return str(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return int(float(arr.ravel()[0]))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return float(arr.ravel()[0])


def _parse_shape(filename: str) -> int | None:
    match = SHAPE_RE.search(filename)
    return int(match.group("shape")) if match else None


def _is_color_log(path: Path) -> bool:
    data = loadmat(path, simplify_cells=True)
    rows = data.get("results", [])
    if isinstance(rows, dict):
        rows = [rows]
    names = [_as_text(row.get("imgName")) for row in rows if isinstance(row, dict)]
    color_rows = [name for name in names if "_Color_" in name]
    return bool(color_rows) and len(color_rows) >= max(1, len(names) // 2)


def _swap_task_label(name: str) -> str:
    replacements = {
        "Task2PassiveColorPatches": "Task3PassiveColorPatches",
        "Task3PassiveFruitFull": "Task2PassiveFruitFull",
    }
    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def normalize_labels() -> list[dict[str, str]]:
    """Rename only the known-reversed subjects, using two-phase renaming."""
    records: list[dict[str, str]] = []
    for subject in sorted(SWAPPED_SUBJECTS):
        folder = DATA_ROOT / subject
        candidates = [
            path
            for path in folder.iterdir()
            if path.is_file()
            and ("Task2PassiveColorPatches" in path.name or "Task3PassiveFruitFull" in path.name)
        ]
        for path in sorted(candidates):
            # Session-history files contain the same semantic label but do not
            # contain a results table, so validate only the trial-log files.
            if not path.name.startswith(f"{subject}_Task"):
                continue
            if "Task2PassiveColorPatches" in path.name and not _is_color_log(path):
                raise RuntimeError(f"Expected color log but found otherwise: {path}")
            if "Task3PassiveFruitFull" in path.name and _is_color_log(path):
                raise RuntimeError(f"Expected fruit log but found color log: {path}")

        staged: list[tuple[Path, Path, Path]] = []
        for path in sorted(candidates):
            target = path.with_name(_swap_task_label(path.name))
            if target.exists() and target != path:
                raise FileExistsError(f"Rename target already exists: {target}")
            temporary = path.with_name(f".__task_label_tmp__{path.name}")
            path.rename(temporary)
            staged.append((path, temporary, target))
        for old, temporary, target in staged:
            temporary.rename(target)
            records.append(
                {
                    "subject": subject,
                    "old_name": old.name,
                    "new_name": target.name,
                    "reason": "first four subjects had Task2/Task3 labels reversed",
                }
            )
    return records


def infer_existing_rename_manifest() -> list[dict[str, str]]:
    """Reconstruct the manifest if a previous index run already renamed files."""
    inverse = {
        "Task3PassiveColorPatches": "Task2PassiveColorPatches",
        "Task2PassiveFruitFull": "Task3PassiveFruitFull",
    }
    records: list[dict[str, str]] = []
    for subject in sorted(SWAPPED_SUBJECTS):
        folder = DATA_ROOT / subject
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if "Task3PassiveColorPatches" in path.name:
                old_name = path.name.replace("Task3PassiveColorPatches", inverse["Task3PassiveColorPatches"])
            elif "Task2PassiveFruitFull" in path.name:
                old_name = path.name.replace("Task2PassiveFruitFull", inverse["Task2PassiveFruitFull"])
            else:
                continue
            records.append(
                {
                    "subject": subject,
                    "old_name": old_name,
                    "new_name": path.name,
                    "reason": "first four subjects had Task2/Task3 labels reversed",
                }
            )
    return records


def _decode_h5_ref_string(handle: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(handle[reference][...]).ravel()
    return "".join(chr(int(value)) for value in values).strip()


def load_eeg_events(path: Path) -> tuple[list[int], list[float | None], list[str]]:
    """Read EEGLAB event type/latency from v5 or v7.3 .set files."""
    try:
        eeg = loadmat(path, simplify_cells=True)["EEG"]
        events = eeg.get("event", [])
        if isinstance(events, dict):
            events = [events]
        types = [_as_text(event.get("type")) for event in events]
        latencies = [_as_float(event.get("latency")) for event in events]
        return [_as_int(re.search(r"(\d+)$", event_type).group(1)) for event_type in types], latencies, types
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            type_refs = handle["event/type"][...].ravel()
            latency_refs = handle["event/latency"][...].ravel()
            types = [_decode_h5_ref_string(handle, reference) for reference in type_refs]
            latencies = [float(np.asarray(handle[reference][...]).ravel()[0]) for reference in latency_refs]
            markers = [int(re.search(r"(\d+)$", event_type).group(1)) for event_type in types]
            return markers, latencies, types


def _alignment(log_markers: list[int], eeg_markers: list[int]) -> tuple[dict[int, int], list[dict[str, int | str]]]:
    """Return log-index -> EEG-index mapping and explicit sequence audit rows."""
    matcher = SequenceMatcher(None, log_markers, eeg_markers, autojunk=False)
    mapping: dict[int, int] = {}
    audit: list[dict[str, int | str]] = []
    for tag, log_start, log_end, eeg_start, eeg_end in matcher.get_opcodes():
        if tag == "equal":
            for log_idx, eeg_idx in zip(range(log_start, log_end), range(eeg_start, eeg_end)):
                mapping[log_idx] = eeg_idx
        else:
            audit.append(
                {
                    "operation": tag,
                    "log_start_1based": log_start + 1,
                    "log_end_1based": log_end,
                    "eeg_start_1based": eeg_start + 1 if eeg_start < eeg_end else "",
                    "eeg_end_1based": eeg_end if eeg_start < eeg_end else "",
                    "log_count": log_end - log_start,
                    "eeg_count": eeg_end - eeg_start,
                }
            )
    return mapping, audit


def build_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for subject_dir in sorted(DATA_ROOT.glob("test[0-9][0-9][0-9]")):
        subject = subject_dir.name
        log_files = sorted(subject_dir.glob(f"{subject}_Task3PassiveColorPatches_*.mat"))
        if not log_files:
            raise FileNotFoundError(f"No normalized Task3 ColorPatches log found for {subject}")

        log_rows: list[tuple[Path, int, dict[str, object], dict[str, object]]] = []
        for session_number, path in enumerate(log_files, start=1):
            data = loadmat(path, simplify_cells=True)
            results = data.get("results", [])
            stim_data = data.get("stimData", [])
            if isinstance(results, dict):
                results = [results]
            if isinstance(stim_data, dict):
                stim_data = [stim_data]
            if len(results) != len(stim_data):
                raise RuntimeError(f"results/stimData length mismatch: {path}")
            for trial_number, (result, stim) in enumerate(zip(results, stim_data), start=1):
                log_rows.append((path, session_number, result, stim))

        log_markers = [_as_int(result.get("marker")) for _, _, result, _ in log_rows]
        eeg_path = SEEG_ROOT / subject.replace("test", "test") / "erp3.set"
        # seegdata uses test1 ... test7, while Data uses test001 ... test007.
        eeg_path = SEEG_ROOT / f"test{int(subject[-3:])}" / "erp3.set"
        eeg_markers, eeg_latencies, eeg_types = load_eeg_events(eeg_path)
        mapping, sequence_audit = _alignment(log_markers, eeg_markers)

        for item in sequence_audit:
            audit_rows.append({"subject": subject, **item})
        matched = len(mapping)
        log_count = len(log_rows)
        eeg_count = len(eeg_markers)
        audit_rows.append(
            {
                "subject": subject,
                "operation": "summary",
                "log_start_1based": 1,
                "log_end_1based": log_count,
                "eeg_start_1based": 1,
                "eeg_end_1based": eeg_count,
                "log_count": log_count,
                "eeg_count": eeg_count,
                "matched_count": matched,
                "exact_sequence": log_markers == eeg_markers,
            }
        )

        eeg_epoch_index = 0
        color_shape_occurrences: dict[tuple[str, int], int] = {}
        color_occurrences: dict[str, int] = {}
        for global_trial, (path, session_number, result, stim) in enumerate(log_rows, start=1):
            log_idx = global_trial - 1
            filename = _as_text(result.get("imgName"))
            category = _as_text(result.get("category")) or _as_text(stim.get("catName"))
            color = category if category and category != "CatchTrial" else ""
            shape_id = _parse_shape(filename)
            is_catch = bool(_as_int(result.get("isCatch")) or filename == "Catch_Trial")
            has_shape = shape_id is not None and "_Color_" in filename
            is_color_trial = (not is_catch) and has_shape
            if is_color_trial:
                color_occurrences[color] = color_occurrences.get(color, 0) + 1
                key = (color, shape_id)
                color_shape_occurrences[key] = color_shape_occurrences.get(key, 0) + 1

            eeg_idx = mapping.get(log_idx)
            eeg_present = eeg_idx is not None
            if is_color_trial and eeg_present:
                eeg_epoch_index += 1

            stim_filename = _as_text(stim.get("filename"))
            stim_marker = _as_int(stim.get("marker"))
            alignment_ok = (
                _as_int(result.get("marker")) == stim_marker
                and filename == stim_filename
                and bool(_as_int(result.get("isCatch")) or 0) == bool(_as_int(stim.get("isCatch")) or 0)
            )
            row = {
                "subject": subject,
                "task": 3,
                "session": session_number,
                "source_log_file": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
                "session_trial_index_1based": (global_trial - 1) % 198 + 1,
                "global_trial_index_1based": global_trial,
                "result_index_1based": (global_trial - 1) % 198 + 1,
                "stimData_index_1based": (global_trial - 1) % 198 + 1,
                "marker": _as_int(result.get("marker")),
                "eeg_event_index_1based": eeg_idx + 1 if eeg_present else "",
                "eeg_event_type": eeg_types[eeg_idx] if eeg_present else "",
                "eeg_event_latency_samples": eeg_latencies[eeg_idx] if eeg_present else "",
                "eeg_event_present": eeg_present,
                "is_eeg_epoch": is_color_trial and eeg_present,
                "eeg_color_epoch_index_1based": eeg_epoch_index if is_color_trial and eeg_present else "",
                "is_catch": is_catch,
                "color": color,
                "shape_id": shape_id if has_shape else "",
                "shape_label": f"shape_{shape_id}" if has_shape else "",
                "color_trial_index_1based": color_occurrences.get(color, "") if is_color_trial else "",
                "color_shape_trial_index_1based": color_shape_occurrences.get((color, shape_id), "") if is_color_trial else "",
                "filename": filename,
                "category": category,
                "onset_time_s": _as_float(result.get("onsetTime")),
                "offset_time_s": _as_float(result.get("offsetTime")),
                "duration_s": _as_float(result.get("duration")),
                "response": _as_int(result.get("response")),
                "stimData_marker": stim_marker,
                "stimData_filename": stim_filename,
                "stimData_category": _as_text(stim.get("catName")),
                "log_stimData_alignment_ok": alignment_ok,
            }
            all_rows.append(row)

    index = pd.DataFrame(all_rows)
    audit = pd.DataFrame(audit_rows)
    return index, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rename", action="store_true", help="Do not rename the known-reversed raw log files")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--renames", type=Path, default=DEFAULT_RENAMES)
    args = parser.parse_args()

    if args.skip_rename:
        rename_records = infer_existing_rename_manifest()
    else:
        rename_records = normalize_labels()
    args.renames.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rename_records, columns=["subject", "old_name", "new_name", "reason"]).to_csv(
        args.renames, index=False, encoding="utf-8-sig"
    )

    index, audit = build_index()
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.index, index=False, encoding="utf-8-sig")
    audit.to_csv(args.audit, index=False, encoding="utf-8-sig")

    summaries = audit.loc[audit["operation"] == "summary"]
    print(f"renamed_files={len(rename_records)}")
    print(f"index_rows={len(index)} color_trials={int(index['is_eeg_epoch'].sum())}")
    print(summaries[["subject", "log_count", "eeg_count", "matched_count", "exact_sequence"]].to_string(index=False))
    print(f"index={args.index}")
    print(f"audit={args.audit}")
    print(f"renames={args.renames}")


if __name__ == "__main__":
    main()
