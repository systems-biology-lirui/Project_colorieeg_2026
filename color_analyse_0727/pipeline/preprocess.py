"""Reviewed-channel preprocessing and HDF5 export.

The exporter is deliberately gated by a human-editable review CSV. Blank
decisions mean keep; only an explicit exclusion decision removes a channel.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .condition_registry import TASK_CONDITIONS, condition_for_trigger
from .config import (
    BANDPASS_HZ,
    EPOCH_TMAX_MS,
    EPOCH_TMIN_MS,
    NOTCH_HZ,
    NOTCH_Q,
    PROCESS_DATA_ROOT,
    TARGET_SFREQ,
    MANUAL_DECISIONS_PATH,
    METADATA_ROOT,
)
from .electrode_manifest import load_task_analysis_centers
from .io_seeg import RecordingMetadata, open_fdt
from .signal_processing import filter_continuous


EXCLUDE_DECISIONS = {"exclude", "bad", "confirmed_bad", "remove", "yes", "1", "true"}

REFERENCE_METHODS = (
    "native",
    "global_car",
    "shaft_car",
    "bipolar",
    "laplacian",
)


@dataclass(frozen=True)
class ReferenceSpec:
    """One output channel expressed as a weighted sum of source contacts."""

    label: str
    shaft: str
    source_indices: tuple[int, ...]
    weights: tuple[float, ...]
    members: tuple[str, ...]


def load_review_decisions(review_csv: Path, subject: str) -> set[str]:
    """Read explicit manual exclusions for one subject.

    Previous bad-channel records are not applied automatically. They appear in
    the audit table and require the same explicit manual decision as any other
    candidate.
    """

    if not review_csv.exists():
        raise FileNotFoundError(
            f"Review file not found: {review_csv}. Run the audit and review it before export."
        )
    excluded: set[str] = set()
    decision_files = (review_csv, MANUAL_DECISIONS_PATH)
    for decision_file in decision_files:
        if not decision_file.exists():
            continue
        with decision_file.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("subject", "")).strip() != subject:
                    continue
                label = str(row.get("channel", "")).strip().upper().replace(" ", "")
                decision = str(
                    row.get("manual_decision", row.get("decision", ""))
                ).strip().lower()
                if label and decision in EXCLUDE_DECISIONS:
                    if str(row.get("protected_channel", "")).strip().lower() in {"true", "1", "yes"}:
                        raise ValueError(f"Protected channel cannot be excluded: {subject} {label}")
                    excluded.add(label)
    return excluded


def _filter_continuous(data: np.ndarray, sfreq: float) -> np.ndarray:
    return filter_continuous(data, sfreq)


def _resampled_length(metadata: RecordingMetadata) -> int:
    return int(round(metadata.n_points * TARGET_SFREQ / metadata.sfreq))


def _parsed_contacts(labels: tuple[str, ...]) -> dict[str, tuple[str, int]]:
    parsed: dict[str, tuple[str, int]] = {}
    for label in labels:
        match = re.fullmatch(r"([A-Z]+)(\d+)", label.upper())
        if match:
            parsed[label.upper()] = (match.group(1), int(match.group(2)))
    return parsed


def _reference_specs(
    labels: tuple[str, ...],
    excluded: set[str],
    allowed_centers: set[str] | None = None,
    reference_method: str = "laplacian",
) -> list[ReferenceSpec]:
    if reference_method not in REFERENCE_METHODS:
        raise ValueError(
            f"Unknown reference method {reference_method!r}; expected one of {REFERENCE_METHODS}"
        )
    positions = {label.upper(): idx for idx, label in enumerate(labels)}
    parsed = _parsed_contacts(labels)
    clean_by_shaft: dict[str, list[str]] = {}
    for key, (shaft, number) in parsed.items():
        if key not in excluded:
            clean_by_shaft.setdefault(shaft, []).append(key)
    for shaft in clean_by_shaft:
        clean_by_shaft[shaft].sort(key=lambda key: parsed[key][1])

    specs: list[ReferenceSpec] = []
    for label in labels:
        key = label.upper()
        if key in excluded or key not in parsed:
            continue
        if allowed_centers is not None and key not in allowed_centers:
            continue
        shaft, number = parsed[key]
        center_idx = positions[key]
        if reference_method in {"native", "global_car"}:
            specs.append(
                ReferenceSpec(key, shaft, (center_idx,), (1.0,), (key,))
            )
        elif reference_method == "shaft_car":
            members = tuple(clean_by_shaft.get(shaft, ()))
            if len(members) < 2:
                continue
            indices = tuple(positions[member] for member in members)
            weights = tuple(
                (1.0 if member == key else 0.0) - 1.0 / len(members)
                for member in members
            )
            specs.append(ReferenceSpec(key, shaft, indices, weights, members))
        elif reference_method == "bipolar":
            right = f"{shaft}{number + 1}"
            if right not in positions or right in excluded:
                continue
            specs.append(
                ReferenceSpec(
                    key,
                    shaft,
                    (center_idx, positions[right]),
                    (1.0, -1.0),
                    (key, right),
                )
            )
        else:
            left = f"{shaft}{number - 1}"
            right = f"{shaft}{number + 1}"
            if left not in positions or right not in positions:
                continue
            if left in excluded or right in excluded:
                continue
            specs.append(
                ReferenceSpec(
                    key,
                    shaft,
                    (center_idx, positions[left], positions[right]),
                    (1.0, -0.5, -0.5),
                    (key, left, right),
                )
            )
    return specs


def _global_car_reference(
    source: np.ndarray,
    labels: tuple[str, ...],
    excluded: set[str],
    sfreq: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the filtered mean of all clean shaft contacts.

    Referencing and the shared linear filter chain commute. Computing the raw
    mean first avoids materializing the complete filtered recording in RAM.
    """

    parsed = _parsed_contacts(labels)
    members = tuple(label.upper() for label in labels if label.upper() in parsed and label.upper() not in excluded)
    if len(members) < 2:
        raise ValueError("Global CAR requires at least two clean shaft contacts")
    positions = {label.upper(): idx for idx, label in enumerate(labels)}
    mean_signal = np.zeros(source.shape[1], dtype=np.float64)
    for member in members:
        mean_signal += np.asarray(source[positions[member]], dtype=np.float64)
    mean_signal /= float(len(members))
    return _filter_continuous(mean_signal[None, :], sfreq)[0], members


def _valid_events(metadata: RecordingMetadata) -> list[tuple[str, int]]:
    events: list[tuple[str, int]] = []
    for event in metadata.events:
        condition = condition_for_trigger(metadata.task_num, event.trigger)
        if condition is None:
            continue
        center = int(round((event.latency_samples - 1.0) * TARGET_SFREQ / metadata.sfreq))
        start = center + int(round(EPOCH_TMIN_MS * TARGET_SFREQ / 1000.0))
        end = center + int(round(EPOCH_TMAX_MS * TARGET_SFREQ / 1000.0))
        if start >= 0 and end <= _resampled_length(metadata):
            events.append((condition.name, center))
    return events


def _epoch(signal_1d: np.ndarray, center: int) -> np.ndarray | None:
    start = center + int(round(EPOCH_TMIN_MS * TARGET_SFREQ / 1000.0))
    end = center + int(round(EPOCH_TMAX_MS * TARGET_SFREQ / 1000.0))
    if start < 0 or end > signal_1d.size:
        return None
    expected = int(round((EPOCH_TMAX_MS - EPOCH_TMIN_MS) * TARGET_SFREQ / 1000.0))
    segment = np.asarray(signal_1d[start:end], dtype=np.float32)
    if segment.size != expected:
        return None
    return segment


def export_recording(
    metadata: RecordingMetadata,
    review_csv: Path,
    output_root: Path = PROCESS_DATA_ROOT,
    manifest_csv: Path | None = None,
    reference_method: str = "laplacian",
) -> Path:
    """Export one reviewed recording to HDF5."""

    import h5py

    excluded = load_review_decisions(review_csv, metadata.subject)
    manifest_csv = manifest_csv or (METADATA_ROOT / "electrode_manifest_by_task.csv")
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Electrode manifest not found: {manifest_csv}. "
            "Run scripts/build_electrode_manifest.py first."
        )
    allowed_centers = load_task_analysis_centers(manifest_csv, metadata.subject, metadata.task_num)
    specs = _reference_specs(
        metadata.labels,
        excluded,
        allowed_centers,
        reference_method=reference_method,
    )
    output_labels = [spec.label for spec in specs]
    if not output_labels:
        raise ValueError(
            f"No {reference_method} output channels remain: {metadata.subject} task{metadata.task_num}"
        )

    source = open_fdt(metadata)
    events = _valid_events(metadata)
    channel_epochs: dict[str, list[list[np.ndarray]]] = {
        condition.name: [[] for _ in output_labels]
        for condition in TASK_CONDITIONS[metadata.task_num]
    }

    # Process one shaft-sized block at a time. This avoids loading a complete
    # long recording into RAM and keeps the raw source memory-mapped/read-only.
    global_reference = None
    global_members: tuple[str, ...] = ()
    if reference_method == "global_car":
        global_reference, global_members = _global_car_reference(
            source, metadata.labels, excluded, metadata.sfreq
        )

    source_groups: dict[str, set[int]] = {}
    for spec in specs:
        source_groups.setdefault(spec.shaft, set()).update(spec.source_indices)

    for shaft, source_indices_set in source_groups.items():
        source_indices = tuple(sorted(source_indices_set))
        block = np.asarray(source[list(source_indices), :], dtype=np.float64)
        block = _filter_continuous(block, metadata.sfreq)
        index_map = {source_index: local_index for local_index, source_index in enumerate(source_indices)}
        for output_index, spec in enumerate(specs):
            if spec.shaft != shaft or any(index not in index_map for index in spec.source_indices):
                continue
            referenced = np.zeros(block.shape[-1], dtype=np.float64)
            for source_index, weight in zip(spec.source_indices, spec.weights):
                referenced += weight * block[index_map[source_index]]
            if global_reference is not None:
                referenced -= global_reference
            for condition_name, center in events:
                epoch = _epoch(referenced, center)
                if epoch is not None:
                    channel_epochs[condition_name][output_index].append(epoch)

    subject_dir = output_root / metadata.subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"task{metadata.task_num}_epoched_1_200Hz.h5"
    time_ms = np.arange(
        int(round((EPOCH_TMAX_MS - EPOCH_TMIN_MS) * TARGET_SFREQ / 1000.0)),
        dtype=np.float64,
    ) * (1000.0 / TARGET_SFREQ) + EPOCH_TMIN_MS

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "color_analyse_0727_reviewed_hdf5_v1"
        h5.attrs["subject"] = metadata.subject
        h5.attrs["task_num"] = metadata.task_num
        h5.attrs["source_set"] = str(metadata.set_path)
        h5.attrs["source_fdt"] = str(metadata.fdt_path)
        h5.attrs["electrode_manifest"] = str(manifest_csv)
        h5.attrs["source_sfreq_hz"] = metadata.sfreq
        h5.attrs["target_sfreq_hz"] = TARGET_SFREQ
        h5.attrs["bandpass_hz"] = BANDPASS_HZ
        h5.attrs["notch_hz"] = NOTCH_HZ
        h5.attrs["excluded_channels"] = ",".join(sorted(excluded))
        h5.attrs["reference_method"] = reference_method
        h5.attrs["reference_formula"] = {
            "native": "filtered acquisition-referenced contact",
            "global_car": "contact minus mean of all clean shaft contacts",
            "shaft_car": "contact minus mean of all clean contacts on the same shaft",
            "bipolar": "anchor contact minus its immediate higher-numbered neighbor",
            "laplacian": "center minus mean of immediate lower- and higher-numbered neighbors",
        }[reference_method]
        h5.attrs["global_car_members"] = ",".join(global_members)
        h5.create_dataset("time_ms", data=time_ms)
        h5.create_dataset("labels", data=np.asarray(output_labels, dtype=object), dtype=h5py.string_dtype())
        h5.create_dataset(
            "reference_members",
            data=np.asarray(["|".join(spec.members) for spec in specs], dtype=object),
            dtype=h5py.string_dtype(),
        )
        h5.create_dataset("excluded_channels", data=np.asarray(sorted(excluded), dtype=object), dtype=h5py.string_dtype())
        h5.create_dataset(
            "condition_names",
            data=np.asarray([condition.name for condition in TASK_CONDITIONS[metadata.task_num]], dtype=object),
            dtype=h5py.string_dtype(),
        )
        h5.create_dataset(
            "condition_triggers",
            data=np.asarray([condition.trigger for condition in TASK_CONDITIONS[metadata.task_num]], dtype=object),
            dtype=h5py.string_dtype(),
        )

        epochs_group = h5.create_group("epochs")
        trial_counts: dict[str, int] = {}
        for condition in TASK_CONDITIONS[metadata.task_num]:
            name = condition.name
            per_channel = channel_epochs[name]
            counts = [len(values) for values in per_channel]
            if not counts or min(counts) == 0:
                trial_counts[name] = 0
                continue
            if len(set(counts)) != 1:
                raise ValueError(f"Unequal channel trial counts in {metadata.subject} task{metadata.task_num} {name}: {counts}")
            array = np.stack([np.stack(values, axis=0) for values in per_channel], axis=1)
            epochs_group.create_dataset(name, data=array, compression="gzip", compression_opts=4)
            trial_counts[name] = int(array.shape[0])

        h5.create_dataset("trial_counts", data=np.asarray([trial_counts[c.name] for c in TASK_CONDITIONS[metadata.task_num]], dtype=np.int64))
    return output_path
