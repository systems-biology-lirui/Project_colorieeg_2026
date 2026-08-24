"""Conservative, response-aware channel-quality audit.

This module produces candidates only. It never removes a channel and it never
changes source ``.set/.fdt`` files. A channel becomes excluded only after a
human writes an explicit decision to the review table.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .condition_registry import condition_for_trigger
from .config import KNOWN_BAD_CHANNELS, PROTECTED_CHANNELS, TARGET_SFREQ
from .io_seeg import RecordingMetadata, open_fdt
from .signal_processing import filter_continuous


AUDIT_COLUMNS = [
    "subject",
    "task_num",
    "signal_stage",
    "channel",
    "channel_index",
    "shaft",
    "contact_number",
    "known_bad_record",
    "protected_channel",
    "nan_fraction",
    "flatline_fraction",
    "saturation_fraction",
    "robust_std",
    "baseline_rms",
    "response_rms",
    "response_to_baseline_ratio",
    "line_noise_ratio_db",
    "line_noise_group_median_db",
    "line_noise_relative_db",
    "jump_fraction",
    "candidate_level",
    "candidate_reasons",
    "recommended_action",
    "manual_decision",
    "manual_comment",
]

AGGREGATED_COLUMNS = AUDIT_COLUMNS + ["task_coverage"]


def _median_abs_deviation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = np.median(finite)
    return float(np.median(np.abs(finite - median)))


def _robust_std(values: np.ndarray) -> float:
    mad = _median_abs_deviation(values)
    return float(1.4826 * mad) if np.isfinite(mad) else float("nan")


def _fraction_flat(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 1.0
    diffs = np.abs(np.diff(finite))
    scale = max(_robust_std(finite), 1e-12)
    return float(np.mean(diffs <= max(1e-8, scale * 1e-6)))


def _fraction_saturated(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    lo, hi = np.percentile(finite, [0.1, 99.9])
    if hi <= lo:
        return 0.0
    # This is intentionally conservative: only repeated exact extreme values
    # are treated as possible clipping, not large physiological responses.
    repeated_low = np.mean(finite == np.min(finite))
    repeated_high = np.mean(finite == np.max(finite))
    return float(max(repeated_low, repeated_high))


def _jump_fraction(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 1.0
    diffs = np.abs(np.diff(finite))
    center = np.median(diffs)
    spread = _robust_std(diffs)
    threshold = center + 12.0 * max(spread, 1e-12)
    return float(np.mean(diffs > threshold))


def _line_noise_ratio_db(values: np.ndarray, sfreq: float) -> float:
    """Return the largest line-noise-to-neighbor ratio in dB.

    This is a review signal, not a deletion rule. A strong line component can
    often be fixed by filtering and does not necessarily mean a bad contact.
    """

    try:
        from scipy.signal import welch
    except ImportError:
        return float("nan")

    values = values[np.isfinite(values)]
    if values.size < max(256, int(sfreq)):
        return float("nan")
    segment = values[: min(values.size, int(sfreq * 10))]
    freqs, power = welch(segment, fs=sfreq, nperseg=min(2048, segment.size))
    ratios: list[float] = []
    for target in (50.0, 100.0, 150.0):
        idx = int(np.argmin(np.abs(freqs - target)))
        neighbor = np.ones(len(power), dtype=bool)
        neighbor[max(0, idx - 2) : min(len(freqs), idx + 3)] = False
        local = power[neighbor]
        if local.size and np.median(local) > 0 and power[idx] > 0:
            ratios.append(10.0 * np.log10(power[idx] / np.median(local)))
    return float(max(ratios)) if ratios else float("nan")


def _event_response_metrics(
    channel_data: np.ndarray,
    metadata: RecordingMetadata,
    signal_sfreq: float,
    max_events: int = 120,
) -> tuple[float, float, float]:
    """Compute baseline/response RMS without using response size as badness."""

    baseline_values: list[np.ndarray] = []
    response_values: list[np.ndarray] = []
    for event in metadata.events:
        if condition_for_trigger(metadata.task_num, event.trigger) is None:
            continue
        center = int(round((event.latency_samples - 1.0) * signal_sfreq / metadata.sfreq))
        baseline_start = center + int(round(-0.2 * signal_sfreq))
        baseline_end = center
        response_start = center
        response_end = center + int(round(0.4 * signal_sfreq))
        if baseline_start < 0 or response_end > channel_data.size:
            continue
        baseline_values.append(np.asarray(channel_data[baseline_start:baseline_end], dtype=float))
        response_values.append(np.asarray(channel_data[response_start:response_end], dtype=float))
        if len(baseline_values) >= max_events:
            break

    if not baseline_values or not response_values:
        return float("nan"), float("nan"), float("nan")
    baseline = np.concatenate(baseline_values)
    response = np.concatenate(response_values)
    baseline_rms = float(np.sqrt(np.nanmean(np.square(baseline))))
    response_rms = float(np.sqrt(np.nanmean(np.square(response))))
    ratio = response_rms / max(baseline_rms, 1e-12)
    return baseline_rms, response_rms, float(ratio)


def _contact_parts(label: str) -> tuple[str, str]:
    text = label.strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    shaft = text[: len(text) - len(digits)] if digits else text
    return shaft, digits


def _candidate_reasons(row: dict[str, object]) -> tuple[str, str]:
    reasons: list[str] = []
    hard = False
    if float(row["nan_fraction"]) > 0:
        reasons.append("nonfinite_samples")
        hard = True
    if float(row["flatline_fraction"]) >= 0.20:
        reasons.append("long_flatline")
        hard = True
    if float(row["saturation_fraction"]) >= 0.01:
        reasons.append("possible_clipping")
        hard = True
    if float(row["jump_fraction"]) >= 0.005:
        reasons.append("repeated_abrupt_jumps")
    baseline = float(row["baseline_rms"])
    if np.isfinite(baseline) and baseline <= 1e-9:
        reasons.append("near_zero_baseline")
        hard = True
    if bool(row["known_bad_record"]):
        reasons.append("listed_in_previous_bad_channel_record")
        hard = True
    if bool(row["protected_channel"]):
        reasons.append("protected_functional_channel")

    if bool(row["protected_channel"]):
        level = "protected_review" if reasons else "protected"
    elif hard:
        level = "high_confidence_candidate"
    elif reasons:
        level = "review_candidate"
    else:
        level = "normal"
    return level, ";".join(reasons)


def audit_recording(
    metadata: RecordingMetadata,
    sample_step: int = 10,
    apply_filter: bool = False,
    batch_size: int = 8,
) -> list[dict[str, object]]:
    """Audit all channels in one recording and return candidate rows.

    With ``apply_filter=True``, each batch is processed with the exact final
    resampling, band-pass and 50/100/150 Hz notch chain used for HDF5 export.
    """

    data = open_fdt(metadata)
    rows: list[dict[str, object]] = []
    known = {label.upper() for label in KNOWN_BAD_CHANNELS.get(metadata.subject, ())}
    protected = {label.upper() for label in PROTECTED_CHANNELS.get(metadata.subject, ())}

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    signal_stage = "filtered_1_200Hz_notched" if apply_filter else "raw_unfiltered"
    signal_sfreq = TARGET_SFREQ if apply_filter else metadata.sfreq

    for batch_start in range(0, metadata.n_channels, batch_size):
        batch_end = min(metadata.n_channels, batch_start + batch_size)
        batch = np.asarray(data[batch_start:batch_end, :], dtype=np.float64)
        if apply_filter:
            batch = filter_continuous(batch, metadata.sfreq)

        for local_index, channel_index in enumerate(range(batch_start, batch_end)):
            label = metadata.labels[channel_index]
            channel_data = batch[local_index]
            signal = np.asarray(channel_data[::sample_step], dtype=np.float64)
            line_noise_signal = np.asarray(
                channel_data[: min(channel_data.size, int(signal_sfreq * 10))],
                dtype=np.float64,
            )
            finite = signal[np.isfinite(signal)]
            baseline_rms, response_rms, response_ratio = _event_response_metrics(
                channel_data, metadata, signal_sfreq
            )
            shaft, contact_number = _contact_parts(label)
            row: dict[str, object] = {
                "subject": metadata.subject,
                "task_num": metadata.task_num,
                "signal_stage": signal_stage,
                "channel": label,
                "channel_index": channel_index,
                "shaft": shaft,
                "contact_number": contact_number,
                "known_bad_record": label.upper() in known,
                "protected_channel": label.upper() in protected,
                "nan_fraction": float(1.0 - (finite.size / max(signal.size, 1))),
                "flatline_fraction": _fraction_flat(signal),
                "saturation_fraction": _fraction_saturated(signal),
                "robust_std": _robust_std(signal),
                "baseline_rms": baseline_rms,
                "response_rms": response_rms,
                "response_to_baseline_ratio": response_ratio,
                "line_noise_ratio_db": _line_noise_ratio_db(line_noise_signal, signal_sfreq),
                "jump_fraction": _jump_fraction(signal),
                "manual_decision": "",
                "manual_comment": "",
            }
            level, reasons = _candidate_reasons(row)
            row["candidate_level"] = level
            row["candidate_reasons"] = reasons
            row["recommended_action"] = "review" if reasons else "keep_unless_manual_evidence"
            rows.append(row)

    # Absolute line-noise power is often shared by the entire recording. Use
    # only a robust within-recording outlier rule for candidate labeling, while
    # retaining the absolute value in the table for manual inspection.
    line_values = np.asarray(
        [float(row["line_noise_ratio_db"]) for row in rows], dtype=float
    )
    finite_line = line_values[np.isfinite(line_values)]
    if finite_line.size:
        group_median = float(np.median(finite_line))
        group_mad = _median_abs_deviation(finite_line)
        threshold = group_median + max(8.0, 4.0 * 1.4826 * max(group_mad, 1e-12))
        for row in rows:
            line_value = float(row["line_noise_ratio_db"])
            row["line_noise_group_median_db"] = group_median
            row["line_noise_relative_db"] = (
                line_value - group_median if np.isfinite(line_value) else float("nan")
            )
            if np.isfinite(line_value) and line_value > threshold:
                reasons = str(row["candidate_reasons"])
                reasons = ";".join(filter(None, [reasons, "relative_line_noise_outlier"]))
                row["candidate_reasons"] = reasons
                if row["protected_channel"]:
                    row["candidate_level"] = "protected_review"
                elif row["candidate_level"] == "normal":
                    row["candidate_level"] = "review_candidate"
                row["recommended_action"] = "review"
    return rows


def write_review_table(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    """Write a human-editable review table with blank decisions."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_channel_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse task-level rows to one review row per subject/channel."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["subject"]), str(row["channel"]).upper())
        grouped.setdefault(key, []).append(row)

    level_priority = {
        "normal": 0,
        "review_candidate": 1,
        "protected": 2,
        "protected_review": 3,
        "high_confidence_candidate": 4,
    }

    def finite_values(items: list[dict[str, object]], key: str) -> list[float]:
        values: list[float] = []
        for item in items:
            try:
                value = float(item[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    aggregate: list[dict[str, object]] = []
    for (subject, channel), items in sorted(grouped.items()):
        levels = [str(item["candidate_level"]) for item in items]
        level = max(levels, key=lambda item: level_priority.get(item, -1))
        reasons = sorted(
            {
                reason
                for item in items
                for reason in str(item["candidate_reasons"]).split(";")
                if reason
            }
        )
        row: dict[str, object] = {
            "subject": subject,
            "task_num": "all",
            "signal_stage": items[0]["signal_stage"],
            "channel": channel,
            "channel_index": items[0]["channel_index"],
            "shaft": items[0]["shaft"],
            "contact_number": items[0]["contact_number"],
            "known_bad_record": any(bool(item["known_bad_record"]) for item in items),
            "protected_channel": any(bool(item["protected_channel"]) for item in items),
            "nan_fraction": max(float(item["nan_fraction"]) for item in items),
            "flatline_fraction": max(float(item["flatline_fraction"]) for item in items),
            "saturation_fraction": max(float(item["saturation_fraction"]) for item in items),
            "robust_std": max(finite_values(items, "robust_std"), default=float("nan")),
            "baseline_rms": float(np.median(finite_values(items, "baseline_rms"))) if finite_values(items, "baseline_rms") else float("nan"),
            "response_rms": float(np.median(finite_values(items, "response_rms"))) if finite_values(items, "response_rms") else float("nan"),
            "response_to_baseline_ratio": float(np.median(finite_values(items, "response_to_baseline_ratio"))) if finite_values(items, "response_to_baseline_ratio") else float("nan"),
            "line_noise_ratio_db": max(finite_values(items, "line_noise_ratio_db"), default=float("nan")),
            "line_noise_group_median_db": float(np.median(finite_values(items, "line_noise_group_median_db"))) if finite_values(items, "line_noise_group_median_db") else float("nan"),
            "line_noise_relative_db": max(finite_values(items, "line_noise_relative_db"), default=float("nan")),
            "jump_fraction": max(float(item["jump_fraction"]) for item in items),
            "candidate_level": level,
            "candidate_reasons": ";".join(reasons),
            "recommended_action": "review" if reasons else "keep_unless_manual_evidence",
            "manual_decision": "",
            "manual_comment": "",
            "task_coverage": ",".join(sorted({str(item["task_num"]) for item in items})),
        }
        aggregate.append(row)
    return aggregate


def write_aggregated_review_table(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    """Write a compact recording-level summary for quick inspection."""

    records: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (str(row["subject"]), int(row["task_num"]))
        item = records.setdefault(
            key,
            {
                "subject": key[0],
                "task_num": key[1],
                "channels": 0,
                "high_confidence_candidates": 0,
                "review_candidates": 0,
                "protected_channels": 0,
                "known_bad_records": 0,
            },
        )
        item["channels"] = int(item["channels"]) + 1
        if row["candidate_level"] == "high_confidence_candidate":
            item["high_confidence_candidates"] = int(item["high_confidence_candidates"]) + 1
        if row["candidate_level"] == "review_candidate":
            item["review_candidates"] = int(item["review_candidates"]) + 1
        if row["protected_channel"]:
            item["protected_channels"] = int(item["protected_channels"]) + 1
        if row["known_bad_record"]:
            item["known_bad_records"] = int(item["known_bad_records"]) + 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out = list(records.values())
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        if not rows_out:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
