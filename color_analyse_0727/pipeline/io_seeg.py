"""Read EEGLAB ``.set/.fdt`` recordings without changing source files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .condition_registry import normalize_trigger
from .config import SUBJECTS, TASKS, recording_paths


@dataclass(frozen=True)
class Event:
    trigger: str
    latency_samples: float


@dataclass(frozen=True)
class RecordingMetadata:
    subject: str
    task_num: int
    set_path: Path
    fdt_path: Path
    sfreq: float
    n_channels: int
    n_points: int
    labels: tuple[str, ...]
    events: tuple[Event, ...]


def clean_channel_name(value: object) -> str:
    text = _to_text(value)
    return text.strip().upper().replace(" ", "")


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").replace("\x00", "")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.ravel()).replace("\x00", "")
        return _to_text(value.ravel()[0])
    if isinstance(value, (list, tuple)):
        return _to_text(value[0]) if value else ""
    return str(value)


def _scalar(value: object) -> float:
    array = np.asarray(value).squeeze()
    if array.size == 0:
        raise ValueError("Expected a scalar value")
    return float(array.reshape(-1)[0])


def _mat_struct_field(struct: object, field: str, default: object = None) -> object:
    if hasattr(struct, field):
        return getattr(struct, field)
    if isinstance(struct, np.void) and struct.dtype.names and field in struct.dtype.names:
        return struct[field]
    return default


def _load_set_scipy(set_path: Path) -> tuple[float, int, int, tuple[str, ...], tuple[Event, ...]]:
    import scipy.io as sio

    mat = sio.loadmat(set_path, squeeze_me=False, struct_as_record=False)
    eeg = mat.get("EEG")
    if eeg is None:
        raise ValueError("No EEG struct found")
    eeg = eeg.reshape(-1)[0]

    sfreq = _scalar(_mat_struct_field(eeg, "srate"))
    n_channels = int(_scalar(_mat_struct_field(eeg, "nbchan")))
    n_points = int(_scalar(_mat_struct_field(eeg, "pnts")))

    raw_chanlocs = _mat_struct_field(eeg, "chanlocs", [])
    labels: list[str] = []
    for item in np.asarray(raw_chanlocs).reshape(-1):
        labels.append(clean_channel_name(_mat_struct_field(item, "labels", "")))
    labels = labels[:n_channels]
    if len(labels) != n_channels:
        labels = [f"CH{i + 1}" for i in range(n_channels)]

    raw_events = _mat_struct_field(eeg, "event", [])
    events: list[Event] = []
    for item in np.asarray(raw_events).reshape(-1):
        trigger = normalize_trigger(_mat_struct_field(item, "type", ""))
        latency_value = _mat_struct_field(item, "latency", None)
        if latency_value is None or not trigger:
            continue
        try:
            latency = _scalar(latency_value)
        except (TypeError, ValueError):
            continue
        events.append(Event(trigger=trigger, latency_samples=latency))
    return sfreq, n_channels, n_points, tuple(labels), tuple(events)


def _h5_text(h5: Any, value: object) -> str:
    """Decode an HDF5 scalar, char array, or object reference."""

    if isinstance(value, h5py.Reference):
        return _h5_text(h5, h5[value][()])
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        if value.size == 0:
            return ""
        return _h5_text(h5, value.reshape(-1)[0])
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        flat = value.reshape(-1)
        if flat.size and np.all((flat >= 0) & (flat <= 0x10FFFF)):
            try:
                return "".join(chr(int(code)) for code in flat if int(code) != 0)
            except (ValueError, OverflowError):
                pass
    return _to_text(value)


def _load_set_h5(set_path: Path) -> tuple[float, int, int, tuple[str, ...], tuple[Event, ...]]:
    global h5py
    import h5py  # type: ignore

    with h5py.File(set_path, "r") as h5:
        sfreq = _scalar(h5["srate"][()])
        n_channels = int(_scalar(h5["nbchan"][()]))
        n_points = int(_scalar(h5["pnts"][()]))

        labels: list[str] = []
        chanlocs = h5.get("chanlocs")
        if chanlocs is not None and "labels" in chanlocs:
            for ref in chanlocs["labels"][()].reshape(-1):
                labels.append(clean_channel_name(_h5_text(h5, ref)))
        if len(labels) != n_channels:
            labels = [f"CH{i + 1}" for i in range(n_channels)]

        events: list[Event] = []
        event_group = h5.get("event")
        if event_group is not None and "type" in event_group and "latency" in event_group:
            raw_types = event_group["type"][()].reshape(-1)
            raw_latencies = event_group["latency"][()].reshape(-1)
            for raw_type, raw_latency in zip(raw_types, raw_latencies):
                trigger = normalize_trigger(_h5_text(h5, raw_type))
                try:
                    latency = _scalar(
                        h5[raw_latency][()] if isinstance(raw_latency, h5py.Reference) else raw_latency
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                if trigger:
                    events.append(Event(trigger=trigger, latency_samples=latency))
    return sfreq, n_channels, n_points, tuple(labels), tuple(events)


def load_set_metadata(subject: str, task_num: int) -> RecordingMetadata:
    set_path, fdt_path = recording_paths(subject, task_num)
    if not set_path.exists():
        raise FileNotFoundError(set_path)
    if not fdt_path.exists():
        raise FileNotFoundError(fdt_path)

    try:
        values = _load_set_scipy(set_path)
    except (NotImplementedError, ValueError, TypeError, ImportError):
        values = _load_set_h5(set_path)

    sfreq, n_channels, n_points, labels, events = values
    expected_bytes = n_channels * n_points * np.dtype(np.float32).itemsize
    actual_bytes = fdt_path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"FDT is smaller than metadata expects: {fdt_path} "
            f"({actual_bytes} < {expected_bytes} bytes)"
        )
    return RecordingMetadata(
        subject=subject,
        task_num=int(task_num),
        set_path=set_path,
        fdt_path=fdt_path,
        sfreq=sfreq,
        n_channels=n_channels,
        n_points=n_points,
        labels=labels,
        events=events,
    )


def open_fdt(metadata: RecordingMetadata) -> np.memmap:
    """Open source data as a read-only Fortran-order memory map."""

    return np.memmap(
        metadata.fdt_path,
        dtype=np.float32,
        mode="r",
        shape=(metadata.n_channels, metadata.n_points),
        order="F",
    )


def scan_recordings(
    subjects: tuple[str, ...] = SUBJECTS,
    tasks: tuple[int, ...] = TASKS,
) -> list[RecordingMetadata]:
    recordings: list[RecordingMetadata] = []
    for subject in subjects:
        for task_num in tasks:
            recordings.append(load_set_metadata(subject, task_num))
    return recordings

