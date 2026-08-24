"""Small, reusable readers and validators for the rebuilt HDF5 format."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def inspect_hdf5(path: Path) -> dict[str, object]:
    import h5py

    with h5py.File(path, "r") as h5:
        labels = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["labels"][()]]
        conditions = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["condition_names"][()]
        ]
        triggers = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["condition_triggers"][()]
        ]
        trial_counts = [int(value) for value in h5["trial_counts"][()]]
        datasets = {
            name: tuple(h5["epochs"][name].shape)
            for name in h5["epochs"]
        }
        return {
            "path": str(path),
            "subject": str(h5.attrs["subject"]),
            "task_num": int(h5.attrs["task_num"]),
            "n_channels": len(labels),
            "labels": labels,
            "n_times": int(h5["time_ms"].shape[0]),
            "time_start_ms": float(h5["time_ms"][0]),
            "time_end_ms": float(h5["time_ms"][-1]),
            "condition_names": conditions,
            "condition_triggers": triggers,
            "trial_counts": trial_counts,
            "epoch_shapes": datasets,
        }


def validate_hdf5_file(path: Path) -> list[str]:
    """Return validation errors; an empty list means the file passed."""

    import h5py

    errors: list[str] = []
    with h5py.File(path, "r") as h5:
        required = {
            "time_ms",
            "labels",
            "condition_names",
            "condition_triggers",
            "trial_counts",
            "epochs",
        }
        missing = sorted(required.difference(h5.keys()))
        if missing:
            errors.append(f"missing datasets/groups: {missing}")
            return errors

        time_ms = np.asarray(h5["time_ms"][()])
        if time_ms.size != 750:
            errors.append(f"expected 750 time points, got {time_ms.size}")
        if time_ms.size and (not np.isclose(time_ms[0], -500.0) or not np.isclose(time_ms[-1], 998.0)):
            errors.append(f"unexpected time axis: {time_ms[0]}..{time_ms[-1]}")

        labels = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["labels"][()]]
        if len(labels) != len(set(labels)):
            errors.append("duplicate output channel labels")

        names = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["condition_names"][()]]
        triggers = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["condition_triggers"][()]]
        counts = [int(value) for value in h5["trial_counts"][()]]
        if not (len(names) == len(triggers) == len(counts)):
            errors.append("condition metadata lengths do not match")

        for name, count in zip(names, counts):
            if count <= 0:
                errors.append(f"condition has no epochs: {name}")
                continue
            if name not in h5["epochs"]:
                errors.append(f"missing epoch dataset: {name}")
                continue
            dataset = h5["epochs"][name]
            expected_shape = (count, len(labels), 750)
            if tuple(dataset.shape) != expected_shape:
                errors.append(f"{name} shape {tuple(dataset.shape)} != {expected_shape}")
            sample = np.asarray(dataset[: min(count, 3)])
            if not np.isfinite(sample).all():
                errors.append(f"non-finite values in {name}")
    return errors


def load_condition_epochs(path: Path, condition: str) -> np.ndarray:
    """Load one condition as ``(trials, channels, time)``."""

    import h5py

    with h5py.File(path, "r") as h5:
        if condition not in h5["epochs"]:
            raise KeyError(f"Condition not found in {path}: {condition}")
        return np.asarray(h5["epochs"][condition][()])
