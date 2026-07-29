"""SEEG channel parsing and contact-centered rereferencing."""
from __future__ import annotations

import re
import numpy as np
import mne


def parse_contact(label: str):
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(label).strip())
    return (match.group(1), int(match.group(2))) if match else None


def add_laplacian_neighbors(labels, selected):
    available = set(labels)
    expanded = set(selected)
    for label in selected:
        parsed = parse_contact(label)
        if parsed is None:
            continue
        shaft, number = parsed
        expanded.update(x for x in (f"{shaft}{number-1}", f"{shaft}{number+1}") if x in available)
    return [label for label in labels if label in expanded]


def contact_laplacian(raw, output_labels=None, bad_channels=()):
    labels = raw.ch_names
    bad = set(bad_channels)
    index = {label: i for i, label in enumerate(labels)}
    data = raw.get_data()
    output_labels = list(output_labels or labels)
    output = []
    kept = []
    for label in output_labels:
        parsed = parse_contact(label)
        if parsed is None or label in bad:
            continue
        shaft, number = parsed
        left, right = f"{shaft}{number-1}", f"{shaft}{number+1}"
        if left not in index or right not in index or left in bad or right in bad:
            continue
        output.append(data[index[label]] - (data[index[left]] + data[index[right]]) / 2.0)
        kept.append(label)
    if not output:
        raise RuntimeError("No channel has two valid Laplacian neighbors")
    info = mne.create_info(kept, raw.info["sfreq"], ch_types="seeg")
    rereferenced = mne.io.RawArray(np.asarray(output), info, verbose=False)
    rereferenced.set_annotations(raw.annotations.copy())
    return rereferenced
