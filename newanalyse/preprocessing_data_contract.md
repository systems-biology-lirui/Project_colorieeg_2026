# SEEG preprocessing data contract (restart version)

This contract defines the boundary between raw acquisition data, preprocessing,
and analysis notebooks. Existing `processed_data` files remain legacy inputs and
must not be overwritten while the restarted pipeline is being validated.

## 1. Immutable inputs

- Raw input: `seegdata/testN/erp{1,2,3}.set` plus its matching `.fdt`.
- Raw files are read-only. Every derived dataset records source relative path,
  file size, sampling rate, channel list, and event counts.
- Analysis subject IDs use `test001` style; raw directories may use `test1`.

## 2. Required trial table

Each retained or rejected trial must have one row with at least:

| Field | Meaning |
|---|---|
| `subject` | Canonical subject ID |
| `task` / `run` | Analysis task and source run |
| `condition` / `trigger` | Exact experimental condition and raw trigger |
| `trial_id` | Stable ID within subject and run |
| `event_sample_raw` | Event latency at the original sampling rate |
| `original_repeat_index` | Repeat index before rejection |
| `keep` | Whether the trial enters analysis |
| `bad_reason` | Explicit rejection reason; empty when retained |
| `boundary_overlap` | Whether the epoch crosses a discontinuity |

Unequal trial counts are preserved. Conditions must not be truncated to the
minimum count merely to create a rectangular array.

## 3. Required channel table

Every channel has a stable row containing raw label, shaft, contact number,
channel type, anatomical coordinates/ROI when available, clinical exclusion,
automatic QC metrics, manual review status, and final keep/exclude decision.
Bad channels are resolved before rereferencing and cannot serve as neighbors.

## 4. Rereferencing

One spatial transform is used per exported modality and recorded explicitly.
The main contact-centered branch should not mix Laplacian interior contacts,
bipolar endpoints, and unreferenced isolated contacts in one channel axis.

- Contact-centered option: 1-D Laplacian for contacts with two valid neighbors;
  endpoints and contacts adjacent to bad/missing contacts are marked unavailable.
- Sensitivity option: bipolar pairs exported separately with pair names and pair
  coordinates. They are not silently substituted for Laplacian endpoints.

## 5. Signal branches

All branches derive from a versioned cleaned continuous master and record filter
type, order, transition bandwidth, software version, sampling rate, and units.

- ERP: filtered time-domain signal; baseline interval is stored in metadata.
- Wideband: time-domain signal and must not be called a time-frequency map.
- High gamma: one versioned definition of bands, line-noise handling, power
  transform, smoothing, and baseline normalization.
- Analyses extending to 200 Hz retain a sampling rate/filter design with enough
  guard band; they do not inherit a 500 Hz setting without validation.

Baseline correction occurs once. Downstream loaders must read the recorded
baseline state instead of applying a default `time < 0` correction.

## 6. Output layout

Restarted outputs should be written outside legacy subject folders until parity
checks are complete, for example:

```text
derived_restart/
  inventory/
  continuous_clean/
  epochs/
  features/
  provenance/
```

Every output includes a configuration snapshot, source manifest, software
versions, QC decisions, and a content/schema version.
