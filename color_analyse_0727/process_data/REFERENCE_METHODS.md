# Re-reference sensitivity datasets

These folders are new analysis inputs and do not replace the historical HDF5 files directly under `process_data/testXXX/`.

| folder | definition |
|---|---|
| `reference_native/` | filtered acquisition-referenced contact; no additional re-reference |
| `reference_global_car/` | contact minus the mean of all clean parseable shaft contacts in that task recording |
| `reference_shaft_car/` | contact minus the mean of all clean contacts on the same shaft |
| `reference_bipolar/` | anchor contact minus its immediate higher-numbered neighbor |
| `reference_laplacian/` | center minus the mean of its immediate lower- and higher-numbered neighbors |

Each folder contains `test001` through `test007`, with Task1/2/3 HDF5 files. Every HDF5 records `reference_method`, `reference_formula`, and per-output `reference_members`.

For cross-method scientific comparison, use only anchor contacts present in all requested methods and all three tasks. The completed Task1 color-selection comparison is under `result/rereference_color_selection_20260824/`.
