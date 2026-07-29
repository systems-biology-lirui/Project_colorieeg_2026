# SEEG raw inventory and lightweight QC

- Subjects: 6 (test001, test002, test003, test004, test005, test006)
- ERP runs: 18
- Runs with flags: 5
- Sampled channels with flags: 45
- Raw inputs modified: no

## Flagged runs

| Subject | Run | Flags |
|---|---|---|
| test003 | erp3 | excess_vs_nominal_trials |
| test004 | erp2 | missing_expected_trials |
| test004 | erp3 | missing_expected_trials |
| test005 | erp1 | missing_expected_trials |
| test005 | erp2 | boundary_event;missing_expected_trials |

## Repeated sampled channel flags

| Subject | Channel | Flagged runs |
|---|---|---:|
| test001 | F15 | 2 |
| test002 | A8 | 2 |
| test002 | G7 | 3 |
| test003 | C13 | 2 |
| test003 | D5 | 2 |
| test003 | H7 | 3 |
| test003 | I1 | 2 |
| test003 | I2 | 2 |
| test003 | I3 | 2 |
| test005 | C13 | 3 |
| test005 | C14 | 2 |
| test005 | F7 | 3 |
| test005 | I10 | 2 |

## Metadata gaps

- Missing standardized `*_ieegloc.xlsx`: test004, test005
- Missing task1 groupedData: test004, test005, test006

## Interpretation

`channel_sampled_qc.csv` uses regularly sampled continuous values. Flags are candidates for visual/full-resolution review only.
No channel or epoch is rejected by this inventory stage.
