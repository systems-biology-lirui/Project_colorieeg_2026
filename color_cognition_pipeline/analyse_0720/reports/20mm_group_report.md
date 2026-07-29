# 20-mm fMRI color-patch SEEG batch report

## Scope

This batch uses both fMRI target coordinates and includes every localization file found for test001–test006. The primary spatial criterion is the nearest left/right fMRI peak within 20 mm. Subject-level artifacts are isolated under `result_20mm/subjects/<subject>/`; group artifacts are under `result_20mm/group/`.

## Coverage and eligibility

Only test001 and test003 have electrodes within 20 mm of either target. The number of localization candidates is 24 and 19, respectively. After removing confirmed/repeated high-variance channels and contacts that cannot form a valid three-contact Laplacian, the clean feature matrices contain 21 and 15 channels. test002, test005 and test006 have no 20-mm candidates; test004 has no localization file. They are retained in the coverage table but excluded from signal-level group statistics.

## Preprocessing

Each eligible subject was processed run-by-run from `seegdata`: notch filtering, ERP 1–30 Hz filtering and 500 Hz resampling, contact-centered Laplacian rereferencing, robust epoch QC, and 8-band 70–150 Hz high-gamma extraction. ERP and HG share the same epoch keep mask. Confirmed/repeated bad channels were excluded before rereferencing (test001 F15; test003 C13/D5/H7/I1/I2/I3).

## Decoding

Task2 memory-color, Task3 red-green, and Task2 true–false/cross-task decoding were run separately for ERP and HG. The fast exploratory null uses 100 label permutations, fixed CV splits, precomputed time-window features, and positive cluster correction. Each subject also has a PNG and PDF curve with the permutation null band.

## Group interpretation

Group curves are descriptive mean ± subject SEM with individual subject traces. The current signal-level sample is n=2, so sign-flip summaries are exploratory and should not be presented as population-level confirmation. The principal next step is to obtain localization files or revise the spatial inclusion rule for additional subjects before confirmatory statistics.

All numerical tables are in `result_20mm/group/tables/`; publication-style figures are in `result_20mm/group/figures/`.
