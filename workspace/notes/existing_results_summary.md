# Existing analysis results shown in Canvas

This note documents the representative figures copied from the existing analysis output. No historical analysis was rerun.

Source result directory:

```text
color_analyse_0727/result/final_analysis_seeg_20260806_corrected/
```

## Representative figure policy

The main Canvas shows one or two figures only when a stage contains two scientifically different outcomes. Full-resolution source files remain in the original result directory; the copies below exist only because the Obsidian Vault is `workspace/`.

| Stage | Canvas figure | Why it is representative | Interpretation limit |
|---|---|---|---|
| 01 Selection | `stage01_patch_csc_mni.png` | Shows the MNI projections, color-patch priors, and selected spatial set in one view. | CSC is an exploratory spatial-prior intersection, not an individual fMRI validation. |
| 02 Amplitude/spectral | `stage02_csc_spectral_raw200.png` | Summarizes the four gray-fruit spectral profiles across CSC electrodes. | Descriptive group-level profile; it does not establish memory-color coding. |
| 03 Decoding | `stage03_task3_red_green_raw200.png` and `stage03_task2_memory_color_raw200.png` | Preserves the key physical-color versus memory-color contrast. | Task 3 has a stimulus-identity repetition caveat; Task 2 is not a positive group-level memory-color result. |
| 04 Luminance | no image | The stage is represented by its audit report and tables. | This is a control/audit stage, not a primary neural result. |
| 05 Hypotheses | no image | The stage is represented by H1–H4 result text and tables. | H2/H3/H4 must retain their exploratory and negative-result caveats. |
| 06 Exploration | `stage06_s1s2_timeresolved_raw200.png` | Directly shows S1/S2 physical-color and memory-color time-resolved curves. | Single-electrode curves and group clusters are exploratory; no global across-electrode correction. |
| 07 CSC decoding | `stage07_csc_time_curves.png` | Shows within-task and cross-task CSC time-resolved decoding together. | The figure is explicitly without permutation shading; cross-task near-chance behavior is not proof of distinct mechanisms. |
| 08 S1/S2 single-electrode | `stage08_s1_memory_latency.png` and `stage08_s2_memory_latency.png` | Shows single-electrode memory-color curves, coordinates, and earliest-cluster latency. | Nominal single-electrode clusters are exploratory and lack global across-electrode correction. |

## Current method contract displayed on Canvas

- Main time window: `100–400 ms` for functional selection.
- Decoding input: `raw200` from the HDF5 epochs.
- Functional selection: S1 = Task 1 ANOVA color main effect; S2 = at least one category Welch test; CSC = N2 ∩ (S1 ∪ S2).
- Existing Canvas figures are frozen evidence. A changed parameter will create a new candidate run and new result nodes; it will not overwrite these figures.

## Next parameter experiment

The Canvas now contains a completed candidate branch for continuous 200 ms electrode selection. Future parameter experiments must specify the parameter and will remain separate from both the historical display and this candidate branch.

## New candidate: continuous 200 ms electrode selection

The new candidate analysis is stored under:

```text
color_analyse_0727/result/continuous_selection_200ms_100-400_20260807/
```

It does not use the mean of the full 100–400 ms interval. Instead, within that interval it tests the four category-wise color-minus-gray contrasts at each time point and applies a category-balanced one-sample test across the four contrasts. An electrode is a candidate when `p<0.05` remains true for at least 200 ms continuously. The pooled color-versus-gray time-point test is retained as a sensitivity column.

The primary candidate counts for `test001` through `test007` are `33, 9, 27, 11, 12, 5, 20`; the historical 100–400 ms S1 counts are `7, 0, 2, 6, 1, 0, 3`. These are exploratory counts and have not replaced the current electrode set. A separate `0–800 ms` sensitivity run is stored under `continuous_selection_200ms_20260807/`.

## New candidate: continuous 100 ms electrode selection

The 100 ms candidate is stored under:

```text
color_analyse_0727/result/continuous_selection_100ms_100-400_20260809/
```

It uses the same 100–400 ms scan and category-balanced pointwise criterion as the 200 ms candidate, but changes the required continuous duration to 100 ms. The primary candidate counts remain `33, 9, 27, 11, 12, 5, 20`; in this dataset, lowering the duration threshold did not add primary category-balanced candidates. The run is retained as a separate exploratory parameter branch.
