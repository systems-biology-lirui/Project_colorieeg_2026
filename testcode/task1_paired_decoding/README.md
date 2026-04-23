# Task1 Paired Color-Gray Decoding

This folder contains binary decoding schemes that explicitly use the one-to-one mapping between each color image and its gray version.

Required groupedData contract:

- Input file is a `.mat` file with a variable named `groupedData`.
- `groupedData` must be a `4 x 2` cell array.
- Rows are categories, in the same order as `--category-names`.
- Column 1 stores the sample ids for the color condition of that category.
- Column 2 stores the sample ids for the gray condition of that category.
- The position inside each cell matches the repeat index inside the ROI tensor.
- The same numeric id across the color and gray cells of the same row means the same image.
- The paired decoding scripts now first read the bad-trial/repeat-selection metadata saved in `processed_data/{subject}/task1_ERP_epoched.mat` or `task1_TFA_epoched.mat`, then reorder/subselect each groupedData cell onto the saved repeat axis used by the ROI feature tensor.
- If your groupedData still contains the original 70 ids, rerun `newanalyse/Sec1_preanalyse.m` first so those metadata fields exist. If your groupedData has already been manually trimmed to the saved repeat axis, the scripts will accept it as-is.

Default Task1 condition mapping:

- Color conditions: `0,2,4,6`
- Gray conditions: `1,3,5,7`
- Categories: `face,body,object,scene`

Scripts:

- `paired_group_cv_decoding.py`
  Use the original trials and run grouped cross-validation so the two samples from one matched image pair are always kept in the same fold.

- `within_pair_centered_decoding.py`
  Subtract the pair mean from color and gray trials before decoding. This removes image-shared content and asks whether the residual inside each pair still separates color from gray.

- `pair_difference_decoding.py`
  Build signed difference vectors from each pair using `+(color-gray)` and `-(color-gray)`. This isolates the color effect most directly, but it is also the most sensitive to noisy pair alignment.

- `plot_pairwise_pca.py`
  Plot PCA projections for raw paired samples, within-pair centered samples, and pair-difference vectors. Supports selecting a subset of categories, optional 3D plots using the first three PCs, and optional bar+scatter summaries over the leading PCs. The bar mode saves one figure split by condition and another merged color-vs-gray figure with per-PC paired significance.

- `plot_condition_pca_trajectories.py`
  Plot the time-evolving PCA trajectories of condition-mean signals in one 2D figure and one 3D figure. Each figure includes raw and within-pair centered versions, supports temporal smoothing, start/end markers, optional intermediate markers every fixed number of time points, an optional plotting/projection time window, and an optional interactive 3D HTML export that can be rotated in a browser.

- `plot_condition_cosine_similarity.py`
  Plot the time-varying cosine similarity between color and gray condition-mean signals for each category. Saves raw and centered versions, and supports temporal smoothing.

Example usage:

```bash
python testcode/task1_paired_decoding/paired_group_cv_decoding.py \
  --subject test001 \
  --feature-kind erp \
  --roi-name Color_with_sti \
  --grouped-data-mat /path/to/groupedData.mat \
  --n-perms 200 \
  --n-repeats-perm 1
```

```bash
python testcode/task1_paired_decoding/plot_pairwise_pca.py \
  --subject test001 \
  --feature-kind erp \
  --roi-name Color_with_sti \
  --grouped-data-mat /path/to/groupedData.mat \
  --selected-categories face,object \
  --plot-3d \
  --plot-bars \
  --bar-n-components 4 \
  --window-start-ms 0 \
  --window-end-ms 100
```

```bash
python testcode/task1_paired_decoding/plot_condition_pca_trajectories.py \
  --subject test001 \
  --feature-kind erp \
  --roi-name Color_with_sti \
  --selected-categories 1,2,3,4 \
  --smooth-win 20 \
  --plot-start-ms 0 \
  --plot-end-ms 300 \
  --interactive-3d \
  --point-step 20
```

```bash
python testcode/task1_paired_decoding/plot_condition_cosine_similarity.py \
  --subject test001 \
  --feature-kind erp \
  --roi-name Color_with_sti \
  --selected-categories all \
  --smooth-win 7
```

Outputs are written under `result/reports/task1_paired_decoding/`.

Permutation / cluster significance:

- `--n-perms` controls the number of null permutations. `0` disables permutation significance and only saves the real decoding curve.
- `--n-repeats-perm` controls the grouped CV repeats used inside each permutation.
- Permutation shuffling is pair-aware: labels are swapped within each color-gray pair instead of being fully randomized across samples.
- The saved `.npz` now includes `perm_dist`, `threshold_95`, `sig_mask`, and latency summaries from the cluster test.