# analyse_0617 select_channel batch: test004/005/006

This batch reproduces the legacy `step1_1_select_channel_extended.py` statistical logic:

- Wilcoxon rank-sum (`scipy.stats.ranksums`), uncorrected `p < 0.05`;
- merged color (11/21/31/41) vs gray (12/22/32/42), 100–400 ms mean;
- merged point-wise significance from 50–400 ms with a continuous 50 ms requirement;
- the same two tests repeated separately for Face/Object/Body/Place;
- a channel passes the functional screen when any of four strategies passes.

`select_channel_summary.csv` contains the functional result for every selected channel. `legacy_select_channel_summary.csv` applies the old anatomical ROI gate as well (calcarine/occipital/lingual, fusiform/temporal-inf, or temporal-mid/pole). Because test004 has no localization, its legacy ROI table is empty; its functional candidates are retained and explicitly marked `has_localization=False`.

The multi-subject tables concatenate subject-level rows; they do not treat channels with different names as the same electrode.
