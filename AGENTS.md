# Research Canvas execution rules

## Scope

- The active scientific analysis workspace is `color_analyse_0727/`.
- Existing results under `color_analyse_0727/result/` are immutable historical evidence unless the user explicitly requests a new analysis.
- The Obsidian Canvas is `workspace/SEEG_analysis.canvas`.
- Representative copies of existing figures are stored under `workspace/figures/existing_results/` only for Canvas display. They must retain their original source path in the result index.

## Existing results

- Do not rerun completed historical stages merely to populate the Canvas.
- Select a small number of representative figures per stage; do not place every figure on the main Canvas.
- Stages without figures should be represented by concise result/summary text nodes linked to their tables or reports.
- Historical results must be labeled as existing, exploratory, or confirmatory according to their source documentation.

## New parameter experiments

- A parameter experiment is a new run and must not overwrite existing results.
- The user must specify the scientific parameter to change, or Codex must clearly state the assumed parameter before execution.
- Never silently change frequency definition, time window, baseline, trial inclusion, classifier, cross-validation strategy, statistical test, or signal variant.
- Save the full parameter configuration, command, input identity, Git commit, environment, timestamps, warnings, and output paths under `runs/`.
- After a successful run, write only a concise scientific summary and useful provenance back to the Canvas. Do not write hidden chain-of-thought or routine execution chatter.
- Continuous electrode-selection candidates are stored under their dated result directories. The 200 ms candidate is under `color_analyse_0727/result/continuous_selection_200ms_100-400_20260807/`; the new 100 ms candidate is under `color_analyse_0727/result/continuous_selection_100ms_100-400_20260809/`. Their primary rule is a category-balanced color-minus-gray effect that remains pointwise significant for the configured minimum duration inside 100–400 ms. The 0–800 ms version is sensitivity-only.

## Canvas safety

- Preserve node IDs, positions, sizes, and unrelated user edits.
- Use targeted, atomic Canvas updates and fail if the Canvas changed after it was read.
- A pending figure node may become a file node only after the new figure exists and the run completed successfully.
