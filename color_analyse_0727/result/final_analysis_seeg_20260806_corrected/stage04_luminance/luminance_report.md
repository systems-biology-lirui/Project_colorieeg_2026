# Stimulus luminance audit

## Task 1: color vs gray (280 pairs)

| category | n_pairs | mean_abs_dLum | max_abs_dLum | mean_abs_dContrast | wilcoxon_p_lum_color_vs_gray | flag_luminance_mismatch | mean_colorfulness_color | category_x_condition_interaction_p | flag_category_interaction | max_category_mean_abs_dLum |
|---|---|---|---|---|---|---|---|---|---|---|
| face | 70 | 1.999 | 3.959 | 1.136 | 3.558e-13 | False | 35.88 | 1.159e-09 | False | 1.999 |
| object | 70 | 0.6928 | 3.088 | 0.8356 | 0.003337 | False | 41.15 | 1.159e-09 | False | 1.999 |
| body | 70 | 1.353 | 8.627 | 0.7075 | 3.345e-09 | False | 30.04 | 1.159e-09 | False | 1.999 |
| place | 70 | 0.6596 | 2.758 | 0.2905 | 0.07865 | False | 37.98 | 1.159e-09 | False | 1.999 |

## Task 2: gray fruits (luminance balance)

One-way ANOVA across four gray fruits: p = 0.0000

| mean | std | min | max |
|---|---|---|---|
| 99.66 | 0.01 | 99.64 | 99.68 |
| 99.47 | 0.023 | 99.41 | 99.49 |
| 99.22 | 0.073 | 99.1 | 99.36 |
| 99.28 | 0.028 | 99.22 | 99.34 |

## Task 3: pure color patches

| mean | std |
|---|---|
| 68.89 | 0.038 |
| 77.08 | 0.028 |
| 87.7 | 0.015 |
| 85.07 | 0.018 |
| 148.2 | 0.058 |
| 93.7 | 0.008 |

Red minus Green luminance: -2.62

Flag threshold for mean |dL|: 3.0 (0-255 Y scale)
