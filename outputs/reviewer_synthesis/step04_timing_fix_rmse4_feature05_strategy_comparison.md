# Step 04 timing-fix strategy comparison

Acceptance contract: `mean_trace_rmse_mV <= 4.0` and `mean_weighted_pass_fraction >= 0.5`.

This run uses detected ATF command-step timing for simulation, feature extraction, scoring, and overlays.

## Summary
| strategy                        | n_candidate_rows | n_accepted_rows | n_accepted_cells | mean_best_trace_rmse_mV | mean_best_weighted_pass_fraction |
| ------------------------------- | ---------------- | --------------- | ---------------- | ----------------------- | -------------------------------- |
| nsga2_trace_feature_adaptive500 | 97               | 45              | 5                | 3.097                   | 0.802                            |
| least_squares_12starts          | 84               | 41              | 4                | 3.283                   | 0.796                            |

## By Cell
| strategy                        | file_id        | region | condition | stim_onset_s | n_candidates | n_accepted_candidates | best_trace_rmse_mV | best_weighted_pass_fraction | best_meets_contract |
| ------------------------------- | -------------- | ------ | --------- | ------------ | ------------ | --------------------- | ------------------ | --------------------------- | ------------------- |
| least_squares_12starts          | 1_DH_1_CONTROL | DH     | CONTROL   | 11.166       | 12           | 12                    | 2.658              | 0.661                       | 1                   |
| nsga2_trace_feature_adaptive500 | 1_DH_1_CONTROL | DH     | CONTROL   | 11.166       | 8            | 8                     | 2.650              | 0.661                       | 1                   |
| least_squares_12starts          | 3_DH_1_CONTROL | DH     | CONTROL   | 21.144       | 12           | 8                     | 3.587              | 0.677                       | 1                   |
| nsga2_trace_feature_adaptive500 | 3_DH_1_CONTROL | DH     | CONTROL   | 21.144       | 8            | 6                     | 3.280              | 0.676                       | 1                   |
| least_squares_12starts          | 2_VH_1_CONTROL | VH     | CONTROL   | 21.117       | 12           | 0                     | 2.783              | 0.434                       | 0                   |
| nsga2_trace_feature_adaptive500 | 2_VH_1_CONTROL | VH     | CONTROL   | 21.117       | 12           | 0                     | 2.783              | 0.433                       | 0                   |
| least_squares_12starts          | DH_1_MFA       | DH     | MFA       | 21.096       | 12           | 0                     | 5.524              | 1.000                       | 0                   |
| nsga2_trace_feature_adaptive500 | DH_1_MFA       | DH     | MFA       | 21.096       | 26           | 0                     | 5.206              | 1.000                       | 0                   |
| least_squares_12starts          | VH_1_MFA       | VH     | MFA       | 21.105       | 12           | 12                    | 2.387              | 0.839                       | 1                   |
| nsga2_trace_feature_adaptive500 | VH_1_MFA       | VH     | MFA       | 21.105       | 19           | 19                    | 2.301              | 0.858                       | 1                   |
| least_squares_12starts          | DH_1_MFA_BA    | DH     | MFA_BA    | 21.106       | 12           | 0                     | 4.216              | 0.988                       | 0                   |
| nsga2_trace_feature_adaptive500 | DH_1_MFA_BA    | DH     | MFA_BA    | 21.106       | 13           | 1                     | 3.981              | 1.000                       | 1                   |
| least_squares_12starts          | VH_4_MFA_BA    | VH     | MFA_BA    | 21.075       | 12           | 9                     | 1.825              | 0.973                       | 1                   |
| nsga2_trace_feature_adaptive500 | VH_4_MFA_BA    | VH     | MFA_BA    | 21.075       | 11           | 11                    | 1.477              | 0.986                       | 1                   |

## Artifacts
- Summary CSV: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_strategy_summary.csv`
- By-cell CSV: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_by_cell.csv`
- Selected candidates CSV: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_selected_candidates.csv`
- Overlay PDF: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_trace_overlays.pdf`
