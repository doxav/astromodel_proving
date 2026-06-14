# Step 04 Vm Level And Amplitude Root-Cause Analysis

## Conclusion

The large raw Vm level mismatch in the prior trace PDF is primarily a display/contract mismatch, not an optimizer failure. Step 04 scores traces after subtracting each sweep's pre-stimulus baseline, while the old overlay plotted raw experimental Vm against raw model Vm. The model raw baseline remains near its internal resting convention, whereas experimental baselines vary widely by cell/sweep.

The overlay-generation code now keeps the raw columns and adds baseline-centered and baseline-aligned prediction columns. The updated PDF plots the baseline-aligned predictions so the visual comparison matches the Step 04 objective convention.

Amplitude differences remain real response-shape differences after baseline alignment. They should not be hidden by post-hoc scaling. The diagnostic table reports peak/end depolarization deltas so these cases can be targeted by the next fitting-objective change if needed.

## Summary By Strategy

| strategy | n_sweeps | median_abs_baseline_delta_mV | median_raw_rmse_mV | median_centered_rmse_mV | median_baseline_aligned_rmse_mV | median_abs_stim_end_delta_mV | median_abs_peak_delta_mV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| least_squares_12starts | 42 | 24.711 | 26.743 | 2.968 | 2.968 | 5.018 | 5.119 |
| nsga2_trace_feature_adaptive500 | 42 | 24.704 | 26.828 | 2.675 | 2.675 | 4.878 | 4.991 |

## Hypotheses Tested

1. **Raw overlay uses the wrong visual convention**: supported. Median raw RMSE is about 26-27 mV, but median baseline-centered/aligned RMSE is about 2.7-3.0 mV on the same saved candidates.
2. **The simulator should start at the experimental baseline Vm**: not supported for the current model. The dedicated z0 diagnostic (`step04_z0_baseline_hypothesis_diagnostic.csv`) showed that using observed `z0[0]` did not remove the baseline offset by the baseline window and worsened centered RMSE.
3. **The remaining amplitude mismatch is purely a plotting offset**: not supported. Some cells still have large centered end-depolarization deltas after baseline alignment, especially DH MFA/MFA_BA high-current sweeps.
4. **Timing misalignment is the main cause of this specific level mismatch**: not supported after the previous timing fix. The per-file onset/offset is now carried into simulation and feature extraction; the remaining raw-level mismatch follows baseline values, not onset values.

## Worst Raw Baseline Offsets

| strategy | file_id | candidate_id | sweep | current_na | obs_baseline_mV | pred_baseline_mV | baseline_delta_pred_minus_obs_mV | raw_rmse_mV | centered_rmse_mV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| least_squares_12starts | VH_4_MFA_BA | VH_4_MFA_BA__cand_07 | 6 | 175 | -38.219 | -89.346 | -51.127 | 53.140 | 3.705 |
| nsga2_trace_feature_adaptive500 | VH_4_MFA_BA | VH_4_MFA_BA__cand_11 | 6 | 175 | -38.219 | -88.809 | -50.590 | 51.751 | 3.148 |
| least_squares_12starts | VH_4_MFA_BA | VH_4_MFA_BA__cand_07 | 5 | 150 | -39.001 | -89.346 | -50.345 | 51.842 | 2.725 |
| least_squares_12starts | VH_4_MFA_BA | VH_4_MFA_BA__cand_07 | 4 | 125 | -39.158 | -89.346 | -50.188 | 51.086 | 1.811 |
| nsga2_trace_feature_adaptive500 | VH_4_MFA_BA | VH_4_MFA_BA__cand_11 | 5 | 150 | -39.001 | -88.809 | -49.808 | 50.457 | 2.227 |
| nsga2_trace_feature_adaptive500 | VH_4_MFA_BA | VH_4_MFA_BA__cand_11 | 4 | 125 | -39.158 | -88.809 | -49.651 | 49.703 | 1.440 |
| least_squares_12starts | VH_4_MFA_BA | VH_4_MFA_BA__cand_07 | 3 | 100 | -40.458 | -89.346 | -48.888 | 49.699 | 1.244 |
| nsga2_trace_feature_adaptive500 | VH_4_MFA_BA | VH_4_MFA_BA__cand_11 | 3 | 100 | -40.458 | -88.809 | -48.351 | 48.316 | 0.691 |

## Worst Centered Amplitude Differences

| strategy | file_id | candidate_id | sweep | current_na | obs_stim_end_centered_mV | pred_stim_end_centered_mV | stim_end_delta_pred_minus_obs_mV | centered_rmse_mV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nsga2_trace_feature_adaptive500 | DH_1_MFA | DH_1_MFA__cand_20 | 6 | 175 | 33.563 | 13.725 | -19.837 | 9.671 |
| least_squares_12starts | DH_1_MFA | DH_1_MFA__cand_01 | 6 | 175 | 33.563 | 13.743 | -19.820 | 9.687 |
| nsga2_trace_feature_adaptive500 | DH_1_MFA_BA | DH_1_MFA_BA__cand_13 | 6 | 175 | 30.232 | 13.599 | -16.633 | 7.650 |
| least_squares_12starts | DH_1_MFA_BA | DH_1_MFA_BA__cand_07 | 6 | 175 | 30.232 | 13.695 | -16.537 | 7.726 |
| nsga2_trace_feature_adaptive500 | DH_1_MFA | DH_1_MFA__cand_20 | 5 | 150 | 29.270 | 13.725 | -15.544 | 7.551 |
| least_squares_12starts | DH_1_MFA | DH_1_MFA__cand_01 | 5 | 150 | 29.270 | 13.743 | -15.527 | 7.625 |
| nsga2_trace_feature_adaptive500 | DH_1_MFA_BA | DH_1_MFA_BA__cand_13 | 5 | 150 | 26.723 | 13.599 | -13.124 | 6.009 |
| least_squares_12starts | DH_1_MFA_BA | DH_1_MFA_BA__cand_07 | 5 | 150 | 26.723 | 13.695 | -13.028 | 6.114 |

## Artifacts

- Updated overlay PDF: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_trace_overlays_baseline_aligned.pdf`
- Full-run baseline-aligned PDF after notebook rerun: `outputs/reviewer_synthesis/step04_full_adaptive500_baseline_aligned_trace_overlays.pdf`
- Overlay points: `outputs/reviewer_synthesis/step04_timing_fix_rmse4_feature05_overlay_points_baseline_aligned.csv`
- Sweep diagnostics: `outputs/reviewer_synthesis/step04_vm_level_amplitude_diagnostics.csv`
- Summary table: `outputs/reviewer_synthesis/step04_vm_level_amplitude_summary.csv`
- z0 hypothesis diagnostic: `outputs/reviewer_synthesis/step04_z0_baseline_hypothesis_diagnostic.csv`
