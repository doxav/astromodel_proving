# Step 04 Full Adaptive500 Rerun Analysis

## Configuration

- Notebook: `analysis/04_cell_specific_six_sweep_fitting.ipynb`
- Preset: `nsga2_trace_feature_adaptive500`
- Optimizer: Optuna multi-objective NSGA2, base 160 trials, adaptive extension to 500 trials when fewer than 3 accepted candidates are found.
- Extended fitted parameters enabled: `K_bath_value_middle` and `eps_middle` search-space extension.
- Acceptance: all-six `mean_trace_rmse_mV <= 4.0` and `mean_weighted_pass_fraction >= 0.5`.
- Heldout screen: `heldout_trace_rmse_mV <= 4.0`, `heldout_weighted_pass_fraction >= 0.5`, reviewer-facing cell requires at least 3 heldout passes.
- Workers: 20 cell-level workers.
- Runtime: 46.3 min.
- Output reset before run: `['/home/xav/code/astromodel_proving/outputs/cell_fits', '/home/xav/code/astromodel_proving/outputs/cell_fits_step04_model_aligned_demo']`.

## Result

The rerun produced 541 retained candidates, 353 accepted all-six candidates, 123 effective-diverse downstream candidates, and 29/37 reviewer-facing cells.

| region | condition | cells | accepted_cells | reviewer_cells | accepted_candidates | median_best_rmse | median_best_pass | median_holdout_pass_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DH | CONTROL | 7 | 7 | 7 | 83 | 3.014 | 0.675 | 4.000 |
| DH | MFA | 6 | 3 | 3 | 23 | 4.424 | 1.000 | 3.500 |
| DH | MFA_BA | 6 | 6 | 6 | 46 | 3.046 | 1.000 | 4.000 |
| VH | CONTROL | 4 | 0 | 0 | 0 | 2.724 | 0.417 | 3.500 |
| VH | MFA | 7 | 6 | 6 | 88 | 2.564 | 0.858 | 5.000 |
| VH | MFA_BA | 7 | 7 | 7 | 113 | 2.127 | 0.985 | 5.000 |

## Remaining Nonaccepted Cells

These cells have zero accepted all-six candidates under the strict 4.0/0.5 contract. There are no cells with accepted all-six candidates that then fail only because of heldout pass count.

| file_id | region | condition | best_trace_rmse_mV | best_weighted_pass_fraction | holdout_pass_count | holdout_mean_rmse_mV | holdout_mean_pass_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1_VH_1_CONTROL | VH | CONTROL | 2.460 | 0.379 | 4 | 2.501 | 0.431 |
| 1_VH_2_CONTROL | VH | CONTROL | 2.653 | 0.322 | 3 | 2.670 | 0.472 |
| 2_VH_1_CONTROL | VH | CONTROL | 2.794 | 0.454 | 2 | 2.796 | 0.428 |
| 3_VH_1_CONTROL | VH | CONTROL | 2.936 | 0.495 | 4 | 2.899 | 0.505 |
| DH_1_MFA | DH | MFA | 5.198 | 1.000 | 3 | 5.173 | 0.981 |
| DH_4_MFA | DH | MFA | 5.473 | 1.000 | 3 | 4.830 | 0.960 |
| DH_5_MFA | DH | MFA | 5.526 | 1.000 | 3 | 5.603 | 1.000 |
| VH_3_MFA | VH | MFA | 5.033 | 0.858 | 3 | 5.024 | 0.857 |

Interpretation: VH CONTROL cells are trace-aligned but fail the feature-pass threshold; DH MFA and `VH_3_MFA` are centered trace/amplitude failures despite good feature pass. These are scientific/model-fit limitations, not stale-output or raw-baseline plotting artifacts.

## Vm Level Root Cause

The raw Vm level mismatch is caused by comparing raw experimental Vm with raw model Vm while Step 04 optimizes baseline-centered responses. The code now writes overlay columns for raw, baseline-centered, and baseline-aligned predicted Vm. The full-run overlay summary shows that raw RMSE is dominated by baseline offsets, while centered RMSE is the relevant fitted metric.

| region | condition | selection_note | cells | sweeps | median_abs_baseline_delta_mV | median_raw_rmse_mV | median_centered_rmse_mV | median_abs_stim_end_delta_mV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DH | CONTROL | accepted_best | 7 | 42 | 21.890 | 23.110 | 2.920 | 5.205 |
| DH | MFA | accepted_best | 3 | 18 | 32.886 | 33.792 | 3.153 | 5.724 |
| DH | MFA | best_nonaccepted | 3 | 18 | 20.483 | 24.260 | 4.890 | 9.873 |
| DH | MFA_BA | accepted_best | 6 | 36 | 33.532 | 34.440 | 2.426 | 4.734 |
| VH | CONTROL | best_nonaccepted | 4 | 24 | 30.666 | 31.472 | 2.571 | 4.510 |
| VH | MFA | accepted_best | 6 | 36 | 27.892 | 28.214 | 2.254 | 4.113 |
| VH | MFA | best_nonaccepted | 1 | 6 | 17.268 | 20.629 | 4.640 | 9.686 |
| VH | MFA_BA | accepted_best | 7 | 42 | 32.892 | 35.025 | 1.557 | 3.350 |

## Final Figure/PDF Selection

Use `outputs/reviewer_synthesis/step04_full_adaptive500_baseline_aligned_trace_overlays.pdf` for the final reviewer-facing Step 04 trace figures/PDF. This is the `nsga2_trace_feature_adaptive500` aligned version: black raw experimental Vm plus solid `vm_predicted_baseline_aligned_mV` model traces. Older raw model Vm overlay PDFs are diagnostic only. The machine-readable selection is in `outputs/reviewer_synthesis/step04_final_figure_manifest.csv`.

## Artifacts

- Canonical Step 04 outputs: `outputs/cell_fits/`
- SQLite audit DB: `outputs/cell_fits/step04_cell_fits.sqlite`
- Full-run baseline-aligned PDF: `outputs/reviewer_synthesis/step04_full_adaptive500_baseline_aligned_trace_overlays.pdf`
- Full-run best-candidate overlay points: `outputs/reviewer_synthesis/step04_full_adaptive500_best_overlay_points_baseline_aligned.csv`
- Full-run overlay diagnostics: `outputs/reviewer_synthesis/step04_full_adaptive500_best_overlay_diagnostics.csv`
- Root-cause analysis: `outputs/reviewer_synthesis/step04_vm_level_amplitude_root_cause_analysis.md`
