# Refined placement instructions for the "TO BE GENERATED" layer

Status: implementation-aligned routing and notebook-update plan for the first
pass. The first pass uses top legacy Optuna trials, the existing Filtered
baseline perturbation protocol, and the updated ATF regional/condition analysis
notebook supplied at `/home/xav/Downloads/astro_atf_analysis_improved_sectioned.ipynb`.

This document tells the next AI assistant where each refined element belongs in
the Step 00-09 pipeline and which source modules/notebooks to modify. Keep the
affected step specs and `specs_roadmap/reviewer_response_implementation_spec.md`
aligned with this contract before treating notebook outputs as final.

## Core Source Boundary

- The baseline configurations for this layer are legacy Optuna configurations,
  not Step 04 ATF cell-specific accepted ensembles.
- Legacy configuration data come from `data/1_Initial_xp_fit/*.db` and
  `data/1_Initial_xp_fit/*_BEST_FIT_PARAM.csv`. The first pass uses the top
  legacy Optuna trials, not thresholded accepted trials.
- Step 04 accepted ensembles may be used only as a separate comparison branch
  with `source_scope = "step04_atf_cell_specific"`. They must not be mixed with
  the legacy baseline set.
- ATF data in `data/2_K+ Pumps Data/*.atf` provide experimental direction-of-
  change targets under `CONTROL -> MFA`, `MFA -> MFA_BA`, and
  `CONTROL -> MFA_BA`, including DH/VH regional differences.
- Naris perturbation magnitudes, I-V protocols, biocytin/membrane-conductance
  data, and movie generation are deliberately deferred. Do not block the first
  pass on these items and do not invent external values for them.
- `gamma_s_eff * Chi(t)` or `alpha2 * Chi(t)` must be written as a reduced-model
  recruited-surface / functional-syncytium proxy, not as a direct anatomical
  astrocyte count.

## Existing Notebook Logic To Reuse

- Unified mechanism/category source:
  `analysis/unified_astrocyte_K_buffering_characterization_EXECUTED_SMOKE.ipynb`
  contains the legacy-configuration phenotype vocabulary, `Chi(t)` /
  dKs-activation interpretation, K_o feature summaries, and sigmoid temporal
  recruitment tags.
- Legacy parameter-sweep protocol source:
  `analysis/Filtered_basline_sweep (1).ipynb` contains the one-at-a-time
  parameter sweep mechanics and fold grids for `gki`, `d`, `pk`, `zth`, `zs`,
  and `gs`. Reuse the sweep mechanics, but apply them to the first-pass top
  legacy Optuna configurations unless the later thresholded good-enough layer is
  explicitly enabled.
- ATF experimental contrast source:
  `/home/xav/Downloads/astro_atf_analysis_improved_sectioned.ipynb` contains
  the updated second-layer condition-pair, DH/VH delta-of-delta, numeric
  sweep-trend, region-blind condition-trend, delta-feature-profile, and selected
  region-perturbation-summary logic. Port that logic into Step 02 outputs rather
  than leaving it as notebook-only state. The repo copy
  `analysis/astro_atf_analysis_improved_sectioned.ipynb` can be used as the
  older reference, but the downloaded notebook is the current refined source.

## Notebook Placement Map

Use headings as the stable edit target. Cell indexes are only orientation aids.

| Step | Notebook | Additions |
|---|---|---|
| 00 | `analysis/00_data_provenance_audit.ipynb` | Keep legacy-vs-ATF source contract explicit. Add external-data provenance only if new Naris or biocytin files are supplied. |
| 01 | `analysis/01_postfit_sqlite_pipeline.ipynb` | Create the legacy configuration library and legacy perturbation-factor candidates. |
| 02 | `analysis/02_rebuild_atf_thresholds.ipynb` | Export ATF experimental membrane-kinetic direction targets and DH/VH perturbation-difference targets. |
| 03 | `analysis/03_combined_identifiability_profiles_fim.ipynb` | Keep effective-parameter definitions and structural confounding evidence. No perturbation simulation here. |
| 04 | `analysis/04_cell_specific_six_sweep_fitting.ipynb` | No primary changes for this refined layer. Do not source baseline configs from Step 04. |
| 05 | `analysis/05_mechanistic_decomposition.ipynb` | Add a legacy-source mechanism/category run or a separate legacy mechanism notebook section. Compute baseline K_o efficiency and sigmoid categories. |
| 06 | `analysis/06_predictive_validation_and_perturbation.ipynb` | Add the biologically meaningful MFA/MFA+BA parameter-variation layer, 1D/2D perturbation sweeps, direction matching, and visual summaries. |
| 07 | `analysis/07_assumption_sensitivity.ipynb` | Only extend if the refined implementation needs stability of the new EF/sigmoid labels across gating-family assumptions. |
| 08 | `analysis/08_parameter_plausibility_and_constrained_reruns.ipynb` | Audit interpretation of newly perturbed raw/effective parameters if claims use them biologically. |
| 09 | `analysis/09_reviewer_response_synthesis.ipynb` | Register new artifacts and gate the final pathway/perturbation claims. Do not run simulations here. |

## Source Modules To Reuse Or Extend

- `src/provenance.py`: source inventory and source-scope labels.
- `src/postfit_sqlite.py`: legacy SQLite top/best-trial loading and hidden
  simulation entrypoint.
- `src/parameter_space.py`: effective coordinate transforms and
  `set_coordinate`.
- `src/atf_features.py::extract_features_from_trace`: membrane/Vm feature
  extraction for simulated traces.
- `src/astro_model.py`: hidden-output simulation, current reconstruction, and
  switching-gate calculation.
- `src/mechanisms.py`: flux summaries.
- `src/phenotype_classifier.py`: windowed mechanism scores, `state_10_90`, and
  sigmoid/recruitment categories.
- `src/step05_mechanistic_decomposition.py`: mechanism/category assignment.
- `src/step06_predictive_validation.py`: perturbation execution and
  direction-of-change summaries.
- `src/step08_parameter_plausibility.py` and `src/reviewer_gate_audits.py`:
  semantic/interpretability gates.
- `src/step09_reviewer_synthesis.py`: final artifact ledger and claim status.

Create a small shared helper module, preferably `src/functional_mapping.py`, if
the K_o efficiency, sigmoid-transition, and direction-of-change helpers are used
by more than one step. Otherwise keep local helpers inside the owning step
module. Full type annotations and useful docstrings are required for every new
function.

## Step 01: Legacy Configuration Library

Chosen location:

- Module: extend `src/postfit_sqlite.py`.
- Notebook: insert after "Effective parameter summary (all 18 DBs)" and before
  "Representative mechanism summary".
- Primary output directory: `outputs/postfit_sqlite/`.

Required outputs:

- `outputs/postfit_sqlite/legacy_configuration_library.csv`
- `outputs/postfit_sqlite/legacy_configuration_status_by_db.csv`
- `outputs/postfit_sqlite/legacy_condition_parameter_ratios.csv`

Implementation requirements:

- Build the first-pass library from the top legacy Optuna DB trials.
- Default selection rule:
  - `legacy_selection_rule = "top_n_by_objective"`
  - `legacy_top_n_requested = 300`
  - top means completed, non-penalty trials sorted by ascending objective and
    then `trial_number`.
  - if a DB contains fewer than 300 valid trials, include all valid trials and
    record `legacy_top_n_available`.
- Legacy best-fit CSVs may be used for metadata reconciliation and sanity
  checks, but they are not the primary source of the first-pass ensemble.
- Reuse `top_trials_with_effective_parameters`,
  `effective_parameters_from_flat`, and `representative_mechanism_summary`
  where possible.
- Merge provenance status from
  `outputs/provenance/control_trace_verification.csv` by `db_name`.
- Use explicit columns:
  - `source_scope = "legacy_single_current_optuna"`
  - `legacy_configuration_status`
  - `legacy_acceptance_rule`
  - `legacy_selection_rule`
  - `legacy_top_n_requested`
  - `legacy_top_n_available`
  - `rank_in_db`
  - `provenance_status`
  - `db_name`, `condition`, `current_na`, `trial_number`, `objective`
  - raw parameters: `gki`, `pk`, `d`, `gs`, `gt`, `zth`, `zs`, `eps`, `gl_a`
  - effective parameters: `P_gap_eff`, `gamma_s_eff`, `gamma_t_eff`,
    `volume_ratio_wa_wo`
- Set `legacy_configuration_status = "legacy_top300_optuna_trial"` for this
  first pass.
- Set `legacy_acceptance_rule = "not_thresholded_top_n_first_pass"` for this
  first pass. Do not use a plain `accepted` column for these rows.
- A later thresholded good-enough layer may be added using the ATF variability
  thresholds and the filtering logic from `analysis/Filtered_basline_sweep
  (1).ipynb`. If added, write it as a separate status such as
  `legacy_threshold_good_enough` and preserve the threshold mode, feature list,
  pass fraction, and soft-threshold settings as columns. Do not overwrite the
  top-300 first-pass status.

Legacy perturbation-factor candidates:

- Derive model-parameter ratios from legacy conditions only as candidate factors,
  not as direct biological truth.
- Write one row per `condition_pair x parameter x current_na` where possible.
- Required factor columns:
  - `perturbation_context`: one of `MFA_like_from_control_legacy`,
    `MFA_like_from_mfa_legacy`, `MFA_BA_from_MFA_legacy`, or
    `MFA_BA_stacked_on_control_legacy`
  - `parameter`: `P_gap_eff`, `gamma_s_eff`, `zth`, `zs`, `gki`
  - `factor`
  - `factor_source`: `filter_baseline_fold_grid`,
    `legacy_best_fit_ratio`, or `legacy_top_trial_ratio`
  - `factor_status`
- For the first pass, use the fold grid from `analysis/Filtered_basline_sweep
  (1).ipynb` rather than Naris-derived magnitudes. For `gki`, the grid is
  `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`; summarize BA-like Kir reduction primarily
  from folds below 1.0, with `0.75` as the closest grid point to an
  approximately 30% reduction.

## Step 02: Experimental Direction Targets From ATF Data

Chosen location:

- Module: add `src/experimental_perturbation_targets.py` or extend
  `src/atf_features.py` only if the helper remains small.
- Notebook: after "Region-effect summary" in
  `analysis/02_rebuild_atf_thresholds.ipynb`.
- Primary output directory: `outputs/features/`.

Required outputs:

- `outputs/features/experimental_kinetic_direction_targets.csv`
- `outputs/features/region_specific_perturbation_direction_targets.csv`
- `outputs/features/experimental_condition_contrast_summary.csv`
- `outputs/features/experimental_region_condition_profile_terms.csv`
- `outputs/features/experimental_second_layer/matched_sweep_delta_of_delta.csv`
- `outputs/features/experimental_second_layer/numeric_sweep_combo_slopes.csv`
- `outputs/features/experimental_second_layer/numeric_sweep_condition_slope_deltas_within_region.csv`
- `outputs/features/experimental_second_layer/numeric_sweep_delta_of_delta_between_regions.csv`
- `outputs/features/experimental_second_layer/region_blind_condition_slopes.csv`
- `outputs/features/experimental_second_layer/region_blind_condition_slope_deltas.csv`
- `outputs/features/experimental_second_layer/delta_feature_profiles_by_region.csv`
- `outputs/features/experimental_second_layer/delta_feature_profiles_region_blind.csv`
- `outputs/features/experimental_second_layer/region_perturbation_summary_selected_features.csv`

Implementation requirements:

- Port the updated ATF-analysis logic from
  `/home/xav/Downloads/astro_atf_analysis_improved_sectioned.ipynb` into a
  reusable source module, not notebook-only cells.
- Use the current Step 02 feature table as input:
  `outputs/features/feature_table_by_sweep.csv`.
- Preserve `region`, `condition`, `sweep`, `current_na`, and `feature`.
- Required condition contrasts:
  - `CONTROL_to_MFA`
  - `MFA_to_MFA_BA`
  - `CONTROL_to_MFA_BA`
- Required regional contrast:
  - `delta_of_delta_DH_minus_VH` for each condition contrast and feature.
- Required cell-profile region-condition outputs:
  - mixed-model term rows from `mixedlm_primary_continuous_term_tests.csv` and
    `mixedlm_conditional_continuous_term_tests.csv`;
  - binary feature rows from `gee_binary_term_tests.csv` if binary features are
    available;
  - retain terms involving `C(region):C(condition)` and, if enabled,
    `C(region):C(condition):C(sweep_cat)` as the experimental cell-profile
    region-by-condition differences.
- Required second-layer outputs to port from the updated notebook:
  - `matched_sweep_delta_of_delta.csv`, estimating
    `(DH_cond2 - DH_cond1) - (VH_cond2 - VH_cond1)` for each feature and sweep;
  - `numeric_sweep_combo_slopes.csv`, with sweep slopes by
    `region x condition`;
  - `numeric_sweep_condition_slope_deltas_within_region.csv`, with
    condition-induced sweep-slope changes separately in DH and VH;
  - `numeric_sweep_delta_of_delta_between_regions.csv`, with DH-minus-VH
    differences of condition-induced sweep-slope changes;
  - `region_blind_condition_slopes.csv` and
    `region_blind_condition_slope_deltas.csv`, for pooled DH/VH condition
    trends;
  - `delta_feature_profiles_by_region.csv` and
    `delta_feature_profiles_region_blind.csv`, for multifeature perturbation
    profiles;
  - `region_perturbation_summary_selected_features.csv`, summarizing how
    region gaps differ in control, MFA, and MFA+BA.
- Features must include at least:
  - `rise_slope_mV_per_s`
  - `rise_tau_s`
  - `decay_slope_mV_per_s`
  - `decay_tau_s`
  - `return_slope_mV_per_s`
  - `stim_end_depolarization_mV`
  - `peak_depolarization_mV`
  - `undershoot_magnitude_mV`
- Compute a signed target direction:
  - `increase`
  - `decrease`
  - `no_clear_change`
  - `undefined`
- Include statistical support columns if available from the updated ATF script:
  `estimate`, `ci_low`, `ci_high`, `pvalue`, `qvalue_bh`,
  `reject_bh_0p05`, `n_obs`, `n_files`.
- Use the updated notebook's FDR behavior for follow-up tests: adjust only
  non-NaN p-values and preserve NaN q-values.
- Keep optional Bayesian cells out of the required Step 02 implementation unless
  the expert explicitly requests Bayesian summaries.
- Step 06 must compare simulated perturbation directions against these targets.

## Step 03: Effective Parameter Definitions

Chosen location:

- Keep definitions in Step 03 and `src/parameter_space.py`.
- Do not move structural-confounding or effective-coordinate definitions into
  Step 06 perturbation code.

Required definitions:

- `P_gap_eff = d * pk`
- `gamma_t_eff = gt * Sig_a / (w_a * F)`
- `gamma_s_eff = gs * Sig_a / (w_a * F)`
- `volume_ratio_wa_wo = w_a / wo`

Implementation requirements:

- Use `src/parameter_space.py` as the single source of truth.
- Perturb `P_gap_eff` and `gamma_s_eff` with `set_coordinate` rather than
  manually changing raw `d`, `pk`, or `gs` unless a legacy raw-parameter
  diagnostic is explicitly requested.

## Step 05: Legacy Mechanism Categories And Baseline Function Mapping

Chosen location:

- Module: extend `src/step05_mechanistic_decomposition.py` and
  `src/phenotype_classifier.py`.
- Notebook: add a source-scoped legacy section to
  `analysis/05_mechanistic_decomposition.ipynb`.
- Primary legacy output directory: `outputs/legacy_mechanisms/`.

Required outputs:

- `outputs/legacy_mechanisms/legacy_fit_mechanisms.csv`
- `outputs/legacy_mechanisms/legacy_fit_mechanisms_windowed.csv`
- `outputs/legacy_mechanisms/legacy_mechanism_categories.csv`
- `outputs/legacy_mechanisms/legacy_mode_vector_by_configuration.csv`
- `outputs/legacy_mechanisms/legacy_function_efficiency_by_configuration.csv`
- `outputs/legacy_mechanisms/legacy_mechanistic_function_mapping.csv`
- `outputs/legacy_mechanisms/legacy_efficiency_thresholds.csv`
- `outputs/legacy_mechanisms/analysis_summary.json`

Unified notebook categories to port:

- State bins from `state_10_90`:
  - `closed_low`
  - `partial_mid`
  - `open_high`
  - `undefined`
- Windowed ionic-state labels:
  - `closed_low_redistribution`
  - `intermediate_recruitment`
  - `open_long_range_redistribution`
- Temporal recruitment classes:
  - `delayed_ionic_recruitment_after_load`
  - `recruited_during_load_then_closed_by_end`
  - `early_sustained_open_recruitment`
  - `persistently_low_range_closed`
  - `fully_open_at_sim_end`
  - `fully_closed_at_sim_end`
  - `intermediate_or_mixed_temporal_recruitment`

Baseline category columns to include:

- `sigmoid_state_at_stim_end_10_90`
- `sigmoid_state_at_sim_end_10_90`
- `temporal_recruitment_class`
- `gj_ionic_state_10_90`
- `functional_n_flux_proxy_dKs_activation`
- `functional_n_end_proxy_chi_end`
- `recruited_surface_alpha2_x_A_dKs`
- `alpha2_available_surface_proxy` or equivalent `gamma_s_eff`
- `source_scope`

K_o efficiency definition:

- Compute K_o kinetic features from hidden simulated `K_o`, not from Vm.
- Required columns:
  - `Ko_rise_rate_mM_per_s`
  - `Ko_decay_rate_abs_mM_per_s`
  - `Ko_rise_over_decay_rate`
  - `Ko_efficiency_score`
  - `Ko_rise_speed_class`
  - `Ko_decay_speed_class`
  - `Ko_efficiency_quadrant`
  - `Ko_efficiency_status`
- Continuous EF score:
  - `Ko_efficiency_score = Ko_rise_rate_mM_per_s / Ko_decay_rate_abs_mM_per_s`
  - validate finite positive denominator; otherwise set
    `Ko_efficiency_status = "undefined_flat_or_missing"`.
- The continuous EF score is the primary quantity for before/after perturbation
  comparison. Always save baseline EF, perturbed EF, signed EF delta, and EF
  direction.
- Four EF classes:
  - `fast_rise_fast_decay`
  - `slow_rise_fast_decay`
  - `slow_rise_slow_decay`
  - `fast_rise_slow_decay`
- Fast/slow thresholds:
  - There is no externally calibrated experimental threshold for these classes
    in the first pass.
  - Use a frozen descriptive split computed from the legacy baseline library:
    median K_o rise rate for rise fast/slow and median K_o decay rate for decay
    fast/slow.
  - Compute medians within `source_scope x current_na` when enough rows exist.
  - Fall back to global legacy medians if a stratum is underpowered.
  - Apply the same frozen baseline medians to perturbed rows so class transitions
    reflect movement relative to the original baseline distribution.
  - Write threshold columns in each output row:
    `Ko_rise_fast_slow_cutoff`, `Ko_decay_fast_slow_cutoff`,
    `Ko_efficiency_threshold_source`, and
    `Ko_efficiency_threshold_interpretation =
    "descriptive_legacy_baseline_median_no_experimental_observable"`.
  - Do not describe these labels as biological high/low-efficiency thresholds
    unless external validation is later added.

Function-vector to K-function mapping:

- FV columns: mechanism/parameter/state vector:
  `P_gap_eff`, `gamma_s_eff`, `gki`, `zth`, `zs`, sigmoid states,
  `temporal_recruitment_class`, flux fractions, and mechanism scores.
- FK columns: K-buffering function vector:
  `Ko_peak`, `Ko_final`, `Ko_recovery_error`, K_o rise/decay rates,
  `Ko_efficiency_score`, and `Ko_efficiency_quadrant`.
- Keep both continuous values and categorical labels. Do not rely on labels
  alone for downstream comparisons.

## Step 06: Biological Parameter-Variation Layer

Chosen location:

- Module: extend `src/step06_predictive_validation.py` or add a focused module
  imported by it, for example `src/biological_perturbations.py`.
- Notebook: add sections after the existing generic "Perturbation robustness"
  section in `analysis/06_predictive_validation_and_perturbation.ipynb`.
- Primary legacy perturbation output directory:
  `outputs/legacy_perturbation/`.

Required outputs:

- `outputs/legacy_perturbation/biological_perturbation_factor_table.csv`
- `outputs/legacy_perturbation/biological_parameter_perturbation_sweeps.csv`
- `outputs/legacy_perturbation/biological_parameter_direction_summary.csv`
- `outputs/legacy_perturbation/biological_parameter_pair_sweeps.csv`
- `outputs/legacy_perturbation/sigmoid_state_change_summary.csv`
- `outputs/legacy_perturbation/experimental_direction_match_summary.csv`
- `outputs/legacy_perturbation/phase_portrait_points.csv`
- `outputs/legacy_perturbation/analysis_summary.json`

Baseline categories to perturb:

- Use `outputs/legacy_mechanisms/legacy_mechanism_categories.csv`.
- Perturb configurations within each baseline category, not only pooled
  configurations.
- Required category grouping columns:
  - `sigmoid_state_at_sim_end_10_90`
  - `sigmoid_state_at_stim_end_10_90`
  - `temporal_recruitment_class`
  - `gj_ionic_state_10_90`
  - `buffering_phenotype` if present
  - `region` or `legacy_region_proxy` only if the source supports it
  - `condition`
  - `current_na`

Perturbation contexts:

- `MFA_like_from_control_legacy`
  - baseline source: legacy CONTROL configurations.
  - perturb: `P_gap_eff`, `gamma_s_eff`, `zth`, and `zs`.
  - claim-focused MFA summaries should prioritize `P_gap_eff`,
    `gamma_s_eff`, and `zth`; `zs` remains a required gating-shape diagnostic
    so gating-threshold and gating-slope effects are separable.
- `MFA_like_from_mfa_legacy`
  - baseline source: legacy MFA configurations, used to inspect category
    behavior already under MFA-like conditions.
  - perturb: `P_gap_eff`, `gamma_s_eff`, `zth`, and `zs`, with the same
    claim-focused priority as `MFA_like_from_control_legacy`.
- `MFA_BA_from_MFA_legacy`
  - baseline source: legacy MFA configurations.
  - perturb: sweep `gki` with the Filtered baseline `gki` fold grid. Treat
    folds below 1.0 as BA-like Kir reduction, with `0.75` as the first-pass
    single-contrast summary fold when a single contrast is required.
- `MFA_BA_stacked_on_control_legacy`
  - baseline source: legacy CONTROL configurations.
  - perturb: apply the configured MFA-like perturbation grid, then sweep `gki`
    with the Filtered baseline grid.

Protocol handling:

- Use the perturbation execution pattern from `analysis/Filtered_basline_sweep
  (1).ipynb`: resimulate selected legacy baselines, apply fold changes to one
  parameter or a parameter pair, extract Vm features, and write baseline,
  perturbed, and delta columns.
- Preserve the baseline simulation context from the source DB/condition/current
  unless the existing source code already exposes an explicit
  `protocol_condition` field. If `protocol_condition` is used, record both
  `baseline_condition` and `simulated_protocol_condition` in every output row.
- Do not introduce a separate Naris magnitude protocol, I-V voltage protocol, or
  movie protocol in this first pass.

One-dimensional perturbations:

- Primary MFA-like parameters:
  - `P_gap_eff`
  - `gamma_s_eff`
  - `zth`
  - `zs`
- Primary MFA+BA/Ba parameter:
  - `gki`
- Use fold-change grids from `analysis/Filtered_basline_sweep (1).ipynb` as the
  starting protocol, but translate `d` and `pk` into `P_gap_eff` for the main
  analysis.
- Required first-pass fold grids:
  - `gki`: `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`
  - `P_gap_eff`: use the shared `d/pk` fold grid
    `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`
  - `gamma_s_eff`: use the `gs` fold grid
    `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`
  - `zth`: `[0.5, 0.75, 1.0, 1.25, 1.5]`
  - `zs`: `[0.5, 0.75, 1.0, 1.25, 1.5]`
- Preserve the original raw `d` and `pk` one-at-a-time sweeps only as a
  `legacy_raw_factor_diagnostic`, because `d` and `pk` are structurally coupled.

Two-dimensional perturbations:

- Required pair sweeps:
  - `P_gap_eff x gamma_s_eff`
  - `P_gap_eff x zth`
  - `gamma_s_eff x zth`
  - `zth x zs`
  - `gki x P_gap_eff` for MFA+BA/Ba interaction checks
- Use a configurable grid. Start with a small 3x3 grid for tests and allow
  manuscript reruns to use a denser grid.
- Every pair-sweep row must include both fold factors and both perturbed values.

Per-row outputs required for every perturbation:

- Identity/provenance:
  - `source_scope`
  - `perturbation_context`
  - `baseline_condition`
  - `simulated_protocol_condition`
  - `baseline_candidate_id`
  - `db_name`
  - `trial_number`
  - `current_na`
  - `category_id`
  - `factor_source`
  - `legacy_selection_rule`
  - `legacy_configuration_status`
- Perturbation:
  - `perturbed_parameter`
  - `perturbation_factor`
  - `baseline_value`
  - `perturbed_value`
  - for 2D: `perturbed_parameter_1`, `factor_1`, `perturbed_parameter_2`,
    `factor_2`
- Membrane kinetics:
  - baseline and perturbed Vm features from `extract_features_from_trace`
  - signed deltas
  - direction labels: `increase`, `decrease`, `no_change`, `undefined`
- K_o kinetics:
  - baseline and perturbed K_o features
  - `Ko_efficiency_score`
  - `Ko_efficiency_quadrant`
  - signed deltas and direction labels
- Sigmoid state:
  - baseline and perturbed `sigmoid_state_at_stim_end_10_90`
  - baseline and perturbed `sigmoid_state_at_sim_end_10_90`
  - baseline and perturbed `temporal_recruitment_class`
  - `sigmoid_state_change`
  - `sigmoid_phase_transition`
- Simulation status:
  - `simulation_status`
  - `failure_reason`

Sigmoid state-change labels:

- `unchanged_closed`
- `unchanged_partial`
- `unchanged_open`
- `closed_to_partial`
- `closed_to_open`
- `partial_to_closed`
- `partial_to_open`
- `open_to_partial`
- `open_to_closed`
- `opened_during_stim_then_closed_by_end`
- `delayed_opening_after_load`
- `undefined_or_failed`

Direction matching against ATF data:

- Join simulated direction summaries to
  `outputs/features/experimental_kinetic_direction_targets.csv` and
  `outputs/features/region_specific_perturbation_direction_targets.csv`.
- Compare simulated membrane-kinetic direction to experimental direction for:
  - `CONTROL_to_MFA`
  - `MFA_to_MFA_BA`
  - `CONTROL_to_MFA_BA`
- Compare class-specific simulated responses to DH/VH regional delta-of-delta
  targets from `outputs/features/region_specific_perturbation_direction_targets.csv`
  and `outputs/features/experimental_second_layer/numeric_sweep_delta_of_delta_between_regions.csv`.
- Because the first-pass legacy configurations are not assigned to DH or VH,
  `regional_match_status` is a conservative alignment screen against the
  DH-minus-VH experimental delta target. It must not be interpreted as a
  simulated regional delta or region-specific phenotype assignment.
- Required columns:
  - `experimental_contrast`
  - `simulated_direction`
  - `experimental_direction`
  - `direction_match_status`: `match`, `opposite`, `no_clear_experimental_change`,
    `simulation_no_change`, `undefined`
  - `regional_experimental_direction`
  - `regional_target_scope`
  - `regional_match_status` for DH/VH perturbation-difference alignment.
  - `regional_match_interpretation`

## Step 06 Visualization Requirements

Create static, testable visual summaries before considering any animation.

Required plot/data products:

- Category-by-parameter heatmap of Vm direction changes.
- Category-by-parameter heatmap of K_o direction changes.
- Category-by-parameter heatmap of sigmoid state changes.
- 2D phase-space grid for each selected parameter pair:
  - control parameters: pair fold factors or perturbed effective values.
  - order parameter: sigmoid end state or sigmoid phase-transition label.
  - color or marker channel: `Ko_efficiency_quadrant`.
  - arrow/delta channel: EF score direction or K_o rise/decay direction.

Required CSVs behind plots:

- `outputs/legacy_perturbation/phase_portrait_points.csv`
- `outputs/legacy_perturbation/sigmoid_state_change_summary.csv`
- `outputs/legacy_perturbation/experimental_direction_match_summary.csv`

Deferred movie generation:

- Do not implement a movie or animation in this refined pass.
- Do not add placeholder movie outputs. Add this only after the expert defines
  frame variable, axes, visual encoding, file format, and runtime budget.

## Step 07: Optional Assumption Sensitivity

Only extend Step 07 if the new labels become claim-critical.

Additions, if needed:

- Test whether `Ko_efficiency_quadrant` and `sigmoid_state_change` are stable
  across configured gating families.
- Write:
  - `outputs/assumption_sensitivity/efficiency_label_sensitivity.csv`
  - `outputs/assumption_sensitivity/sigmoid_state_label_sensitivity.csv`
- Keep final biological claims disabled in Step 07.

## Step 08: Parameter Interpretation Gate

Chosen location:

- Module: `src/step08_parameter_plausibility.py` and
  `src/reviewer_gate_audits.py`.
- Notebook: existing parameter interpretation and constrained rerun sections.

Implementation requirements:

- Add semantic rows for newly claim-critical perturbation coordinates:
  - `P_gap_eff`: effective gap-coupling coordinate.
  - `gamma_s_eff`: available spatial transfer/surface-to-volume capacity proxy.
  - `zth`, `zs`: phenomenological gating coordinates.
  - `gki`: Kir conductance candidate, only directly biological with range,
    unit, citation, and identifiability support.
- Add columns to downstream audits indicating whether a perturbation claim uses:
  - effective-coordinate interpretation only;
  - raw direct physiology;
  - phenomenological gating coordinate;
  - reduced-model proxy.
- Do not allow `gamma_s_eff * Chi(t)` to be promoted to literal cell number.

## Step 09: Synthesis And Claim Gates

Chosen location:

- Module: `src/step09_reviewer_synthesis.py` and `src/reviewer_gate_audits.py`.
- Notebook: traceability, claim maturity, manifest, and restricted gates.

Required outputs:

- update `outputs/reviewer_synthesis/reviewer_traceability_table.csv`
- update `outputs/reviewer_synthesis/claim_maturity_table.csv`
- update `outputs/reviewer_synthesis/manuscript_asset_manifest.csv`
- add `outputs/reviewer_synthesis/mechanistic_pathway_perturbation_gate.csv`
- add `outputs/reviewer_synthesis/legacy_perturbation_claim_gate.csv`
- add `outputs/reviewer_synthesis/reviewer_remark_artifact_links.csv`
- add `outputs/reviewer_synthesis/degeneracy_scientific_value_statement.csv`

Claim gate requirements:

- A mechanistic class can be said to reproduce an experimental perturbation
  direction only if:
  - it is source-scoped to the correct legacy baseline set;
  - the baseline sigmoid category is defined;
  - the perturbation simulation succeeded;
  - Vm direction matches the ATF target;
  - K_o/EF direction is reported, even if it does not match a direct ATF target;
  - sigmoid state after perturbation is reported;
  - parameter interpretation is not blocked by Step 08.
- DH/VH class-specific claims require the region-specific target table from
  Step 02 and must remain restricted to supported contrasts.
- Step 09 must read upstream CSVs only. It must not run simulations.

## Deliberate Deferrals And First-Pass Assumptions

- Legacy baseline selection is defined for the first pass:
  `top_n_by_objective` with `legacy_top_n_requested = 300`. Do not block coding
  on a thresholded accepted-fit definition.
- Thresholded legacy good-enough fits can be added later using ATF variability
  thresholds and the Filtered baseline acceptance logic. Keep that as a separate
  selection status, not a replacement for the top-300 first pass.
- EF has no experimental class thresholds in the first pass. Save the continuous
  EF score and before/after EF deltas as primary outputs; use descriptive
  median-split EF classes only for grouping/visualization.
- Naris-derived MFA/MFA+BA perturbation magnitudes are not required for this
  pass. Use the Filtered baseline fold grid and record `factor_source`.
- I-V curves are out of scope for this pass.
- Biocytin/membrane-conductance data are out of scope for this pass.
- Movie generation is out of scope for this pass.
- The exact final expert-selected category subset is not required before
  implementation. Implement all unified sigmoid/temporal categories first and
  expose a config filter so the final subset can be selected later.
- Anatomical interpretation remains restricted: `gamma_s_eff * Chi(t)` and
  related activation products are reduced-model recruited-surface /
  functional-syncytium proxies unless external validation is added.

## Implementation Order

1. Update the affected step specs and reviewer-response implementation spec.
2. Add shared helpers for K_o kinetics, EF quadrant classification, sigmoid
   transition classification, and direction-of-change labels.
3. Extend Step 01 to write the legacy configuration library and candidate
   perturbation factors.
4. Extend Step 02 to write ATF experimental direction targets.
5. Extend Step 05 to write legacy mechanism categories, baseline sigmoid states,
   and baseline K_o efficiency mappings.
6. Extend Step 06 to run 1D and 2D MFA/MFA+BA biological perturbations by
   category, compare directions against ATF targets, and write visual summary
   CSVs.
7. Extend Step 08 semantic gates if the new perturbation claims need parameter
   interpretation support.
8. Extend Step 09 to register artifacts and gate claims.
9. Add tests before treating notebook plots/tables as reviewer-facing.

## Minimum Test Expectations

- Unit tests for K_o feature extraction:
  - finite rise/decay produces finite EF score;
  - flat or missing K_o trace produces explicit undefined status;
  - zero/near-zero decay denominator does not crash;
  - invalid array lengths raise descriptive errors.
- Unit tests for EF quadrant classification:
  - fast rise / fast decay;
  - slow rise / fast decay;
  - slow rise / slow decay;
  - fast rise / slow decay;
  - missing/nonfinite medians or values return undefined status;
  - perturbed rows use frozen baseline medians, not recomputed perturbed medians.
- Unit tests for sigmoid state classification:
  - closed, partial, open states;
  - opened during stimulation then closed by end;
  - delayed opening after load;
  - missing/nonfinite values produce `undefined_or_failed`.
- Unit tests for perturbing coordinates:
  - raw `gki`, `zth`, and `zs` perturbations validate positivity;
  - `P_gap_eff` and `gamma_s_eff` use `set_coordinate`;
  - unknown parameter raises a descriptive error;
  - 2D pair perturbations record both factors and both values.
- Acceptance tests:
  - Step 01 writes top-300 legacy configuration outputs with provenance fields,
    selection-rule fields, and no plain `accepted` column.
  - Step 02 writes experimental direction target outputs preserving region,
    condition, sweep, and feature.
  - Step 02 writes second-layer regional outputs and region-condition profile
    terms from the updated ATF notebook logic.
  - Step 05 writes legacy mechanism/category and K_o efficiency outputs.
  - Step 06 writes biological perturbation outputs with explicit simulation
    status and direction-match fields.
  - Step 09 manifest includes new artifacts and blocked claims are not upgraded.
