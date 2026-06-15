# Step 02 specification — Rebuild region-aware ATF thresholds from the 37 cells

## Step purpose

Step 02 turns the 37 ATF current-clamp files into the **experimental uncertainty model** used by the later fitting and accepted-ensemble steps. The old `threshold_for_good_enough_fits.csv` is useful as a historical artifact, but it is not reviewer-facing because it does not encode pharmacological condition and does not protect against DH/VH pooling.

This step therefore replaces the old threshold logic with a **region-aware, condition-aware, sweep-aware feature pipeline**.
It also exports experimental direction-of-change targets under CONTROL, MFA,
and MFA+BA so Step 06 can compare simulated perturbation directions against the
observed membrane-kinetic trends.

## Why this step matters

This step primarily addresses reviewer objections that the manuscript does not quantify experimental variability, noise, or data constraints, and that fitting Vm alone can easily overweight weak or redundant observables.

### Reviewer critique coverage

| Critique ID | Coverage level | How Step 02 responds | How it will be verified |
|---|---|---|---|
| R2 — experimental variability, noise, uncertainty, and data constraints are underspecified | **Strong** | Build a sweep-level feature table for all 37 cells and 222 sweeps; preserve DH/VH and condition labels; write threshold tables and missingness/reliability tables. | Counts, feature completeness, threshold tables, notebook plots, and tests. |
| R4 — Vm-only fitting weakly constrains ionic dynamics and some features are weak or non-physiological | **Partial** | Down-weight missing or redundant features so later fitting does not treat all observables as equally informative. | Reliability-weight table; peak vs stim_end redundancy diagnostics; missingness by condition. |
| R7 — figures/tables are unclear | **Partial** | Create clean machine-readable tables and a reproducible validation notebook with explicit units and groupings. | Notebook outputs and integration test. |
| R5 — distinct phenotypes/pathways are not linked to experimental structure | **Partial, preparatory only** | Preserve region and condition in every row so later mechanism summaries can be tied back to the experimental design. | Output schemas and tests that forbid region loss. |

This step does **not** yet prove degeneracy or robustness. It supplies the empirical threshold and weighting layer required before those claims can be evaluated.

## Updated-spec differences that materially change Step 02

Relative to the earlier plan, the updated implementation spec tightens Step 02 in four ways:

1. **Region is now a hard contract, not a convenience label.** Every output row must keep `file_id`, `region`, `condition`, and `sweep`.
2. **Primary thresholds must be `region × condition × sweep × feature`.** Region-pooled or global-pooled tables are sensitivity/fallback outputs only.
3. **Small strata must be explicit.** The `VH-control` stratum has 4 cells and must be flagged rather than silently treated as precise.
4. **Performance must be measured, not guessed.** The implementation must compare the practical value of optional acceleration and decide whether it should remain optional.
5. **Perturbation targets must be reusable source outputs.** The updated ATF
   regional/condition perturbation analyses are ported into source code and CSV
   outputs rather than remaining notebook-only calculations.

## Scientific objectives

### Objective S02.1 — Build the canonical ATF feature table

Create one row per `file_id × sweep`, with robustly extracted kinetic features and explicit biological factors:

- `file_id`
- `region` in `{DH, VH}`
- `condition` in `{CONTROL, MFA, MFA_BA}`
- `sweep` in `{1,2,3,4,5,6}`

Features must include at least:

- `peak_depolarization_mV`
- `stim_end_depolarization_mV`
- `rise_slope_mV_per_s`
- `rise_tau_s`
- `plateau_level_mV`
- `plateau_slope_mV_per_s`
- `decay_slope_mV_per_s`
- `decay_tau_s`
- `undershoot_magnitude_mV`
- `return_slope_mV_per_s`
- `plateau_reached`
- `has_undershoot`

This is the canonical table for later Step 04 and Step 06 inputs.

#### Gherkin

```gherkin
@step02 @R2 @feature-table
Feature: canonical ATF feature table
  Scenario: all 37 ATF cells yield sweep-level rows with experimental factors
    Given 37 ATF files in data/2_K+ Pumps Data
    When the step-02 feature pipeline runs
    Then it writes exactly 222 sweep-level rows
    And every file_id contributes exactly 6 sweeps
    And every row contains file_id, region, condition, and sweep
    And region values are DH or VH only
    And condition values are CONTROL, MFA, or MFA_BA only
```

### Objective S02.2 — Rebuild primary thresholds with region preserved

Build the primary threshold table with rows indexed by:

- `threshold_scope = region_specific`
- `region`
- `condition`
- `sweep`
- `feature`

Each row must include:

- `n_cells`
- `n_nonmissing`
- `mean`, `median`, `std`, `min`, `q1`, `q3`, `max`, `iqr`
- `ci95_low`, `ci95_high`
- `acceptable_low_q1`, `acceptable_high_q3`
- `acceptable_low_ci95`, `acceptable_high_ci95`
- `small_stratum`
- `missing_rate`

Additionally build two sensitivity tables:

- `threshold_scope = region_pooled`
- `threshold_scope = global_pooled`

These are not primary reviewer-facing thresholds.

#### Gherkin

```gherkin
@step02 @R2 @region-thresholds
Feature: region-aware thresholds are primary
  Scenario: threshold rows retain DH and VH rather than silently pooling them
    Given the canonical feature table
    When thresholds are built
    Then the primary threshold table uses threshold_scope region_specific
    And each primary row is indexed by region, condition, sweep, and feature
    And region_pooled and global_pooled outputs are written only as labeled sensitivity tables
```

### Objective S02.3 — Quantify feature reliability, not just feature presence

Build a reliability table that combines at least:

- `missing_rate`
- `coverage_weight = 1 - missing_rate`
- `redundancy_penalty`
- `reliability_weight`
- `recommended_for_primary_loss`

Key scientific rule:

- `peak_depolarization_mV` and `stim_end_depolarization_mV` must not be treated as fully independent if they are near-collinear.
- `return_slope_mV_per_s` must be down-weighted when missingness is high, especially under `MFA_BA`.

#### Gherkin

```gherkin
@step02 @R2 @R4 @feature-reliability
Feature: feature reliability controls later fitting weights
  Scenario: missing and redundant features are down-weighted
    Given the canonical feature table
    When reliability weights are computed
    Then highly missing features receive lower reliability weights
    And highly correlated features are flagged as redundant
    And stim_end_depolarization_mV is down-weighted when redundant with peak_depolarization_mV
```

### Objective S02.4 — Quantify DH/VH regional effects before model-fitting claims

Build a `region_effect_summary.csv` that reports DH vs VH contrasts for each:

- condition
- sweep
- key feature

Minimum fields:

- `n_cells_DH`, `n_cells_VH`
- `dh_median`, `vh_median`
- `dh_minus_vh_median`
- `dh_minus_vh_ci95_low`, `dh_minus_vh_ci95_high`
- `small_stratum`

This is **not** a claim of phenotype. It is a guardrail preventing region-blind fitting from hiding systematic differences.

#### Gherkin

```gherkin
@step02 @R2 @region-effects
Feature: regional differences are quantified before fitting claims
  Scenario: DH and VH summaries exist for key kinetic features
    Given the canonical feature table
    When region-effect summaries are computed
    Then each key feature has DH and VH summary statistics by condition and sweep
    And small strata are flagged explicitly
    And region-blind pooling appears only as a sensitivity analysis, not the primary output
```

### Objective S02.5 — Measure whether numba acceleration is necessary for Step 02

The user pointed out that an extended model (`astrosim`) uses `numba.njit` in a performance-sensitive path. Step 02 must therefore **benchmark first** rather than assuming that acceleration is needed.

#### Design decision criteria

Step 02 is allowed to keep a pure NumPy default if the benchmark shows that:

- ATF parsing and artifact preprocessing dominate end-to-end runtime, and
- numba speeds up only a small compute sub-block, giving little overall gain.

Step 02 should switch to a numba default only if both conditions hold:

- compute-stage speedup is materially positive, and
- estimated end-to-end gain is large enough to matter for notebook/test execution.

For this implementation, the benchmark contract is:

- compare NumPy vs warmed numba on cached, preprocessed traces;
- keep NumPy default unless numba gives at least **15% compute speedup** and at least **1 second estimated end-to-end gain**.

This avoids making numba a mandatory dependency when the real bottleneck is file parsing and artifact preprocessing rather than feature kernels.

#### Gherkin

```gherkin
@step02 @performance @numba-decision
Feature: step-02 acceleration is benchmark-driven
  Scenario: numba is enabled only if it improves end-to-end runtime materially
    Given the ATF step-02 pipeline
    When the benchmark compares numpy and warmed numba feature extraction
    Then the benchmark writes elapsed times and a tuning decision
    And the default engine remains numpy unless the configured speedup and end-to-end gain thresholds are met
```

### Objective S02.6 — Export experimental perturbation direction targets

Build target tables from the canonical ATF feature table for later comparison
against simulated MFA-like and MFA+BA-like perturbations:

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

Required contrasts are `CONTROL_to_MFA`, `MFA_to_MFA_BA`, and
`CONTROL_to_MFA_BA`. Direction labels are `increase`, `decrease`,
`no_clear_change`, or `undefined`. Follow-up FDR adjustment must preserve NaN
p-values rather than converting them into false significant or nonsignificant
entries.

#### Gherkin

```gherkin
@step02 @R2 @perturbation-targets
Feature: experimental perturbation directions are exported for Step 06
  Scenario: ATF condition contrasts produce reusable direction targets
    Given the canonical feature table
    When experimental perturbation targets are computed
    Then CONTROL_to_MFA, MFA_to_MFA_BA, and CONTROL_to_MFA_BA rows are written
    And DH-minus-VH delta-of-delta targets are written where the data support them
    And each target row has a signed direction label and sample-size support columns
```

## Technical objectives

### T02.1 — Robust ATF parsing

Implement a local-file parser for `.atf` files that:

- reads header metadata and tabular data;
- resolves duplicate column names safely;
- builds a sweep map using `Signals` metadata when available;
- fails explicitly on malformed or unknown region labels.

### T02.2 — Conservative artifact preprocessing

Port the artifact correction logic required to keep the 37-cell dataset usable:

- brief jump repair;
- isolated outlier repair;
- automatic transient-interval detection;
- optional manual transient windows for known problematic sweeps.

Preprocessing must write QC fields:

- `n_corrected_points`
- `fraction_corrected`
- `n_auto_intervals`
- `n_manual_intervals`

### T02.3 — Reproducible feature extraction

Implement a deterministic `extract_features(...)` function that takes a preprocessed parsed ATF structure and returns one row per sweep.

The implementation must keep the cleaner semantics from the improved notebook:

- end-of-stimulus level is separate from plateau level;
- decay is computed from `stim_end_level` when possible;
- plateau detection affects plateau outputs but does not censor decay or return by default;
- `return_slope` is only reported when an actual undershoot-and-return pattern is present.

### T02.4 — Structured CSV outputs

Write outputs under `outputs/features/`:

- `atf_region_condition_inventory.csv`
- `feature_table_by_sweep.csv`
- `preprocess_qc_by_sweep.csv`
- `region_condition_cell_counts.csv`
- `condition_region_sweep_thresholds.csv`
- `feature_reliability_weights.csv`
- `feature_correlation_summary.csv`
- `region_effect_summary.csv`
- `performance_benchmark.csv`
- `analysis_summary.json`

## Approaches to compare and how to choose

| Problem | Candidate approaches | Decision rule |
|---|---|---|
| Threshold construction | region-specific, region-pooled, global-pooled | Use `region_specific` as primary. Use pooled tables only as labeled sensitivity/fallback outputs. |
| Summary bounds | IQR-based bounds and CI95-based bounds | Export both; use IQR-based bounds as the main robust default for pass/fail screening. |
| Region effect estimation | mixed model, clustered model, bootstrap cell-level contrasts | Use bootstrap contrasts now because the design is cell-level, unpaired, and small-stratum aware. |
| Redundancy handling | drop one feature, keep both with penalty, PCA compression | Keep both but apply redundancy penalties so downstream loss design remains explicit and reviewer-readable. |
| Performance | pure NumPy, warmed numba for feature kernels | Keep NumPy default unless benchmark shows meaningful end-to-end improvement. |

## Non-goals of Step 02

Step 02 must **not**:

- claim biological degeneracy;
- claim mechanistic phenotypes;
- collapse DH and VH into one primary threshold table;
- hide small-stratum uncertainty;
- depend on Colab, Google Drive, or interactive widgets.

## Test-first implementation plan

### 1. Bootstrap tests to add first

Files:

- `tests/bootstrap/test_step02_bootstrap.py`

These tests must verify:

- required inputs exist;
- parser resolves representative region/condition labels;
- ATF inventory count contract is correct;
- a representative file can be parsed, preprocessed, and featured into 6 rows.

### 2. Acceptance tests to add next

Files:

- `tests/acceptance/test_step02_acceptance.py`

These tests must verify:

- the full step pipeline writes outputs;
- the canonical feature table has 222 rows and 37 file IDs;
- threshold scope row counts are correct;
- reliability weights actually capture redundancy and missingness;
- `CONTROL` regional effects are flagged as small-stratum because `VH-control` has 4 cells.

### 3. Performance tests

Files:

- `tests/performance/test_step02_performance.py`

These tests must verify:

- the performance benchmark runs and writes a decision;
- elapsed times are finite and within generous notebook-safe limits;
- the decision logic remains explicit even when numba is unavailable.

### 4. Integration test

Files:

- `tests/integration/test_step02_integration.py`

This test must execute the notebook and verify that the key output CSV files were written.

## Notebook specification

Notebook file:

- `analysis/02_rebuild_atf_thresholds.ipynb`

The notebook must:

1. load the local project root;
2. run the full step-02 pipeline;
3. display the region-condition cell count contract;
4. compare legacy threshold structure versus new threshold structure;
5. show at least one kinetic feature across `region × condition × sweep`;
6. show redundancy diagnostics for `peak_depolarization_mV` versus `stim_end_depolarization_mV`;
7. show reliability-weight summaries;
8. show region-effect summaries;
9. show the benchmark-driven NumPy/numba decision;
10. save machine-readable outputs under `outputs/features/`.

## Acceptance criteria

Step 02 is complete when all of the following are true:

1. All bootstrap, acceptance, performance, and integration tests pass.
2. The notebook executes top-to-bottom from the repository root.
3. The primary threshold table is `region_specific` and contains the expected DH/VH strata.
4. The feature table contains 222 rows and preserves `file_id`, `region`, `condition`, and `sweep` in every row.
5. Reliability weights explicitly down-weight at least the peak/stim_end redundancy and the high-missingness `return_slope` feature.
6. The performance benchmark writes a decision and justifies whether numba should remain optional or become the default for this step.

## Traceability back to the main roadmap

This step operationalizes the Step 02 requirements of the updated reviewer-response implementation spec and prepares the prerequisites for:

- Step 04 accepted-ensemble mechanism screening;
- Step 06 cell-specific six-sweep fitting and predictive validation;
- Step 09 reviewer-facing figures and threshold tables.
