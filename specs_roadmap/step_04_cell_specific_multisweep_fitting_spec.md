# Step 04 — Cell-specific six-sweep fitting and accepted ensemble construction

This specification defines the first reviewer-facing fitting step that no longer treats historical single-current Optuna trials as the primary scientific object. The goal is to fit one **cell** across its six ordered pump-current sweeps with one shared mechanistic parameter set, then convert those fitted candidates into a cell-specific accepted ensemble that can later support mechanistic decomposition.

## Scientific role in the full roadmap

This step sits after:

- Step 00: provenance audit;
- Step 01: historical SQLite post-fit/effective-parameter extraction;
- Step 02: region-aware ATF thresholds and reliability weights;
- Step 03: structural/practical identifiability and effective-parameter guardrails.

This step must therefore treat as already established:

1. `P_gap_eff = d × pk` is the primary gap-junction coordinate.
2. `gamma_t_eff`, `gamma_s_eff`, and `volume_ratio_wa_wo` are the primary reporting coordinates.
3. DH and VH remain first-class biological factors.
4. Legacy historical DBs are useful for seeding/debug only, not as final reviewer-facing accepted ensembles.

## Reviewer objections targeted

| Reviewer objection | Coverage at this step |
|---|---|
| R1 — degeneracy vs non-identifiability/sloppiness | Partial. The step uses Step 03 guardrails and builds accepted cell-specific ensembles rather than relying on single-current fit multiplicity. |
| R2 — experimental variability/noise/uncertainty | Strong partial. The step applies region-aware, sweep-aware, reliability-weighted feature contracts when deciding whether a fitted candidate is accepted. |
| R4 — Vm-only fit weakly constrains ionic dynamics | Partial. The step forces one shared cell parameter set across six sweeps and evaluates fits through trace and feature contracts rather than raw parameter plausibility alone. |
| R5 — different regimes are not linked to distinct pathways | Prerequisite only. This step creates the accepted cell-specific ensembles that Step 05 will later decompose mechanistically. |
| R6 — robustness beyond fitted traces | Partial prerequisite. This step must output ensembles and shared cell fits that Step 06 can evaluate under held-out currents and perturbations. |
| R7 — clarity/organization | Partial. This step must write clean machine-readable outputs and a validation notebook. |

## Scientific objectives

1. Fit each ATF file as one **independent cell** with one shared parameter set across all six sweeps.
2. Preserve the `region × condition × sweep` feature-contract structure from Step 02.
3. Construct a cell-specific accepted ensemble, not just a single best fit.
4. Ensure accepted candidates respect basic current-order monotonicity of depolarization.
5. Make explicit which cells are accepted, rejected, or unresolved.
6. Export accepted candidates in a format directly consumable by Step 05 mechanism decomposition and Step 06 predictive validation.

## Primary outputs

| Path | Purpose |
|---|---|
| `outputs/step04_cell_specific_multisweep/accepted_candidates.csv` | Cell-specific accepted candidate parameter sets. |
| `outputs/step04_cell_specific_multisweep/accepted_sweep_scores.csv` | Per-sweep trace/feature scores for accepted candidates. |
| `outputs/step04_cell_specific_multisweep/accepted_feature_contracts.csv` | Per-feature threshold-pass table for accepted candidates. |
| `outputs/step04_cell_specific_multisweep/fit_status_by_cell.csv` | One row per cell summarizing whether the cell has an accepted candidate. |
| `outputs/step04_cell_specific_multisweep/accepted_ensemble_summary.csv` | Summary by condition and region. |
| `outputs/step04_cell_specific_multisweep/seed_summary_by_condition.csv` | Condition-level effective-parameter seeds derived from legacy best-fit CSVs. |
| `outputs/step04_cell_specific_multisweep/performance_benchmark.csv` | Coarse-vs-default timing/acceptance comparison used to tune this step. |
| `outputs/step04_cell_specific_multisweep/analysis_summary.json` | Machine-readable notebook/test summary. |

## Technical objectives

- Parse all 37 ATF files and group each file’s six sweeps into one cell object.
- Use one shared cell parameter vector per candidate.
- Use effective parameters as fitting coordinates:
  - `P_gap_eff`
  - `gamma_t_eff`
  - `gamma_s_eff`
  - `volume_ratio_wa_wo`
- Include additional constrained nuisance/effective coordinates needed for trace quality:
  - `g_kir`
  - `gl_a`
  - `zth`
  - `zs`
  - `eps`
  - `k_bath_gain`
- Score each candidate by combining:
  - trace agreement on a common comparison grid;
  - reliability-weighted feature threshold agreement;
  - monotonicity of predicted peak depolarization across sweeps.
- Build leave-one-cell-out thresholds for the target cell to avoid trivial self-inclusion.
- Keep at most `max_accepted_per_cell` candidates per cell.
- Mark cells with no accepted candidate as `rejected_but_ranked` rather than failing silently.
- Benchmark at least two practical tuning presets (for example `coarse` and `default`) on a small subset and write a machine-readable recommendation.

## Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Legacy single-current DBs | Seeding only | Allowed for candidate initialization and debugging. |
| Condition-median effective seeds | Primary initialization | Default starting point for cell-specific search. |
| Random/log-space candidate perturbation around seeds | Required | Used to construct accepted cell ensembles without assuming raw-parameter identifiability. |
| Shared six-sweep cell fit | Required | One parameter vector per cell; no sweep-specific mechanistic parameters. |
| Per-sweep protocol timing from observed ATF trace | Required | Use observed onset/offset estimates to align model stimulation windows. |
| Region-specific thresholds | Primary acceptance criterion | Use for reviewer-facing acceptance. |
| Region-pooled/global thresholds | Sensitivity only | Only for debugging or explicit fallback. |
| Coarse vs default tuning presets | Required benchmark | Keep the faster preset only if acceptance behavior stays close enough to the default. |
| Ordered cell subset | Deterministic debugging | Useful for exact reproducibility when diagnosing one condition block. |
| Group-balanced cell subset | Notebook/demo/test runtime control | Prefer when `max_cells` is used in a validation notebook so small runs still cover multiple `condition × region` groups. |

## Acceptance criteria for a candidate fit

A candidate is accepted when all of the following hold:

1. `mean_weighted_pass_fraction >= feature_mean_pass_threshold`
2. `min_weighted_pass_fraction >= feature_min_sweep_pass_threshold`
3. `mean_trace_nrmse <= max_mean_trace_nrmse`
4. predicted peak depolarization is monotonic non-decreasing with current across the six sweeps
5. no sweep simulation fails
6. at least 5/6 sweeps satisfy the per-sweep pass criterion

These thresholds must be configurable and reported in `analysis_summary.json`.

## Outputs required for downstream steps

### For Step 05 mechanistic decomposition

`accepted_candidates.csv` must include at least:

- `file_id`
- `region`
- `condition`
- `candidate_id`
- `accepted`
- `P_gap_eff`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`
- `g_kir`
- `gl_a`
- `zth`
- `zs`
- `eps`
- `k_bath_gain`
- `switching_function`

### For Step 06 predictive validation

`accepted_sweep_scores.csv` must include at least:

- `file_id`
- `sweep`
- `current_na`
- `trace_nrmse`
- `weighted_pass_fraction`
- `pred_peak_depolarization_mV`
- `simulation_failed`

## Gherkin specifications

```gherkin
@step04 @R2 @R4 @cell-specific-fit
Feature: one cell is fit jointly across six sweeps
  Scenario: shared parameters explain all six ordered currents
    Given one ATF file with six sweeps
    When the step 04 fitting pipeline evaluates cell-level candidates
    Then every candidate uses one shared mechanistic parameter set across all six sweeps
    And the output stores per-sweep trace and feature scores
    And the cell-level status is accepted, rejected_but_ranked, or failed
```

```gherkin
@step04 @R2 @leave-one-out-thresholds
Feature: the acceptance contract excludes the target cell from its threshold source
  Scenario: leave-one-cell-out thresholds are used during fitting
    Given a target cell in one region and condition
    When thresholds are built for that target cell
    Then the target cell is excluded from the threshold statistics
    And the threshold scope is leave-one-cell-out region-specific
```

```gherkin
@step04 @R4 @current-order
Feature: accepted candidates preserve the ordered current response
  Scenario: depolarization should not decrease with stronger pump-current sweeps
    Given a fitted candidate for one cell
    When predicted peak depolarization is computed across sweeps 1 to 6
    Then the predicted peaks are monotonic non-decreasing within tolerance
    And non-monotonic candidates are not accepted
```

```gherkin
@step04 @performance @tuning
Feature: step-04 tuning is benchmark-driven
  Scenario: coarse and default presets are compared on a small subset
    Given the step 04 fitting pipeline
    When the benchmark compares coarse and default presets on a fixed cell subset
    Then the benchmark writes elapsed time and acceptance statistics
    And the recommended default preset is recorded in performance_benchmark.csv
```

```gherkin
@step04 @R5 @downstream-interface
Feature: accepted cell candidates are exportable to mechanism decomposition
  Scenario: accepted candidates provide effective coordinates and status fields
    Given a successful step 04 run
    When accepted_candidates.csv is written
    Then each row contains file_id, region, condition, candidate_id, accepted, and effective parameters
    And the file can be read without needing notebook state
```

## Validation notebook required

`analysis/04_cell_specific_multisweep_fitting.ipynb`

The notebook must show:

- the fitted-cell count by region and condition;
- accepted-cell count by region and condition;
- summary of best pass fraction and trace error;
- at least one observed-vs-predicted multi-sweep panel for each condition represented in the executed run;
- a short interpretation block stating that this step constructs accepted ensembles but does not yet claim mechanism diversity or predictive robustness.

The notebook may use a small `max_cells` override for runtime control, but when it does so it should prefer a **group-balanced** subset over a purely ordered slice so the executed notebook remains scientifically informative across `condition × region` groups.
- a coarse-vs-default performance/tuning comparison table.

## Tests required

- bootstrap/tuning tests:
  - seed construction;
  - more candidates should not worsen the best objective on a fixed cell under a fixed RNG seed.
- acceptance tests:
  - output schemas;
  - region/condition/file/sweep contract;
  - accepted candidates carry required columns.
- performance tests:
  - coarse-vs-default benchmark writes elapsed times and a recommendation;
  - the benchmark output contains acceptance and runtime fields needed for tuning decisions.
- integration tests:
  - full step run on a small cell subset;
  - notebook execution in fast mode;
  - outputs written under `outputs/step04_cell_specific_multisweep/`.
