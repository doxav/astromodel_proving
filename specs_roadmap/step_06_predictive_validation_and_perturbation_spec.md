# Step 06 — Predictive validation, posterior predictive checks, and perturbation robustness

## Reviewer-response role

Step 06 operationalizes R6 by testing whether Step 04 cell-specific accepted ensembles and Step 05 candidate mechanism summaries remain useful beyond fitted traces.  A mechanism-distinct ensemble is **not** described as biological degeneracy unless it has auditable support from held-out-current prediction, posterior predictive feature checks, and perturbation robustness.

## Inputs

- `outputs/cell_fits/accepted_cell_ensembles.csv` from Step 04, with `file_id`, `region`, `condition`, `candidate_id`, effective parameters, acceptance flags, and held-out-current summary columns when present.
- `outputs/mechanisms/mechanism_clusters.csv` from Step 05, with mechanism labels and conservative claim scopes.  If the table is absent, the pipeline may run a small Step 05 pass to create mechanism labels.
- Step 02 feature thresholds and reliability weights:
  - `outputs/features/condition_region_sweep_thresholds.csv`
  - `outputs/features/region_pooled_condition_sweep_thresholds.csv`
  - `outputs/features/feature_reliability_weights.csv`
  - `outputs/features/feature_table_by_sweep.csv`

## Outputs

All outputs are written under `outputs/predictive_validation/`:

| File | Purpose |
|---|---|
| `heldout_current_errors.csv` | Per-candidate, per-sweep held-out prediction/error records from the Step 04 acceptance contract, plus low-to-high and high-to-low stress-test summaries when available. |
| `prediction_intervals.csv` | Accepted-ensemble predictive intervals by `region × condition × sweep × feature`. |
| `feature_distribution_ppc.csv` | Posterior predictive coverage of simulated features against empirical Step 02 bands with reliability weights. |
| `perturbation_sweeps.csv` | Nominal and perturbed hidden-output summaries by candidate, mechanism cluster, perturbation type, and sweep. |
| `robustness_summary.csv` | Mechanism/region/condition-level validation labels: `predictive_supported`, `prediction_limited`, `fit_only`, or `insufficient_evidence`. |
| `analysis_summary.json` | Machine-readable counts, configuration, and headline claim scope. |
| `performance_benchmark.csv` | Coarse/default runtime comparison to support notebook/test tuning decisions. |

## Scientific contract

1. **Held-out current prediction is primary.** Each reviewer-facing candidate must expose a row for every ordered current sweep `{50, 75, 100, 125, 150, 175}` nA with trace error, feature-pass fraction, and a status field.
2. **Posterior predictive checks are region-aware.** Primary PPC summaries are grouped by `region × condition × sweep × feature`; pooled thresholds are used only as explicit fallback.
3. **Perturbation robustness is mechanistic.** Perturbation rows carry Step 05 mechanism labels and hidden K-buffering metrics, not only raw parameters.
4. **Claims remain conservative.** Clusters failing prediction or perturbation are labeled `prediction_limited` or `fit_only`; clusters with too few cells/candidates remain `insufficient_evidence`. Step 06 reports `step06_screen_claim` separately from `final_biological_degeneracy_claim_allowed`, which defaults to `False`.
5. **No silent dropping.** Failed simulations produce explicit rows with `simulation_status = failed` and a `failure_reason`. Missing Step 05 mechanism labels, incomplete Step 04 held-out screens, and perturbations blocked by simulator API limitations are represented with explicit status fields.

## Perturbations

The default lightweight perturbation panel tests:

- bath coupling `epsilon` scaling (`eps_scale_low`, `eps_scale_high`);
- stimulus duration / observation-window sensitivity (`stimulus_duration_short`, `stimulus_duration_long`);
- baseline extracellular potassium (`baseline_K_o_low`, `baseline_K_o_high`);
- current-amplitude drive scaling through the bath-drive middle value (`current_scale_low`, `current_scale_high`).

The default tests/notebook use a coarse grid; manuscript reruns may increase `time_points`, candidate count, and perturbation factors.

## Gherkin specifications

```gherkin
@step06 @R6 @heldout-current
Feature: accepted ensembles predict held-out currents
  Scenario: a held-out sweep is predicted from the remaining sweeps
    Given accepted cell-specific candidates from Step 04
    When Step 06 aggregates held-out-current predictions
    Then trace error and feature-pass metrics are reported for all six sweeps
    And rows are summarized by region, condition, and sweep
```

```gherkin
@step06 @R2 @posterior-predictive
Feature: simulated feature distributions are compared with empirical distributions
  Scenario: accepted ensemble predictions match empirical feature bands
    Given empirical Step 02 thresholds
    And accepted ensemble simulations
    When posterior predictive feature checks are computed
    Then coverage is reported by region, condition, sweep, and feature
    And redundant or low-reliability features are weighted accordingly
```

```gherkin
@step06 @R6 @perturbation
Feature: mechanism regimes are stress-tested by perturbation
  Scenario: accepted mechanism clusters are simulated under altered inputs
    Given mechanism-labeled accepted ensembles
    When bath coupling, stimulus duration, baseline K_o, or current-amplitude drive are perturbed
    Then functional buffering metrics are reported
    And clusters are classified by robustness without upgrading weak evidence to degeneracy
```

## Tests required

- Bootstrap:
  - Step 04/05 inputs can be merged without losing `file_id`, `region`, `condition`, `sweep`, or mechanism labels.
  - Prediction interval construction returns finite quantiles with provenance counts.
  - Perturbation rows include explicit simulation statuses and hidden K-buffering metrics.
- Acceptance:
  - Running Step 06 writes all required CSV/JSON outputs.
  - PPC rows include `region`, `condition`, `sweep`, `feature`, empirical interval bounds, coverage, and reliability weight.
  - Robustness labels never use `candidate_degenerate_regimes` and downgrade unsupported clusters.
- Integration:
  - `analysis/06_predictive_validation_and_perturbation.ipynb` executes from the repository root and saves an executed copy.
  - Region/condition/sweep summaries are coherent with candidate-level validation rows.
- Performance:
  - A one-candidate coarse Step 06 run completes within a practical runtime budget.
  - `compare_step06_runtime_presets` records elapsed time and a tuning recommendation for coarse/default grids, with `compare_step06_performance` kept as a backward-compatible alias.

## Notebook contract

`analysis/06_predictive_validation_and_perturbation.ipynb` must include an Open-in-Colab badge at the top, run without Google Drive dependencies, write machine-readable outputs, and demonstrate:

1. accepted ensemble and mechanism-label inventory;
2. held-out-current error table;
3. prediction interval table and figure;
4. feature posterior predictive coverage table;
5. perturbation robustness table and plot;
6. mechanism/region/condition robustness summary;
7. explicit claim-scope text explaining what is and is not supported after Step 06.
