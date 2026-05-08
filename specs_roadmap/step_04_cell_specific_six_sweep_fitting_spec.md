# Step 04 — Cell-specific six-sweep fitting and accepted ensemble construction

This step builds the first reviewer-facing accepted ensembles by fitting the **expected astrocyte ODE model** jointly across the six ordered pump-current sweeps of each cell. The goal is to move from historical single-current fits to **cell-specific, six-sweep, region-aware** inference targets that can later support mechanistic decomposition and predictive validation.

## Scientific role in the reviewer-response pipeline

Step 03 established that raw Vm-compatible parameter multiplicity must not be interpreted as biological degeneracy without further evidence. Step 04 addresses that gap by requiring one shared cell-level mechanism to explain six sweeps of the same cell under the expected model. This is the first step that creates reviewer-facing accepted ensembles suitable for downstream mechanism decomposition.

### Critiques targeted

| ID | Targeted response in Step 04 |
|---|---|
| R2 | Uses region-aware empirical thresholds and reliability-weighted feature contracts from Step 02. |
| R4 | Reduces overfitting by fitting one shared mechanism across six sweeps instead of one fit per current. |
| R6 | Introduces leave-one-sweep-out / held-out-current screening as part of the acceptance contract. |
| R5 (preparatory) | Produces cell-specific accepted ensembles that Step 05 can decompose mechanistically. |
| R1 (protective) | Prevents single-current non-identifiability from being mistaken for cell-level degeneracy. |

## Model contract

The fitted dynamic model is the reviewer-facing astrocyte ODE model with state

- `Va`
- `DK_a_t`
- `K_a_s`
- `Kg`

and currents / dynamics:

- `I_Kir`
- `I_k_a`
- `I_l_a`
- `I_kgap`
- `dVa`
- `dDK_a_t`
- `dK_a_s`
- `dKg`

The implementation must preserve the expected equation structure, including:

- `P_kgap = d_gap * P_k`
- switching function families `sigmoid`, `tanh`, `hill`
- optional `epsilon_middle` and `w_o_middle` modifications when `idx == 1`
- external `K_bath` protocol as a known time/value schedule driven by the sweep current

Numerical safeguards are allowed only to avoid invalid logarithms/divisions/overflow; they must not change the intended equation structure.

## Inference contract

### Primary fit unit

- one ATF file = one cell
- one cell = six ordered sweeps
- one candidate fit = one shared cell-level parameter set

### Known protocol inputs

For each sweep:

- current level is known from sweep order
- `K_bath` schedule is determined by condition and current level

### Optimization coordinates

Use effective parameters as primary inference coordinates:

- `P_gap_eff`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`

and optimize additional raw/nuisance terms only where needed for fit quality:

- `gki`
- `eps`
- `gl_a`
- `zth`
- `zs`

Raw `d` and `pk` are not optimized independently for interpretation; they are reconstructed from `P_gap_eff`.

## Acceptance contract

A candidate is evaluated with three layers:

1. **All-six fit quality**
   - baseline-subtracted trace agreement across six sweeps
   - reliability-weighted feature agreement using Step 02 thresholds
   - simulation health must remain finite
2. **Held-out sweep screening**
   - fit on 5 sweeps
   - predict the held-out 6th sweep
   - rotate across all 6 held-out choices
3. **Cell-level reviewer-facing status**
   - at least one accepted all-six candidate
   - held-out pass count above configured minimum

### Default acceptance rules

| Criterion | Scope | Rule |
|---|---|---|
| `mean_trace_rmse_mV` | all6 | `<= trace_rmse_accept` |
| `mean_weighted_pass_fraction` | all6 | `>= feature_pass_accept` |
| `heldout_trace_rmse_mV` | leave-one-out | `<= heldout_trace_rmse_accept` |
| `heldout_weighted_pass_fraction` | leave-one-out | `>= heldout_pass_accept` |
| `holdout_pass_count` | cell | `>= heldout_min_pass_count` |

## Technical objectives

- Fit each selected cell jointly across six sweeps with one shared parameter vector.
- Use the expected ODE model from `src/astro_model.py`.
- Use Step 02 region-aware threshold tables and reliability weights.
- Support optional historical verified seeds, but do not require them.
- Save candidate-level, sweep-level, held-out, and cell-level summary tables.
- Save machine-readable summary JSON.
- Save an executed notebook under `outputs/executed_notebooks/`.

## Output files

Required outputs under `outputs/cell_fits/`:

- `cell_fit_candidates.csv`
- `accepted_cell_ensembles.csv`
- `candidate_sweep_metrics.csv`
- `cell_fit_quality_summary.csv`
- `heldout_current_screen.csv`
- `acceptance_contract.csv`
- `cell_trace_inventory.csv`
- `analysis_summary.json`

## Tests required

### Unit

- exact alignment of `src.astro_model.model` with the expected reference equations
- selected-file inventory returns the expected six sweeps
- accepted-fit schema contains required columns

### Acceptance

- representative CONTROL cell can become reviewer-facing under audit/demo settings
- representative MFA cell can become reviewer-facing under audit/demo settings

### Resolution / stability

- control-cell fit remains qualitatively stable across nearby fitting resolutions

### Performance / tuning

- one-cell fit completes within practical time on compact settings
- 10-point grid is not catastrophically slower than 8-point grid on the same cell

### Integration

- Step 04 writes the required outputs
- summary JSON advertises model alignment and Step 02 threshold usage
- executed notebook exists for audit

## Notebook requirements

Notebook: `analysis/04_cell_specific_six_sweep_fitting.ipynb`

It must include:

1. Open in Colab badge
2. local/Colab setup cell
3. explicit model-alignment audit against the expected equations
4. compact Step 04 run on a representative subset using runtime-safe settings
5. display of acceptance contract, accepted ensembles, held-out screen, and summary
6. explicit claim boundary:
   - Step 04 creates accepted ensembles
   - Step 04 does not yet claim biological degeneracy
   - Step 05 is required for mechanism decomposition
   - Step 06 is required for predictive robustness claims

## Gherkin specifications

```gherkin
@step04 @R4 @model-alignment
Feature: Step 04 fits use the expected reviewer-facing astrocyte model
  Scenario: the implemented ODE matches the expected equations
    Given a parameter dictionary, state vector, and protocol time
    When src.astro_model.model is compared with the reference model equations
    Then the derivatives are numerically identical within floating-point tolerance
```

```gherkin
@step04 @R2 @R4 @six-sweep-fit
Feature: one shared mechanism is fitted across the six sweeps of a cell
  Scenario: a single cell fit produces all-six and held-out summaries
    Given one ATF cell with six ordered sweeps
    When Step 04 fits one shared parameter vector under the expected ODE model
    Then candidate-level metrics are written
    And held-out sweep prediction metrics are written
    And accepted candidates are identified using Step 02 thresholds
```

```gherkin
@step04 @R6 @heldout-screen
Feature: held-out current screening is part of the acceptance contract
  Scenario: reviewer-facing cells must predict held-out sweeps
    Given a fitted cell candidate
    When each of the six sweeps is held out in turn
    Then a held-out prediction row is written for each held-out sweep
    And reviewer-facing status requires a minimum hold-out pass count
```
