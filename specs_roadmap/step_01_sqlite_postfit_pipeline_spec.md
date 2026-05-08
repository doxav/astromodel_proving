# Step 01 specification — SQLite post-fit pipeline and hidden-mechanism simulation

**Step identifier:** `step_01_sqlite_postfit_pipeline`  
**Pareto rank:** `1`  
**Primary reviewer role:** convert historical single-current fits from opaque parameter vectors into auditable effective-parameter and hidden-mechanism diagnostics  
**Notebook contract:** `analysis/01_postfit_sqlite_pipeline.ipynb`  
**Primary output directory:** `outputs/postfit_sqlite/`

---

## 1. Why this step exists

The historical Optuna DBs are still scientifically useful, but only if they stop being treated as black-box parameter dumps. Step 01 extracts best/top-N trials directly from SQLite, normalizes parameter dictionaries, converts structurally confounded raw parameters into effective coordinates, and computes hidden-current/flux summaries from representative best fits.

This step is still **partial** relative to the reviewer objections. The historical DBs are single-current fits and therefore do **not** support final reviewer-facing claims about degeneracy, predictive robustness, or region-level biology. Their role is:

- to prove exact structural confoundings like `P_gap_eff = d × pk`,
- to provide reusable hidden-current readouts,
- to create a bridge toward later accepted-ensemble, profile-likelihood, FIM, and six-sweep cell-specific analyses.

---

## 2. Reviewer objections addressed

| Reviewer ID | Coverage | How step 01 responds | How to verify |
|---|---|---|---|
| R1 | Partial | Demonstrates exact structural confounding of `d` and `pk`, and starts reporting effective parameters rather than raw ones. | Invariance test + notebook demonstration. |
| R4 | Partial | Replaces overinterpretation of raw best-fit values with effective-parameter summaries and normalized SQLite readers. | Effective-parameter CSV + tests. |
| R5 | Partial | Computes hidden Kir/gap/leak/K_o metrics for representative best trials. | Representative mechanism summary and plots. |
| R7 | Partial | Produces clear tables and mechanism plots from the historical DBs. | Executed notebook with visible tables/figures. |
| R2 | Indirect/partial | Maintains direct linkage back to DB provenance and prepares auditable post-fit tables. | Reuse of step 00 outputs where relevant. |
| R3, R6 | Not solved here | Historical single-current DBs remain provisional and are not sufficient for final robustness or assumption-sensitivity claims. | Notebook states limitations explicitly. |

---

## 3. Scientific objectives

### SO-01.1 — Structural confounding is demonstrated exactly, not rhetorically

The pipeline must show that changing `d` and `pk` while keeping their product fixed leaves the relevant RHS terms and `I_kgap` invariant in the reduced model.

**Scientific value:** prevents the paper from interpreting `d` and `pk` independently when Vm cannot separate them.

**Expected result:** an invariance demonstration where equal `d × pk` yields identical `P_gap_eff`, identical `I_kgap`, and identical derivatives at the same state/time.

**Verification:** acceptance tests + notebook cell.

### SO-01.2 — Historical best trials are readable without Optuna

The project must be able to recover the best complete finite trial from a SQLite DB and decode its parameter dictionary without installing Optuna.

**Scientific value:** removes hidden tooling dependence from reviewer-response analyses.

**Expected result:** direct SQLite reader returns trial ID, trial number, objective, study metadata, and normalized parameter dictionary for representative studies.

**Verification:** bootstrap and acceptance tests.

### SO-01.3 — Effective parameters become the primary reporting coordinates

The step must export at least the following effective combinations:

- `P_gap_eff = d × pk`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`

**Scientific value:** aligns the post-fit analysis with the updated specification and prepares later identifiability work.

**Expected result:** both top-trial and best-trial outputs include these columns.

**Verification:** acceptance tests + CSV schema checks.

### SO-01.4 — Hidden-current summaries are available for representative best trials

Representative best trials from Control, MFA, and Barium studies must be simulated with hidden outputs so that reviewers can inspect mechanism-level readouts instead of raw parameters alone.

**Scientific value:** turns historical fits into mechanistic diagnostics rather than parameter lists.

**Expected result:** representative simulations yield finite Vm plus `I_Kir`, `I_kgap`, `I_leak`, `K_o`, flux ratios, dominant mechanism, and proxy-validity metrics.

**Verification:** acceptance tests + notebook plots/tables.

---

## 4. Technical objectives

### TO-01.1 — Direct SQLite trial reader

Implement or reuse a direct SQLite reader that can:

- locate the best complete finite trial,
- optionally load top-N complete finite trials,
- decode categorical distributions,
- merge `fixed_params` when present,
- normalize parameter aliases and defaults.

### TO-01.2 — Effective-parameter helpers

Implement a stable function that converts a flat parameter dictionary into reviewer-facing effective-parameter columns.

### TO-01.3 — Structural invariance helper

Implement a function that takes a base flat parameter set and returns an explicit `d/pk` invariance diagnostic suitable for tests and notebook use.

### TO-01.4 — Representative simulation helper

Implement a representative trial simulation helper that:

- loads a best trial from SQLite,
- simulates it with `simulate_with_hidden_outputs`,
- computes flux summaries,
- computes proxy validity,
- returns a flat mechanism-summary row plus the simulation arrays needed for plotting.

### TO-01.5 — Pipeline entry point

Implement a single callable pipeline:

```python
run_step01_postfit_sqlite(project_root, top_n=5, representative_dbs=None, output_dir=None)
```

that writes machine-readable step-01 outputs under `outputs/postfit_sqlite/`.

---

## 5. Inputs

### Required repository inputs

- `data/1_Initial_xp_fit/*.db`
- step 00 provenance context is optional but recommended when interpreting outputs

### Required source-code dependencies

- `src.astro_model`
- `src.mechanisms`
- `src.optuna_sqlite`

---

## 6. Outputs

| Output file | Required | Description |
|---|---|---|
| `outputs/postfit_sqlite/top_trials_all_dbs.csv` | Yes | Top-N complete finite trials from all DBs with normalized/effective parameters. |
| `outputs/postfit_sqlite/effective_parameter_summary.csv` | Yes | Best-trial effective-parameter summary across all DBs. |
| `outputs/postfit_sqlite/representative_mechanism_summary.csv` | Yes | Representative best-trial mechanism/proxy summary for selected DBs. |

Optional additional exports are allowed, but these three are required.

---

## 7. Explicit non-goals

This step must **not**:

- claim final biological degeneracy;
- treat historical single-current best trials as equivalent to the future cell-specific six-sweep fits;
- claim fair gating-family model comparison from mixed historical DB settings;
- interpret raw best-fit parameters as direct molecular estimates when they are structurally or practically confounded.

---

## 8. Proposed approaches and comparison rules

| Approach | Purpose | Selection rule |
|---|---|---|
| Direct SQLite reader | Mandatory | Preferred over Optuna runtime. |
| Best-trial summaries | Initial diagnostic | Useful for notebook and smoke validation, but not enough for final claims. |
| Top-N trial summaries | Main machine-readable export | Use for triage, parameter-range inspection, and later accepted-fit development. |
| Effective-parameter reporting | Mandatory | Use as the primary interpretation space. |
| Hidden-current summaries | Mandatory for representative trials | Mechanistic interpretation must not rely on raw parameters alone. |

---

## 9. Decision logic

### Effective-parameter priority

When a raw parameter participates in a structurally confounded combination, the effective combination must be the primary reporting coordinate.

For step 01, the minimum required rule is:

- report `P_gap_eff`, not `d` and `pk` as independent interpretable quantities.

### Representative-trial status

Representative best trials are for **diagnostic illustration only** at this step.

They may be used to:

- illustrate `I_Kir`/`I_kgap`/`I_leak`/`K_o` behavior,
- show that different conditions occupy different provisional hidden-current regimes,
- validate that the refactored hidden-output machinery works.

They may **not** be used to:

- claim stable mechanism clusters,
- claim predictive robustness,
- claim region-enriched phenotypes,
- claim final accepted-ensemble degeneracy.

---

## 10. Verification strategy

### 10.1 Bootstrap tests

Bootstrap tests only prove the new step can start and the core API is present.

They must verify:

- step 01 modules import successfully;
- representative DBs exist;
- the direct SQLite reader returns a normalized best-trial parameter dictionary;
- the `d/pk` invariance helper produces matching `P_gap_eff`.

### 10.2 Acceptance tests

Acceptance tests verify the scientific contract of the implemented step.

They must verify:

- top-N export contains rows from all 18 DBs;
- effective-parameter summary contains all required columns;
- invariance diagnostics show identical derivatives and `I_kgap` when `d × pk` is fixed;
- representative simulations produce finite Vm and hidden-current outputs;
- mechanism summaries include flux/proxy columns.

### 10.3 Integration tests

Integration tests verify the full notebook execution.

They must verify:

- `analysis/01_postfit_sqlite_pipeline.ipynb` executes top-to-bottom;
- the required CSV outputs exist;
- the executed notebook contains tables and figures.

---

## 11. Test-first development order

1. Write bootstrap tests for module imports and direct best-trial loading.
2. Write acceptance tests for effective parameters, top-N exports, and representative mechanism summaries.
3. Write the notebook integration test.
4. Implement `src.optuna_sqlite` reader extensions if needed.
5. Implement `src.postfit_sqlite` helpers and pipeline.
6. Run the acceptance tests.
7. Build the notebook.
8. Execute the notebook and confirm output tables/plots.

---

## 12. Required Gherkin specifications

```gherkin
@step01 @bootstrap @R4 @postfit-sqlite
Feature: historical best trials are readable without Optuna
  Scenario: representative SQLite studies yield normalized best-trial parameter dictionaries
    Given representative historical DBs for CONTROL, MFA, and BARIUM
    When the direct SQLite reader loads the best complete finite trial
    Then it returns the stored objective
    And it returns the study metadata
    And it returns a normalized parameter dictionary with categorical choices decoded
```

```gherkin
@step01 @acceptance @R1 @effective-parameters
Feature: structurally confounded raw parameters are represented as effective parameters
  Scenario: d and pk are not interpreted independently in the reduced model
    Given two parameter sets with different d and pk but the same product d × pk
    When the RHS and hidden currents are evaluated at the same state and time
    Then the derivatives are numerically identical
    And I_kgap is numerically identical
    And P_gap_eff is the primary reported parameter
```

```gherkin
@step01 @acceptance @R4 @effective-summary
Feature: effective parameters are exported for top and best trials
  Scenario: the post-fit pipeline writes normalized effective-parameter outputs
    Given the historical SQLite studies
    When the step 01 pipeline runs
    Then top_trials_all_dbs.csv contains rows from all 18 DBs
    And effective_parameter_summary.csv contains P_gap_eff, gamma_t_eff, gamma_s_eff, and volume_ratio_wa_wo
```

```gherkin
@step01 @acceptance @R5 @mechanism-readout
Feature: representative best trials produce hidden-current and proxy summaries
  Scenario: Control, MFA, and Barium representatives are simulated with hidden outputs
    Given representative best trials from CONTROL_75nA.db, MFA_100nA.db, and BARIUM_100nA.db
    When the representative post-fit mechanism summary is computed
    Then each summary row contains I_Kir, I_kgap, I_leak, K_o, gap/Kir ratio, dominant mechanism, and proxy validity fields
    And each simulation returns finite Vm values
```

```gherkin
@step01 @integration @R7 @notebook
Feature: post-fit notebook validates the implemented SQLite pipeline
  Scenario: the notebook executes and writes machine-readable outputs
    Given the implemented post-fit SQLite pipeline
    When analysis/01_postfit_sqlite_pipeline.ipynb is executed from the repository root
    Then the notebook completes without error
    And outputs/postfit_sqlite/top_trials_all_dbs.csv exists
    And outputs/postfit_sqlite/effective_parameter_summary.csv exists
    And outputs/postfit_sqlite/representative_mechanism_summary.csv exists
```

---

## 13. Notebook contract

The notebook must let a reader verify, visually and numerically, that step 01 is implemented and functioning.

### Mandatory sections

1. Title and scope/limitations of the historical DB analysis.
2. Project-root discovery and pipeline execution.
3. Direct SQLite best-trial table.
4. Top-N overview table.
5. Effective-parameter summary table.
6. Exact `d/pk` invariance demonstration.
7. Representative Vm traces.
8. Representative hidden-current overlays.
9. Mechanism/proxy summary table.
10. Short markdown interpretation stating what is shown and what remains provisional.

### Mandatory notebook outputs

- visible tables;
- at least two figures;
- required CSV outputs under `outputs/postfit_sqlite/`.

---

## 14. Done criteria

The step is complete only when all of the following are true:

- bootstrap, acceptance, and integration tests pass;
- the notebook executes without manual edits;
- required output CSVs exist and contain the required columns;
- the `d/pk` invariance demonstration is exact up to numerical tolerance;
- representative mechanism summaries are finite and readable;
- notebook markdown explicitly states that historical single-current DBs are provisional and do not replace later six-sweep cell-specific inference.

---

## 15. Risks and interpretation boundaries

| Risk | Mitigation |
|---|---|
| Treating historical single-current best trials as final evidence | The notebook and tests must label them as diagnostic only. |
| Hidden-output machinery diverges from the historical model | Use the refactored `src.astro_model` functions and assert finite representative simulations. |
| Raw-parameter overinterpretation | Export effective parameters and make invariance tests mandatory. |
| Misreading mixed historical gating choices as a fair model comparison | Report decoded historical switching function, but explicitly defer fair family comparison to the later assumption-sensitivity step. |

---

## 16. Files expected to be created in this step

- `src/postfit_sqlite.py`
- `analysis/01_postfit_sqlite_pipeline.ipynb`
- `tests/bootstrap/test_step01_bootstrap.py`
- `tests/acceptance/test_step01_acceptance.py`
- `tests/integration/test_step01_integration.py`

