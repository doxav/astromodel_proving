# Step 00 specification — Data provenance and objective reproducibility audit

**Step identifier:** `step_00_data_provenance_audit`  
**Pareto rank:** `0`  
**Primary reviewer role:** prevent invalid inference before any scientific reinterpretation  
**Notebook contract:** `analysis/00_data_provenance_audit.ipynb`  
**Primary output directory:** `outputs/provenance/`

---

## 1. Why this step exists

The project cannot defend or reinterpret historical optimization results until the pipeline can prove, in a machine-readable way, that the underlying SQLite studies, trace sources, and ATF cell inventory are discoverable and auditable without hidden Colab state.

This step is the required gate before any later claims about degeneracy, effective parameters, predictive validity, or mechanism-space clustering.

It is intentionally **partial** with respect to reviewer objections: it does not improve the model and it does not prove biological degeneracy. It ensures that later conclusions are not built on silent provenance errors.

---

## 2. Reviewer objections addressed

| Reviewer ID | Coverage | How step 00 responds | How to verify |
|---|---|---|---|
| R2 | Partial | Makes the data sources, historical trace sources, and ATF cell inventory explicit and machine-readable. | Inventory tables, region-condition counts, provenance status table. |
| R4 | Partial | Prevents interpretation of historical fit parameters until the objective source trace is verified or explicitly unresolved. | Objective recomputation table with `verified`/`unresolved`/`missing_source`. |
| R7 | Partial | Produces clear provenance tables and basic audit plots suitable for a methods supplement. | Executed notebook with labeled tables/plots. |
| R1 | Indirect/partial | Avoids premature interpretation of historical raw parameters by forcing provenance verification first. | Step 00 notebook must explicitly state that no degeneracy claim is made here. |
| R3, R5, R6 | None directly | These are downstream. Step 00 only prepares trustworthy inputs. | Notebook explicitly states non-goals. |

---

## 3. Scientific objectives

### SO-00.1 — Historical fitting inputs are locally discoverable and readable

The repository must discover the full initial-fit dataset without Colab, Google Drive, or Optuna runtime dependencies.

**Scientific value:** ensures the old fitting results can be audited reproducibly.

**Expected result:** exactly 18 SQLite DBs, all historical trace CSVs, all best-fit CSVs, and the legacy threshold CSV are found.

**Verification:** bootstrap tests + notebook inventory cells.

### SO-00.2 — Control provenance ambiguity is explicit, never silent

Because both `CONTROL_TRACES.csv` and `CONTROL_TRACES_old.csv` are present, the pipeline must never silently assume one source. It must score candidate sources and label unresolved cases explicitly.

**Scientific value:** prevents invalid use of Control historical fits in later interpretation.

**Expected result:** each Control DB has a provenance status with the best candidate source and relative objective error; if no candidate matches tolerance, status remains `unresolved`.

**Verification:** acceptance tests + notebook provenance table.

### SO-00.3 — Historical objective reproducibility is quantified per condition/current

The best trial of each historical study must be re-simulated and re-scored against the candidate trace source using the documented legacy preprocessing contract.

**Scientific value:** prevents treating historical DB objectives as trustworthy black boxes.

**Expected result:** BARIUM studies and most MFA studies should be reproducible within tolerance; Control should remain unresolved unless a true match is demonstrated.

**Verification:** acceptance tests on provenance status distribution.

### SO-00.4 — DH/VH region provenance is audited from the ATF dataset

The 37 ATF files must be parsed into `file_id`, `region`, and `condition`, with a count table against the expected region-condition design.

**Scientific value:** preserves the updated specification’s region-aware design contract.

**Expected result:** 37 ATF files, 19 DH cells, 18 VH cells, and the exact expected `region × condition` counts unless the dataset has changed.

**Verification:** acceptance tests + notebook region-count table.

---

## 4. Technical objectives

### TO-00.1 — Direct SQLite study summaries without Optuna

Implement a plain-SQLite reader that can:

- list tables,
- read `studies.study_name`,
- count trials,
- locate the best complete finite trial,
- parse study names into condition/current/loss/target-point metadata.

### TO-00.2 — Trace-source discovery and metadata summary

Implement a trace-source inventory that records:

- filename,
- condition,
- row count,
- column count,
- time start/end,
- median `dt`,
- whether the file uses a header row,
- file size.

### TO-00.3 — Legacy trace preprocessing contract

Implement the historical preprocessing required to recompute stored DB objectives:

- load the CSV source,
- select the current-specific trace column,
- apply historical `cut_after_index` and `cut_before_ratio`,
- reproduce the original “skip first row” behavior,
- normalize the trace using the target mean mode,
- apply the documented MFA 100 nA outlier cleanup.

### TO-00.4 — Historical objective recomputation contract

Implement a pure-Python objective recomputation for the best trial of each DB:

- simulate the historical ODE on a dense time grid,
- trim the stable simulation window,
- downsample the simulated Vm by median binning,
- normalize using the historical target mode,
- compute the historical loss type (`L2`, `COMBINED`, etc.),
- report stored objective, recomputed objective, relative objective error, and status.

### TO-00.5 — ATF provenance parsing

Implement ATF filename parsing that:

- infers `region` as `DH` or `VH`,
- infers `condition` as `CONTROL`, `MFA`, or `MFA_BA`,
- fails explicitly on unknown or ambiguous labels,
- writes both an inventory table and a region-condition count table.

### TO-00.6 — Pipeline entry point

Implement a single callable pipeline:

```python
run_step00_provenance(project_root, relative_tolerance=1e-2, output_dir=None)
```

that writes all outputs under `outputs/provenance/`.

---

## 5. Inputs

### Required repository inputs

- `data/1_Initial_xp_fit/*.db`
- `data/1_Initial_xp_fit/*_TRACES*.csv`
- `data/1_Initial_xp_fit/*_BEST_FIT_PARAM.csv`
- `data/threshold_for_good_enough_fits.csv`
- `data/2_K+ Pumps Data/*.atf`

### Required source-code dependencies

- `src.astro_model`
- `src.optuna_sqlite`
- `pandas`, `numpy`, `sqlite3`

---

## 6. Outputs

The pipeline must write at least the following files.

| Output file | Required | Description |
|---|---|---|
| `outputs/provenance/db_study_summary.csv` | Yes | One row per historical DB. |
| `outputs/provenance/trace_source_summary.csv` | Yes | One row per trace-source CSV. |
| `outputs/provenance/control_trace_verification.csv` | Yes | One row per `db × candidate_trace_source`, including chosen source and status. |
| `outputs/provenance/atf_region_condition_inventory.csv` | Yes | One row per ATF file. |
| `outputs/provenance/atf_region_condition_counts.csv` | Yes | Expected vs observed counts by `region × condition`. |

Optional extra outputs are allowed if they do not replace the required files.

---

## 7. Explicit non-goals

This step must **not**:

- claim or imply biological degeneracy;
- interpret raw historical parameters as physiological;
- use pooled ATF cell counts to erase DH/VH structure;
- rewrite the historical model;
- treat unresolved Control provenance as “close enough”.

---

## 8. Proposed approaches and comparison rules

| Approach | Purpose | Selection rule |
|---|---|---|
| Direct SQLite audit | Mandatory baseline | Must work without Optuna installed. |
| Objective recomputation against `*_TRACES.csv` | Primary candidate source check | Accept only when the relative error is below tolerance and the preprocessing is documented. |
| Objective recomputation against `CONTROL_TRACES_old.csv` | Mandatory Control comparison | Prefer over `CONTROL_TRACES.csv` only if it reduces the error materially, but still require tolerance for verification. |
| ATF filename parsing | Required for region contract | Reject silent pooling or unknown region labels. |
| ATF metadata fallback | Optional future extension | Only needed if filename parsing becomes insufficient. |

---

## 9. Decision logic

### Provenance status classes

| Status | Meaning |
|---|---|
| `verified` | Recomputed best-trial objective matches the stored DB objective within tolerance for the candidate source. |
| `unresolved` | Candidate sources exist, but none match within tolerance. |
| `missing_source` | The expected trace source does not exist. |

### Region provenance rules

- `DH` and `VH` are the only valid region labels.
- `CONTROL`, `MFA`, and `MFA_BA` are the only valid conditions in the ATF inventory.
- Unknown or ambiguous filenames are errors, not warnings.

---

## 10. Verification strategy

### 10.1 Bootstrap tests

Bootstrap tests only prove that the repository can start the step.

They must verify:

- required directories/files exist;
- the SQLite schema is readable;
- DB names and study names parse correctly;
- ATF filenames parse into region and condition.

### 10.2 Acceptance tests

Acceptance tests verify the scientific contract of the implemented step.

They must verify:

- all 18 DBs appear in the DB summary;
- all 37 ATFs appear in the ATF inventory;
- the expected DH/VH counts are reproduced;
- Control provenance remains unresolved unless a true objective match is found;
- BARIUM provenance is verified across all 6 currents;
- MFA provenance is verified where the legacy pipeline matches and unresolved where it does not.

### 10.3 Integration tests

Integration tests verify that the full validation notebook executes and writes outputs.

They must verify:

- `analysis/00_data_provenance_audit.ipynb` executes top-to-bottom;
- the expected CSV outputs are created;
- the executed notebook contains visible outputs.

---

## 11. Test-first development order

1. Write bootstrap tests for path discovery and parsing.
2. Write acceptance tests for full provenance audit and ATF count checks.
3. Write the notebook integration test.
4. Implement `src.optuna_sqlite` helpers if missing.
5. Implement `src.provenance` with objective recomputation and ATF parsing.
6. Run acceptance tests.
7. Build the notebook.
8. Execute the notebook and confirm output tables/plots.

No production code should be marked complete before all three test levels pass.

---

## 12. Required Gherkin specifications

```gherkin
@step00 @bootstrap @R2 @R4 @R7 @provenance
Feature: Historical fitting-data provenance audit bootstrap
  Scenario: required local inputs are discoverable without Colab or Optuna
    Given the repository root contains data/1_Initial_xp_fit and data/2_K+ Pumps Data
    When the bootstrap audit scans the repository
    Then it finds 18 SQLite DB files
    And it finds the historical trace CSVs
    And it finds the ATF directory
    And it can parse representative DB and ATF filenames
```

```gherkin
@step00 @acceptance @R2 @R4 @objective-recompute
Feature: Historical objective reproducibility
  Scenario: best-trial objectives are recomputed against candidate trace sources
    Given a historical Optuna SQLite DB and a candidate trace CSV
    When the step 00 provenance pipeline recomputes the best-trial objective
    Then the stored objective is reported
    And the recomputed objective is reported
    And the relative objective error is reported
    And status is verified only when the error is below tolerance
```

```gherkin
@step00 @acceptance @R2 @control-provenance
Feature: Control trace ambiguity is explicit
  Scenario: Control provenance is unresolved when no candidate source verifies
    Given CONTROL_TRACES.csv and CONTROL_TRACES_old.csv are both present
    When the provenance audit scores both candidate sources for each Control DB
    Then the chosen Control source is recorded
    And the status is unresolved when the relative error remains above tolerance
    And no downstream output marks Control as validated silently
```

```gherkin
@step00 @acceptance @R2 @region-provenance
Feature: DH and VH provenance is explicit at the ATF level
  Scenario: each ATF file contributes one region and one condition
    Given 37 ATF files under data/2_K+ Pumps Data
    When the provenance audit parses ATF filenames
    Then every file has region DH or VH
    And every file has condition CONTROL, MFA, or MFA_BA
    And the region-condition count table matches the expected design or reports a mismatch explicitly
```

```gherkin
@step00 @integration @R7 @notebook
Feature: provenance notebook validates the implemented pipeline
  Scenario: the notebook executes and writes machine-readable outputs
    Given the implemented provenance pipeline
    When analysis/00_data_provenance_audit.ipynb is executed from the repository root
    Then the notebook completes without error
    And outputs/provenance/db_study_summary.csv exists
    And outputs/provenance/control_trace_verification.csv exists
    And outputs/provenance/atf_region_condition_counts.csv exists
```

---

## 13. Notebook contract

The notebook must be structured so that a reviewer or developer can visually confirm the step contract.

### Mandatory sections

1. Title and step purpose.
2. Project-root/path discovery.
3. DB study summary table.
4. Trace source summary table.
5. Objective reproducibility table with candidate source comparison.
6. Control provenance focus table.
7. ATF region-condition inventory/count table.
8. At least one diagnostic plot of best objectives.
9. At least one diagnostic plot of trial counts or provenance status.
10. Short markdown interpretation stating what is resolved and unresolved.

### Mandatory notebook outputs

- visible tabular outputs;
- at least two figures;
- machine-readable CSVs under `outputs/provenance/`.

---

## 14. Done criteria

The step is complete only when all of the following are true:

- bootstrap, acceptance, and integration tests pass;
- the notebook executes from the repository root without manual edits;
- all required CSVs are produced;
- the ATF inventory reports the region-aware design correctly;
- Control provenance is explicit and not silently marked verified;
- notebook markdown clearly states that this step is partial and does not make a degeneracy claim.

---

## 15. Risks and interpretation boundaries

| Risk | Mitigation |
|---|---|
| Historical objective mismatch due to undocumented preprocessing | Keep status as `unresolved`; never coerce to verified. |
| Future dataset changes modify ATF counts | The count table stores both observed and expected values and can surface mismatches immediately. |
| Temptation to pool DH and VH too early | Step 00 writes region-aware outputs and the notebook must state that region-blind pooling is not primary. |
| Confusing historical DB trust with scientific validity | The notebook must separate “machine-readable and reproducible” from “scientifically sufficient for reviewer claims.” |

---

## 16. Files expected to be created in this step

- `src/provenance.py`
- `src/optuna_sqlite.py` (if not already present)
- `analysis/00_data_provenance_audit.ipynb`
- `tests/bootstrap/test_step00_bootstrap.py`
- `tests/acceptance/test_step00_acceptance.py`
- `tests/integration/test_step00_integration.py`

