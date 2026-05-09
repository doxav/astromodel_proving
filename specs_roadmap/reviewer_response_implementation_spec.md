# Reviewer-response implementation specification

This specification defines a test-first, notebook-validated implementation plan for the astrocytic potassium-buffering model revision. It is ordered by Pareto priority: each step should increase reviewer-facing credibility before lower-impact refinements are attempted.

The target scientific reframing is:

> Multiple Vm-compatible parameter sets are not, by themselves, evidence of biological degeneracy. The revised pipeline must first remove obvious structural non-separabilities, quantify practical identifiability and sloppiness, and then reserve the term degeneracy for accepted ensembles that are mechanistically distinct, physiologically interpretable, and predictive under held-out currents or perturbations.

## Reviewer critique taxonomy used by the plan

| ID | Reviewer objection | Required computational response |
|---|---|---|
| R1 | Degeneracy is not distinguished from structural non-identifiability, practical non-identifiability, sloppiness, or parameter compensation. | Operational definitions, effective parameters, FIM/profile-likelihood diagnostics, compensation-vs-mode geometry, and mechanism mapping. |
| R2 | Experimental variability, noise, uncertainty, and data constraints are insufficiently described. | Data provenance audit, ATF feature reliability, condition-specific thresholds, uncertainty-aware losses. |
| R3 | Model assumptions are insufficiently justified: sigmoid gating, intracellular K as ECS proxy, and local/syncytial split. | Alternative model families, proxy validity quantification, compartment-split sensitivity. |
| R4 | Vm-only fitting weakly constrains ionic dynamics and yields potentially non-physiological parameters. | Effective parameter reporting, parameter-range audits, constrained inference, hidden-current diagnostics. |
| R5 | Different parameter regimes are not shown to represent different astrocytic pathways or phenotypes. | Kir/gap/leak/K_o flux decomposition, mechanism-space clustering, representative selection. |
| R6 | Robustness beyond fitted traces and perturbations is not systematically tested. | Leave-one-current-out validation, prediction intervals, accepted-ensemble perturbation sweeps. |
| R7 | Figures/equations/units/layouts are unclear and cluttered. | Reproducible notebooks with cleaner panels, tables, unit checks, and exported manuscript-ready figures. |

## Implementation principles

1. **Tests precede code.** Each step starts by adding unit, functional, and integration tests. New code is considered complete only when the step tests pass.
2. **Each step has a validation notebook.** The notebook must load real data, generate at least one table and one visual diagnostic, and write machine-readable outputs under `outputs/<step>/`.
3. **No Colab or Google Drive dependence.** Every notebook and test must run from the repository root with local paths.
4. **No silent provenance assumptions.** Any ambiguous trace source, threshold source, objective mismatch, or candidate provenance mismatch must be represented as an explicit status field.
5. **Mechanisms are not inferred from raw parameters alone.** Mechanistic claims require hidden-current or flux summaries.
6. **Claims are graded as full or partial.** A step may partially answer a reviewer objection, but the notebook must state what remains unresolved.
7. **Brain region is a first-class biological factor.** The 37 ATF files include dorsal hippocampus (`DH`) and ventral hippocampus (`VH`) cells. Region must be parsed, audited, retained in every table, used in thresholds/model evaluation, and shown in reviewer-facing summaries. Region-blind pooling is allowed only as an explicitly labeled sensitivity or shrinkage fallback, not as the primary analysis.
8. **Legacy single-current DBs are provisional assets.** They are useful for provenance, debugging, structural confounding checks, and mechanism tooling. They are not sufficient for final claims about cell-level mechanisms, DH/VH enrichment, or biological degeneracy.
9. **Cell-specific six-sweep ensembles are the primary reviewer-facing inference target.** Mechanism claims should be based on accepted parameter sets that jointly explain the six ordered sweeps of a cell and support held-out-current prediction or perturbation checks.

## Coverage audit against the current recommendation

This table is part of the specification. It records whether the development plan fully covers each recommendation and what must be refined before the item can be treated as reviewer-facing.

| Recommendation from analysis | Current coverage | Required refinement in this spec |
|---|---|---|
| Fit one cell across all 6 sweeps jointly, with one shared cell parameter set. | Promoted to the main Step 04 objective. | Historical single-current DB transfer remains debug/triage only. Reviewer-facing accepted ensembles are cell-specific and six-sweep aware. |
| Treat DH/VH brain region as a biological factor. | Covered by Step 02 and carried into Steps 04, 05, 06, and 09. | Preserve region in every cell-level output; stratify thresholds and predictive checks by region; use region-blind pooling only as a labeled sensitivity or shrinkage fallback. |
| Reparameterize raw parameters into effective combinations before interpretation. | Covered by Step 01 and Step 03. | Keep `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, and `volume_ratio_wa_wo` as primary reporting coordinates. Add profile interpretation rules: clear valley, flat profile, boundary hit, broad valley. |
| Merge structural and practical identifiability using a soft STRIKE-GOLDD-inspired workflow. | Covered by Step 03 as a structural-inspection + practical-profile workflow. | Avoid claiming full STRIKE-GOLDD unless a formal symbolic tool is implemented. |
| Use FIM/sloppiness diagnostics. | Covered by Step 03. | Keep FIM after effective-parameter reparameterization and run on verified representative centers before Step 04; rerun or extend on accepted cell-specific centers once Step 04 exists. |
| Analyze accepted-fit geometry as continuous compensation manifold vs separated modes. | Covered by Step 05 after cell-specific accepted ensembles exist. | Add bootstrap cluster stability and interpolation tests between candidate modes. Interpret continuous connected sets as compensation, not degeneracy. |
| Mechanistic decomposition of accepted regimes. | Covered by Step 05. | Use Step 04 cell-specific accepted ensembles for primary claims. Legacy DB-derived mechanisms remain provisional tooling/debug output. |
| Assumption sensitivity for gating form. | Covered by Step 07. | Include sigmoid, tanh, Hill, soft-threshold, hard-threshold, and double-sigmoid variants; compare all with identical data splits, loss definitions, and evaluation metrics. |
| Proxy and compartment-split sensitivity. | Covered by Step 07. | Keep explicit ECS variant optional unless proxy validity fails; keep one-state intracellular variant as sensitivity. |
| Parameter plausibility and constrained reruns. | Covered by Step 08. | Distinguish `within_range`, `identifiable`, and `physiologically_interpretable`; a parameter inside bounds may still be weakly identified. |
| Population-level posterior predictive checks. | Covered by Step 06 and Step 09. | Add explicit feature-distribution posterior predictive checks for `region × condition × sweep` groups using accepted ensembles. |
| Figures and reviewer-facing outputs. | Covered by Step 09. | Add a traceability table mapping each figure/table to reviewer critique IDs and source outputs. |

## Region-aware experimental-design contract

The new ATF dataset must be treated as an unpaired, two-region, repeated-sweep design. Each ATF file is one independent cell; each cell has six ordered pump-current sweeps; each cell belongs to one condition and one brain region. Region labels must be interpreted as:

- `DH`: dorsal hippocampus
- `VH`: ventral hippocampus

The expected current dataset counts are:

| Region | control | MFA | MFA_BA | Total cells |
|---|---:|---:|---:|---:|
| DH | 7 | 6 | 6 | 19 |
| VH | 4 | 7 | 7 | 18 |
| Total | 11 | 13 | 13 | 37 |

Consequences for the reviewer-response analyses:

1. Every experimental feature table, threshold table, fitting output, accepted-ensemble table, posterior predictive table, and figure source table must include `file_id`, `region`, `condition`, and `sweep`.
2. Primary feature thresholds must be region-aware: `region × condition × sweep × feature`.
3. Primary posterior predictive checks must be region-aware: `region × condition × sweep`.
4. Primary fit-error and held-out prediction summaries must report DH and VH separately before any pooled result.
5. Region-blind pooling is allowed only as a sensitivity analysis or a shrinkage fallback when a stratum is too small. If pooling is used, the output must include `threshold_scope = region_specific`, `region_pooled`, or `global_pooled`.
6. Because cells are unpaired and animal/slice IDs are unavailable, the manuscript must not claim paired pharmacology effects or animal-level effects. Condition and region effects are population-level cell effects.
7. Statistical summaries should use the cell/file as the independent unit and treat sweep as repeated within cell. Recommended descriptive/statistical model: `feature ~ region * condition * sweep`, with file-level grouping/random intercept where feasible; otherwise use robust bootstrap confidence intervals at the cell level.
8. Small strata, especially VH-control, must be flagged. Use shrinkage or pooled sensitivity checks rather than silently treating uncertain region-specific thresholds as precise.

---

# Pareto implementation roadmap

## Step 00 — Data provenance and objective reproducibility audit

**Pareto rank:** 0. Mandatory before scientific reinterpretation.

**Primary output:** `outputs/provenance/control_trace_verification.csv`, `outputs/provenance/db_study_summary.csv`, `outputs/provenance/trace_source_summary.csv`, `outputs/provenance/atf_region_condition_inventory.csv`.

### Scientific objectives

1. **Resolve R2 partially:** prove that the historical Optuna DBs, trace files, and threshold CSV are discoverable and machine-readable.
2. **Resolve R4 partially:** prevent parameter-fit interpretation until the objective source trace is verified.
3. **Resolve R7 partially:** generate clean provenance tables and simple diagnostic plots for the methods supplement.
4. **Resolve R2 partially:** audit DH/VH region labels and condition labels for the 37 ATF cells before downstream thresholding/fitting.

This step is a partial response because it does not yet improve the model. It prevents false inference from mismatched sources.

### Technical objectives

- Discover exactly 18 SQLite DBs: 3 conditions × 6 currents.
- Read Optuna SQLite summaries without importing Optuna.
- Read all trace CSVs and detect sources.
- Parse ATF filenames/metadata to infer `region`, `condition`, and `file_id`; fail explicitly on unknown or ambiguous region labels.
- Write an ATF inventory with expected counts by `region × condition`, including DH-control, VH-control, DH-MFA, VH-MFA, DH-MFA_BA, and VH-MFA_BA.
- Recompute objective values for the best trial against each candidate trace source when the simulation and preprocessing contract is available.
- Write an explicit status per condition/current:
  - `verified`: recomputed objective matches DB objective within tolerance;
  - `unresolved`: objective cannot be matched yet;
  - `missing_source`: expected trace or DB is absent.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Direct SQLite audit | Always required | Must pass without Optuna. |
| Objective recomputation against `*_TRACES.csv` | First candidate | Accept only if relative error is below threshold and preprocessing is documented. |
| Objective recomputation from ATF-derived traces | Fallback | Use if neither historical CSV source matches the Control DB objective. |

### How to verify

- `pytest tests/test_00_data_provenance_audit.py` passes.
- The notebook `analysis/00_data_provenance_audit.ipynb` runs top-to-bottom.
- Output tables contain all 18 DBs and all trace sources.
- ATF inventory contains 37 files, both regions `DH` and `VH`, all three conditions, and no unknown-region files.

### Gherkin specifications

```gherkin
@step00 @R2 @R4 @R7 @provenance
Feature: Historical fitting-data provenance audit
  Scenario: all initial-fit inputs are discoverable without Colab or Optuna
    Given the repository data directory "data/1_Initial_xp_fit"
    When the provenance audit scans SQLite DBs, trace CSVs, best-fit CSVs, and the threshold example CSV (to be later recomputed on ATF data)
    Then it finds 18 Optuna DB files
    And it finds 6 currents for each of CONTROL, MFA, and BARIUM
    And it finds CONTROL_TRACES.csv, MFA_TRACES.csv, and BARIUM_TRACES.csv
    And it writes a machine-readable inventory table
```

```gherkin
@step00 @R4 @objective-recompute
Feature: Optuna objective reproducibility
  Scenario: a best trial objective is recomputed from the documented trace source
    Given an Optuna SQLite DB and a candidate experimental trace source
    When the best trial is simulated with the documented preprocessing pipeline
    Then the recomputed objective is reported
    And the relative objective error versus the stored DB value is reported
    And status is "verified" only when the relative error is below tolerance
```

```gherkin
@step00 @R2 @region-provenance
Feature: DH/VH ATF region provenance is explicit
  Scenario: each ATF file has one region and one condition
    Given 37 ATF files under "data/2_K+ Pumps Data"
    When the provenance audit parses ATF filenames and metadata
    Then every file has region DH or VH
    And every file has condition control, MFA, or MFA_BA
    And the inventory reports cell counts by region and condition
    And unknown or ambiguous region labels are reported as errors, not silently pooled
```

### Notebook required

`analysis/00_data_provenance_audit.ipynb`

The notebook must show:

- DB inventory table;
- trace source table;
- best objective by condition/current plot;
- trial count by condition/current plot;
- Control provenance status table;
- ATF `region × condition` cell-count table.

### Tests required before implementation

- Unit: DB filename parser; SQLite schema reader; trace-source summarizer.
- Functional: all 18 DBs are readable without Optuna.
- Integration: Control ambiguity is represented explicitly.

---

## Step 01 — SQLite post-fit pipeline and hidden-mechanism simulation

**Pareto rank:** 1. Converts old single-current DBs from black-box fits into reviewer-useful post-fit evidence.

**Primary output:** `outputs/postfit_sqlite/top_trials_all_dbs.csv`, `outputs/postfit_sqlite/representative_mechanism_summary.csv`, `outputs/postfit_sqlite/effective_parameter_summary.csv`.

### Scientific objectives

1. **Resolve R1 partially:** demonstrate exact structural confounding of `d` and `pk` through `P_gap_eff = d × pk`.
2. **Resolve R4 partially:** report identifiable/effective combinations instead of overinterpreting raw parameters.
3. **Resolve R5 partially:** compute hidden Kir, gap, leak, and K_o metrics for representative best trials.
4. **Resolve R7 partially:** produce clear tables and mechanism plots.

This remains partial because best-trial or top-N single-current analysis is not sufficient for final degeneracy claims. It is the bridge from old DBs to accepted-ensemble and multi-current analyses.

### Technical objectives

- Read best/top-N trials directly from Optuna SQLite.
- Decode categorical distributions and `fixed_params` system attributes.
- Normalize missing parameter defaults such as `switching_function`, `wo_middle`, and `eps_middle`.
- Simulate representative trials with `src.astro_model.simulate_with_hidden_outputs`.
- Compute `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`.
- Compute flux/proxy metrics with `src.mechanisms`.
- Produce per-condition summary tables.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Direct SQLite reader | Required | Preferred because it works without Optuna installation. |
| Optuna API reader | Optional | Allowed only as a convenience wrapper if it reproduces direct SQLite output. |
| Best-trial mechanism summary | Initial diagnostic | Useful for debugging, not enough for final claims. |
| Top-N accepted ensemble summary | Next target | Reviewer-facing once thresholds are condition-specific. |

### How to verify

- `pytest tests/test_01_postfit_sqlite_pipeline.py` passes.
- The notebook `analysis/01_postfit_sqlite_pipeline.ipynb` runs top-to-bottom.
- Structural confounding test shows equal `d × pk` gives equal RHS and equal `I_kgap`.
- Representative simulations return finite Vm, hidden currents, flux summaries, and proxy validity.

### Gherkin specifications

```gherkin
@step01 @R1 @effective-parameters
Feature: structurally confounded raw parameters are represented as effective parameters
  Scenario: d and pk cannot be interpreted independently in the reduced model
    Given two parameter sets with different d and pk
    And both parameter sets have the same product d × pk
    When the astrocyte RHS and hidden currents are evaluated at the same state and time
    Then the derivatives are numerically identical
    And I_kgap is numerically identical
    And P_gap_eff is reported as the interpretable parameter
```

```gherkin
@step01 @R4 @postfit-sqlite
Feature: best trials can be loaded without Optuna
  Scenario: SQLite DBs yield normalized parameter dictionaries
    Given a historical Optuna SQLite DB
    When the post-fit reader selects the best complete finite trial
    Then it returns the stored objective
    And it returns a parameter dictionary including numeric and categorical choices
    And missing defaults are filled explicitly
```

```gherkin
@step01 @R5 @mechanism-readout
Feature: hidden-current outputs support mechanistic interpretation
  Scenario: representative best trials produce flux and proxy summaries
    Given representative best trials from CONTROL, MFA, and BARIUM
    When each trial is simulated with hidden outputs
    Then the output contains I_Kir, I_kgap, I_leak, K_o, and effective parameters
    And the flux summary reports gap/Kir ratio and dominant mechanism
    And the proxy summary reports ΔK_a,t to K_o validity
```

### Notebook required

`analysis/01_postfit_sqlite_pipeline.ipynb`

The notebook must show:

- direct SQLite best-trial table;
- structural `d/pk` confounding demonstration;
- representative Vm traces;
- representative hidden-current overlays;
- mechanism/proxy summary table.

### Tests required before implementation

- Unit: categorical decoding; effective parameter invariance.
- Functional: direct best-trial loading from representative DBs.
- Integration: representative best trials simulate and produce mechanism/proxy summaries.

---

## Step 02 — Rebuild region-aware feature thresholds from the 37 ATF files

**Pareto rank:** 2. Converts experimental variability into the acceptance model used for fits.

**Primary output:** `outputs/features/feature_table_by_sweep.csv`, `outputs/features/condition_region_sweep_thresholds.csv`, `outputs/features/feature_reliability_weights.csv`, `outputs/features/region_condition_cell_counts.csv`, `outputs/features/region_effect_summary.csv`.

### Scientific objectives

1. **Resolve R2 strongly:** quantify experimental variability by cell, condition, region, and sweep.
2. **Resolve R4 partially:** prevent weak or missing features from overweighting the loss.
3. **Resolve R7 partially:** create clean feature-reliability figures for supplement.
4. **Resolve R2 strongly:** prevent DH/VH biological differences from being washed out by region-blind thresholds or pooled fits.

This step is stronger than the legacy `threshold_for_good_enough_fits(TO BE RECOMPUTED BASED ON ATF 2_K+ Pumpts Data).csv` because the legacy threshold lacks condition-specific pharmacological structure and does not protect against DH/VH pooling artifacts.

### Technical objectives

- Parse all 37 ATF files in `data/2_K+ Pumps Data`.
- Extract per-sweep features using the robust ATF notebook logic.
- Preserve `region` as `DH` or `VH` in every feature row; reject or quarantine files with unknown region.
- Produce thresholds by `condition × region × sweep × feature` as the primary threshold scope.
- Compute missingness and reliability weights separately by region and condition.
- Produce `region_condition_cell_counts.csv` and flag strata with small sample size, especially VH-control.
- Estimate region effects and `region × condition` interactions for key features using cell-level bootstrap or mixed/clustered models.
- Identify redundant features, especially `peak_depolarization_mV` versus `stim_end_depolarization_mV`.
- Export a threshold file that can be consumed by the post-fit and multi-current fitting steps.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Region-specific quantile/IQR thresholds | Primary robust option | Use for pass/fail accepted-fit screening by `region × condition × sweep`. |
| CI95 thresholds | Report as sensitivity | Use when sample size is adequate and distribution is stable. |
| Region-pooled thresholds | Sensitivity/shrinkage fallback | Use only when a condition × region stratum is too small; label output with `threshold_scope = region_pooled`. |
| Global pooled thresholds | Negative-control sensitivity | Use only to demonstrate whether region-blind pooling changes conclusions. Do not use as primary reviewer-facing threshold. |
| Feature reliability weights | Required for loss | Down-weight missing/unstable/redundant features by region and condition. |
| Region-effect summaries | Required for interpretation | Report DH vs VH differences and region × condition interactions for key features. |

### How to verify

- All 37 ATF files are parsed.
- Each file contributes 6 sweeps.
- Both regions are present: `DH` and `VH`.
- The region-condition count table reports 7 DH-control, 4 VH-control, 6 DH-MFA, 7 VH-MFA, 6 DH-MFA_BA, and 7 VH-MFA_BA cells, or explicitly explains any dataset change.
- Threshold table contains condition, region, sweep, feature, median, IQR, acceptable bounds, missing rate, and reliability weight.
- `stim_end` and `peak` redundancy is flagged.

### Gherkin specifications

```gherkin
@step02 @R2 @atf-thresholds
Feature: condition-specific ATF feature thresholds
  Scenario: all ATF files produce sweep-level features
    Given 37 ATF files under "data/2_K+ Pumps Data"
    When the ATF pipeline extracts features
    Then it returns 222 sweep-level rows
    And every file has exactly 6 sweeps
    And every row has condition, region, file_id, and sweep
    And the output includes both DH and VH regions
    And counts by region and condition are written to outputs/features/region_condition_cell_counts.csv
```

```gherkin
@step02 @R2 @R4 @feature-reliability
Feature: feature reliability controls accepted-fit weighting
  Scenario: missing and redundant features are down-weighted
    Given a feature table by sweep
    When reliability weights are computed
    Then features with high missingness receive lower weights
    And highly correlated features are flagged as redundant
    And the accepted-fit loss uses the reliability-weighted feature set
```

```gherkin
@step02 @R2 @region-thresholds
Feature: DH and VH thresholds are primary, not an afterthought
  Scenario: feature thresholds retain brain-region structure
    Given a sweep-level feature table with DH and VH cells
    When thresholds are built
    Then the primary threshold table is indexed by region, condition, sweep, and feature
    And any region-pooled fallback row is labeled with threshold_scope "region_pooled"
    And reviewer-facing analyses use region-specific thresholds when the stratum has enough data
```

```gherkin
@step02 @R2 @region-statistics
Feature: regional differences are quantified before fitting claims are made
  Scenario: DH/VH effects are summarized for key kinetic features
    Given the feature table by sweep
    When region-effect summaries are computed
    Then each key feature has DH vs VH effect estimates by condition and sweep
    And small strata are flagged
    And region-blind pooling is reported only as a sensitivity analysis
```

### Notebook required

`analysis/02_rebuild_atf_thresholds.ipynb`

The notebook must show:

- DH/VH cell counts by condition;
- feature distributions faceted by region, condition, and sweep;
- primary region-specific threshold table;
- region-pooled/global-pooled sensitivity thresholds, explicitly labeled;
- region-effect summary for key kinetic features.

---

## Step 03 — Combined identifiability screen: soft structural inspection, profile likelihood, and FIM

**Pareto rank:** 3. Directly addresses the conceptual degeneracy objection without overclaiming a full symbolic STRIKE-GOLDD implementation.

**Primary output:** `outputs/identifiability/effective_parameter_map.csv`, `outputs/identifiability/profile_likelihoods.csv`, `outputs/identifiability/fim_spectrum.csv`, `outputs/identifiability/fim_mode_loadings.csv`.

### Scientific objectives

1. **Resolve R1 strongly:** distinguish obvious structural non-separability, practical non-identifiability, sloppiness, and candidate degeneracy.
2. **Resolve R4 strongly:** show which parameters or combinations Vm can constrain.
3. **Resolve R7 partially:** create a clean definitions/diagnostic figure and table.

This step should be presented as a **structural-identifiability-informed practical identifiability analysis**, not as a complete STRIKE-GOLDD proof unless a formal symbolic tool is actually implemented.

### Technical objectives

- Use effective parameters as the primary coordinate system.
- Perform equation-level inspection for products, ratios, and scaling combinations that cannot be interpreted separately from Vm-only observations.
- Write an `effective_parameter_map.csv` classifying each raw parameter as `direct_candidate`, `effective_combination_member`, `fixed_constant`, `protocol_input`, or `weakly_interpretable`.
- Demonstrate exact or near-exact structural confoundings where possible, starting with `P_gap_eff = d × pk`.
- Profile selected raw/effective parameters by fixing one value, refitting nuisance parameters, and reporting the best achievable loss.
- Interpret profiles as `clear_valley`, `broad_valley`, `flat_unbounded`, or `boundary_hit`.
- Compute finite-difference Jacobian and FIM around verified representative centers before Step 04, then allow later reuse on Step 04 accepted cell-specific centers.
- Report eigenvalue spectra on log scale and map stiff/sloppy modes to raw/effective parameters.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Equation-level structural inspection | Required | Use to define effective parameters before fitting interpretation. |
| Exact invariance demonstrations | Required where possible | Start with `d × pk`; add other product/ratio demonstrations if valid. |
| Profile likelihood | Main practical identifiability evidence | Use for representative conditions and key effective parameters. |
| FIM finite differences | Sloppiness diagnostic | Use after reparameterization, around verified representative centers; later repeat on Step 04 accepted cell-specific centers when available. |
| Full STRIKE-GOLDD or symbolic tool | Optional backlog | Only claim it if implemented and documented. |

### How to verify

- `effective_parameter_map.csv` is written and lists every fitted/raw parameter.
- `d`/`pk` confounding is represented as `P_gap_eff`.
- Profile curves are saved with explicit interpretation classes.
- FIM output is finite and symmetric after stabilization.
- Eigenvalues show which directions are stiff/sloppy.
- Notebook text states that flat profiles and sloppy directions are not interpreted as biological degeneracy.

### Gherkin specifications

```gherkin
@step03 @R1 @effective-parameters @soft-structural
Feature: equation-level structural inspection defines effective parameters
  Scenario: raw parameters that enter only through products or ratios are not overinterpreted
    Given the astrocyte model equations and a parameter dictionary
    When the identifiability screen builds the effective-parameter map
    Then d and pk are classified as effective_combination_member
    And P_gap_eff is classified as primary_interpretable
    And gamma_t_eff, gamma_s_eff, and volume_ratio_wa_wo are reported
    And the notebook states that this is a structural-inspection screen, not a full STRIKE-GOLDD proof
```

```gherkin
@step03 @R1 @profile-likelihood
Feature: practical identifiability by profile likelihood
  Scenario: a parameter profile is classified by shape
    Given a verified representative fit center
    When one effective parameter is profiled and nuisance parameters are refit
    Then the profile curve is saved
    And the profile is classified as clear_valley, broad_valley, flat_unbounded, or boundary_hit
    And parameters with flat profiles are not interpreted as direct molecular estimates
```

```gherkin
@step03 @R1 @sloppiness @FIM
Feature: FIM sloppiness spectrum
  Scenario: Vm/features expose stiff and sloppy parameter combinations
    Given a verified representative fit center
    When the finite-difference Jacobian of Vm/features is computed
    Then the FIM eigen-spectrum is saved
    And each mode is annotated by dominant raw and effective parameters
    And sloppy directions are not interpreted as biological degeneracy
```

### Notebook required

`analysis/03_combined_identifiability_profiles_fim.ipynb`

The notebook must show:

- effective-parameter map;
- exact `d × pk` invariance diagnostic;
- at least one profile-likelihood curve per key effective parameter class;
- FIM eigenvalue spectrum;
- stiff/sloppy mode loading table;
- short reviewer-facing interpretation text.

---

## Transition from identifiability diagnostics to reviewer-facing ensembles

Step 03 uses verified representative fit centers to diagnose raw/effective identifiability limits. These centers are not the final reviewer-facing accepted ensembles. The reviewer-facing accepted ensembles are constructed in Step 04 by fitting each cell across its six sweeps with shared cell-level parameters and explicit held-out-current checks.

Consequences:

1. Step 03 can support statements about structural non-separability, practical non-identifiability, and sloppiness.
2. Step 03 cannot, by itself, support biological degeneracy or DH/VH mechanism-enrichment claims.
3. Step 04 is the first step that creates the accepted cell-specific inference target for downstream mechanism decomposition.
4. Step 05 is the first step that can evaluate candidate mechanism regimes from accepted cell-specific ensembles.
5. Step 06 is required before any mechanism regime is described as robust under prediction or perturbation.


## Step 04 — Cell-specific six-sweep fitting and accepted ensemble construction

**Pareto rank:** 4. Builds the reviewer-facing accepted ensembles that make downstream mechanistic decomposition meaningful. This step fits each cell across its six ordered sweeps with one shared cell-level parameter set and current-specific known inputs or nuisance terms.

**Primary output:** `outputs/cell_fits/cell_fit_candidates.csv`, `outputs/cell_fits/accepted_cell_ensembles.csv`, `outputs/cell_fits/cell_fit_quality_summary.csv`, `outputs/cell_fits/heldout_current_screen.csv`, `outputs/cell_fits/acceptance_contract.csv`.

### Scientific objectives

1. **Resolve R2 strongly:** use empirical region-aware thresholds and feature reliability weights when defining accepted fits.
2. **Resolve R4 strongly:** reduce Vm-only overfitting by requiring one shared mechanism to explain six sweeps of a cell.
3. **Resolve R6 partially:** establish held-out-current prediction as part of the accepted-ensemble contract.
4. **Prepare R5:** create cell-level accepted ensembles suitable for mechanism decomposition in Step 05.
5. **Protect R1:** prevent single-current non-identifiability from being mistaken for cell-level degeneracy.

This step should be the main source of reviewer-facing accepted ensembles. Historical single-current DBs remain useful for initialization, parameter-range priors, debugging, and method comparison, but not for final mechanism claims.

### Technical objectives

- Fit each ATF file/cell jointly across its six sweeps.
- Use one shared cell-level parameter vector per candidate fit.
- Treat current level, stimulus amplitude, or bath-driving terms as known protocol inputs or clearly labeled nuisance terms.
- Use effective parameters as primary optimization/reporting coordinates where possible: `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`.
- Use Step 02 reliability-weighted feature thresholds in the acceptance contract.
- Use a composite loss with:
  - baseline-subtracted Vm trace loss;
  - reliability-weighted feature loss;
  - binary penalties for plateau/undershoot presence when appropriate;
  - weak priors or penalties for physiological plausibility;
  - explicit failure penalties for non-finite simulations.
- Implement at least two validation modes:
  - fit all six sweeps and report fit quality;
  - fit five sweeps and predict the held-out sixth sweep, rotated across all six sweeps.
- Preserve `file_id`, `region`, `condition`, `sweep`, `candidate_id`, `fit_scope`, `heldout_sweep`, and `provenance_status` in all outputs.
- Record whether a candidate is accepted by trace error, feature contract, held-out prediction, and simulation health.
- Allow historical DB best/top candidates only as optional initialization seeds with `seed_source = legacy_db`, not as accepted final candidates. Only Step 00 `verified` DBs may be used as initialization seeds for reviewer-facing runs; `unresolved` or `missing_source` DBs are debug-only.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Cell-specific six-sweep joint fit | Primary | Required reviewer-facing accepted ensemble source. |
| Five-sweep fit with held-out prediction | Primary validation | Required before downstream robust mechanism claims. |
| Historical DB initialization | Optional speed/debug | Allowed only from Step 00 `verified` DBs as initialization; must not determine acceptance by itself. Unresolved sources are debug-only. |
| Region-specific acceptance thresholds | Primary | Use when stratum is sufficient. |
| Region-pooled thresholds | Sensitivity/shrinkage fallback | Use only when small strata make region-specific thresholds unstable. |
| Global thresholds | Negative-control sensitivity | Never use as the primary reviewer-facing acceptance rule. |

### How to verify

- `accepted_cell_ensembles.csv` contains `file_id`, `region`, `condition`, `candidate_id`, and shared cell-level parameters.
- Each accepted candidate links to six sweep-level simulations or a documented held-out prediction split.
- Every cell reports the number of accepted candidates and failure counts.
- Acceptance is based on both trace quality and Step 02 reliability-weighted feature contracts.
- Held-out prediction errors are reported by `region × condition × sweep`.
- Historical DB seeds, when used, are labeled as seeds only.

### Gherkin specifications

```gherkin
@step04 @R2 @R4 @cell-six-sweep-fit
Feature: one shared cell mechanism fits six ordered sweeps
  Scenario: a cell-level candidate explains all six sweeps
    Given one ATF cell with six sweeps
    And region-aware feature thresholds from Step 02
    When the six-sweep fitting pipeline optimizes a shared parameter vector
    Then the output contains one candidate_id linked to six sweep simulations
    And the candidate preserves file_id, region, condition, and sweep
    And the acceptance decision uses reliability-weighted trace and feature criteria
```

```gherkin
@step04 @R6 @heldout-current
Feature: held-out current prediction is part of accepted ensemble construction
  Scenario: five sweeps are fit and the sixth sweep is predicted
    Given one ATF cell with six ordered sweeps
    When the pipeline fits five sweeps and predicts the held-out sweep
    Then the held-out sweep error is reported
    And the held-out feature-pass fraction is reported
    And the candidate is not marked reviewer-facing if held-out prediction fails the configured tolerance
```

```gherkin
@step04 @R4 @legacy-seeds
Feature: historical single-current DBs can seed but not define final accepted ensembles
  Scenario: legacy candidates are used as initialization
    Given verified historical DB parameters
    When they are used to initialize a cell-specific six-sweep fit
    Then the output records seed_source = legacy_db
    And final acceptance depends only on cell-specific six-sweep fit and validation criteria
    And unresolved historical DBs cannot produce reviewer-facing accepted candidates
```

```gherkin
@step04 @R2 @region-contract
Feature: accepted cell ensembles retain region identity
  Scenario: accepted candidates are summarized by region and condition
    Given cell-level accepted candidates
    When acceptance summaries are written
    Then every row includes region and condition
    And DH/VH summaries are reported separately
    And small strata are flagged rather than silently pooled
```

### Notebook required

`analysis/04_cell_specific_six_sweep_fitting.ipynb`

The notebook must show:

- cell-level fit inventory by `region × condition`;
- accepted candidate counts per cell;
- example six-sweep fit overlays for at least one DH and one VH cell per condition when available;
- held-out-current prediction summaries;
- acceptance contract table;
- failure/provenance table.

### Tests required before implementation

- Unit: cell protocol builder; effective-parameter conversion; feature-contract scoring; held-out split generator.
- Functional: one cell can be fit or scored across all six sweeps.
- Integration: a small subset of cells produces `accepted_cell_ensembles.csv` and held-out summaries.

---

## Step 05 — Mechanistic decomposition of accepted cell ensembles

**Pareto rank:** 5. Converts cell-specific accepted ensembles into mechanism-level evidence. This is the primary step for R5 and the first step where candidate mechanism regimes may be assessed. These regimes remain candidate regimes until Step 06 predictive and perturbation checks support them.

**Primary output:** `outputs/mechanisms/accepted_fit_mechanisms.csv`, `outputs/mechanisms/mechanism_clusters.csv`, `outputs/mechanisms/representatives.csv`, `outputs/mechanisms/region_mechanism_enrichment.csv`, `outputs/mechanisms/geometry_classification.csv`, `outputs/mechanisms/bootstrap_cluster_stability.csv`, `outputs/mechanisms/claim_scope_table.csv`, and `outputs/mechanisms/analysis_summary.json`.

### Scientific objectives

1. **Resolve R5 strongly:** show whether accepted parameter regimes correspond to distinct Kir/gap/leak buffering balances.
2. **Resolve R1 partially:** distinguish continuous compensation manifolds from separated mechanism modes.
3. **Resolve R4 partially:** interpret accepted fits through fluxes, effective parameters, and hidden states rather than raw parameter values.
4. **Resolve R2/R5 partially:** summarize candidate mechanism regimes by DH/VH and condition as population-level cell effects.
5. **Prepare R6:** select representative mechanism-diverse candidates for perturbation and predictive checks.

### Technical objectives

- Load `outputs/cell_fits/accepted_cell_ensembles.csv` from Step 04.
- Simulate accepted cell-level candidates with hidden outputs for all six sweeps.
- Compute flux summaries:
  - Kir integral and peak;
  - gap-junction integral and peak;
  - leak integral and peak;
  - K_o peak, final, recovery error;
  - gap/Kir ratio;
  - proxy validity metrics for `ΔK_a,t` versus `K_o`.
- Cluster in effective/mechanism space, not raw parameter space alone.
- Use `P_gap_eff = d × pk`, `gamma_t_eff`, `gamma_s_eff`, and `volume_ratio_wa_wo` as primary geometry coordinates.
- Estimate whether accepted sets form continuous compensation manifolds or separated modes.
- Test cluster stability by bootstrap resampling at the cell level.
- Test interpolations between candidate modes in log/effective-parameter space.
- Select representative fits that preserve function while maximizing mechanism diversity.
- Report cluster occupancy and enrichment by `region × condition`, with cell-level counts and small-stratum flags.
- Keep all failed simulations and proxy failures as explicit statuses.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Best-trial only | Debug only | Not reviewer-facing. |
| Cell-specific accepted ensemble | Main analysis | Required for reviewer-facing mechanism claims. |
| Mechanism-space clustering | Main clustering | Prefer over raw-parameter UMAP for claims. |
| Bootstrap cluster stability | Required for separated-mode claims | A cluster is reviewer-facing only if it is stable under cell-level resampling. |
| Interpolation between candidate modes | Required for separated-mode claims | If interpolated points remain accepted, call the structure compensation rather than separated degeneracy. |
| Representative maximin selection | Figure construction | Choose same-function but mechanism-diverse examples. |
| Region enrichment | Population-level summary | Report as cell-level region/condition association, not paired pharmacology or animal-level phenotype. |

### How to verify

- Accepted ensemble size is reported per `region × condition × cell`.
- Each accepted fit has flux and proxy metrics for each sweep.
- Clusters differ in mechanism metrics while preserving functional metrics.
- Cluster summaries include region and condition occupancy, and no cluster claim is based on pooled regions alone.
- Continuous accepted sets are labeled `compensation_manifold`.
- Only stable separated sets with distinct flux decompositions are labeled `candidate_mechanism_regimes_pending_validation` before Step 06. The stronger label `candidate_degenerate_regimes` is allowed only after Step 06 predictive or perturbation checks support functional robustness.
- Candidate degeneracy labels include a status field indicating whether predictive validation and perturbation support are still pending.

### Gherkin specifications

```gherkin
@step05 @R5 @accepted-ensemble
Feature: accepted cell fits are translated into mechanism summaries
  Scenario: every accepted cell candidate has hidden-current metrics
    Given Step 04 accepted cell-specific six-sweep candidates
    When the mechanism pipeline simulates them
    Then each fit has I_Kir, I_kgap, I_leak, K_o, gap/Kir ratio, and proxy validity
    And failed simulations are reported without corrupting the ensemble
```

```gherkin
@step05 @R1 @R5 @mechanism-diversity
Feature: same-function but different-mechanism representatives
  Scenario: representatives preserve function while maximizing mechanism distance
    Given an accepted ensemble with functional and mechanism columns
    When representatives are selected by maximin mechanism distance
    Then selected representatives remain within functional tolerances
    And their mechanism distances exceed the configured diversity threshold
```

```gherkin
@step05 @R1 @geometry
Feature: accepted-fit geometry separates compensation from modes
  Scenario: interpolated parameters test whether two clusters are disconnected
    Given two candidate accepted clusters in effective-parameter space
    When representative centers are interpolated in log/effective coordinates
    Then each interpolated point is simulated and scored
    And the pair is classified as compensation_manifold when interpolations remain accepted
    And the pair is classified as separated_modes only when interpolations pass through poor-fit regions
    And separated modes remain pending_validation until Step 06 robustness checks are passed
```

```gherkin
@step05 @R2 @R5 @region-mechanisms
Feature: mechanism regimes are summarized by brain region
  Scenario: mechanism-cluster occupancy is reported separately for DH and VH
    Given accepted cell-level ensembles with mechanism-cluster labels
    When mechanism enrichment is summarized
    Then the output reports counts and fractions by region, condition, and cluster
    And DH/VH differences are reported as population-level cell effects
    And no phenotype claim is made from region enrichment alone
```

### Notebook required

`analysis/05_mechanistic_decomposition.ipynb`

The notebook must show:

- accepted cell ensemble inventory;
- flux decomposition figures for representative candidates;
- mechanism-space clustering plots;
- compensation-vs-separated-mode interpolation diagnostics;
- bootstrap stability table;
- DH/VH and condition occupancy table;
- explicit claim-scope table.

---

## Step 06 — Predictive validation, posterior predictive checks, and perturbation robustness

**Pareto rank:** 6. Tests whether accepted ensembles predict beyond fitted traces and remain functionally robust under perturbations.

**Primary output:** `outputs/predictive_validation/heldout_current_errors.csv`, `outputs/predictive_validation/prediction_intervals.csv`, `outputs/predictive_validation/feature_distribution_ppc.csv`, `outputs/predictive_validation/perturbation_sweeps.csv`, `outputs/predictive_validation/robustness_summary.csv`.

### Scientific objectives

1. **Resolve R6 strongly:** test robustness beyond fitted sweeps.
2. **Resolve R1/R5 partially:** separate parameter compensation that only fits training traces from mechanism regimes that preserve function under prediction and perturbation.
3. **Resolve R2 partially:** compare simulated predictive distributions to empirical feature distributions by `region × condition × sweep`.
4. **Resolve R7 partially:** create clear predictive validation figures.

### Technical objectives

- Load accepted cell ensembles from Step 04 and mechanism labels from Step 05.
- Aggregate held-out-current results from Step 04.
- Produce prediction intervals from accepted ensembles.
- Compare feature distributions from simulations with empirical Step 02 distributions.
- Run perturbation sweeps for:
  - bath coupling `epsilon`;
  - stimulus duration;
  - baseline `K_o`;
  - current amplitude scaling;
  - plausible physiologic nuisance terms if implemented.
- Summarize robustness by mechanism cluster, region, condition, and sweep.
- Label mechanisms as `predictive_supported`, `prediction_limited`, or `fit_only`.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Leave-one-current-out within cell | Primary | Required for reviewer-facing robustness. |
| Low-to-high prediction | Stress test | Fit lower currents and predict high-current sweeps. |
| High-to-low prediction | Stress test | Fit high currents and predict lower-current sweeps. |
| Posterior predictive feature checks | Primary | Compare accepted-ensemble feature distributions to Step 02 empirical bands. |
| Perturbation sweeps | Primary | Test whether candidate mechanisms preserve K buffering under altered inputs. |

### How to verify

- Held-out prediction errors are written for all attempted cells and sweeps.
- Prediction intervals are finite and traceable to accepted candidates.
- Feature PPC tables include `region`, `condition`, `sweep`, and `feature`.
- Perturbation outputs include mechanism labels and robustness classifications.
- Any mechanism cluster that fails prediction or perturbation is not labeled as reviewer-facing biological degeneracy.

### Gherkin specifications

```gherkin
@step06 @R6 @heldout-current
Feature: accepted ensembles predict held-out currents
  Scenario: a held-out sweep is predicted from the remaining sweeps
    Given accepted cell-specific candidates from Step 04
    When a held-out current is predicted
    Then trace error and feature-pass metrics are reported
    And results are summarized by region, condition, and sweep
```

```gherkin
@step06 @R2 @posterior-predictive
Feature: simulated feature distributions are compared with empirical distributions
  Scenario: accepted ensemble predictions match empirical feature bands
    Given empirical Step 02 feature thresholds
    And accepted ensemble simulations
    When posterior predictive feature checks are computed
    Then the output reports coverage by region, condition, sweep, and feature
    And redundant or low-reliability features are weighted accordingly
```

```gherkin
@step06 @R6 @perturbation
Feature: mechanism regimes are stress-tested by perturbation
  Scenario: accepted mechanism clusters are simulated under altered inputs
    Given mechanism-labeled accepted ensembles
    When bath coupling, stimulus duration, baseline K_o, or current amplitude are perturbed
    Then functional buffering metrics are reported
    And clusters are classified by robustness
```

### Notebook required

`analysis/06_predictive_validation_and_perturbation.ipynb`

---

## Step 07 — Assumption sensitivity: gating, proxy, and compartment split

**Pareto rank:** 7. Addresses model-assumption criticism after the main inference pipeline is stable.

**Primary output:** `outputs/assumption_sensitivity/model_comparison.csv`, `outputs/assumption_sensitivity/gating_family_comparison.csv`, `outputs/assumption_sensitivity/proxy_validity_by_ensemble.csv`, `outputs/assumption_sensitivity/compartment_split_sensitivity.csv`.

### Scientific objectives

1. **Resolve R3 strongly:** show whether conclusions are brittle to gating form, proxy choice, or local/syncytial split.
2. **Resolve R6 partially:** quantify where model predictions diverge under assumption changes.
3. **Resolve R1/R5 partially:** test whether candidate mechanism regimes are stable across plausible model formulations.

### Technical objectives

- Implement separate model-family fits or scored variants using identical data splits and acceptance contracts:
  - sigmoid;
  - tanh;
  - Hill;
  - soft-threshold linear;
  - hard-threshold;
  - double-sigmoid, if identifiable enough.
- Quantify proxy validity between `ΔK_a,t` and `K_o` for accepted ensembles.
- If proxy validity fails systematically, implement or score a minimal explicit ECS variant.
- Compare two-state local/syncytial intracellular K model against a one-state intracellular variant.
- Report whether accepted mechanisms and robustness classifications persist.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Same split/same loss model-family comparison | Primary | Required for gating sensitivity. |
| Proxy correlation and lag metrics | Primary | Required for intracellular-K-as-ECS-proxy criticism. |
| Minimal explicit ECS variant | Conditional | Required if proxy validity fails in reviewer-facing regimes. |
| One-state intracellular variant | Sensitivity | Required to test local/syncytial split dependence. |

### How to verify

- Every compared model family uses identical splits, thresholds, and reporting metrics.
- Model comparison tables include fit quality, held-out prediction, feature PPC, and robustness metrics.
- Proxy validity is reported by region, condition, sweep, and mechanism cluster.
- Assumption sensitivity text explicitly states which claims are robust and which are model-dependent.

### Gherkin specifications

```gherkin
@step07 @R3 @gating-sensitivity
Feature: gating-family conclusions are compared under identical contracts
  Scenario: different gating forms are evaluated fairly
    Given accepted-fit contracts and data splits
    When each gating family is fit or scored
    Then the output reports fit, prediction, and mechanism metrics with identical definitions
    And mechanism claims are marked robust only if they persist across configured families
```

```gherkin
@step07 @R3 @proxy-validity
Feature: intracellular K proxy validity is quantified
  Scenario: ΔK_a,t is compared with K_o
    Given accepted ensemble simulations with hidden states
    When proxy validity metrics are computed
    Then Pearson/Spearman correlation, RMSE after scaling, and lag are reported
    And failed proxy regimes are not described as reliable ECS K readouts
```

```gherkin
@step07 @R3 @compartment-split
Feature: local/syncytial split sensitivity is tested
  Scenario: one-state and two-state intracellular formulations are compared
    Given the same empirical data and acceptance contract
    When the split and non-split variants are evaluated
    Then the output reports whether accepted mechanism structure persists
```

### Notebook required

`analysis/07_assumption_sensitivity.ipynb`

---

## Step 08 — Parameter plausibility and constrained reruns

**Pareto rank:** 8. Converts accepted-ensemble results into biophysically cautious parameter interpretation.

**Primary output:** `outputs/parameter_plausibility/parameter_range_audit.csv`, `outputs/parameter_plausibility/effective_parameter_plausibility.csv`, `outputs/parameter_plausibility/constrained_rerun_comparison.csv`, `outputs/parameter_plausibility/interpretability_status.csv`.

### Scientific objectives

1. **Resolve R4 strongly:** quantify whether accepted parameters are within plausible ranges and whether they are identifiable.
2. **Resolve R1 partially:** avoid labeling broad or boundary-hit parameter distributions as mechanisms.
3. **Resolve R5 partially:** keep only physiologically interpretable mechanism claims in the main text, moving weakly interpretable claims to limitations or supplement.

### Technical objectives

- Report raw and effective parameter distributions by region, condition, mechanism cluster, and accepted-candidate status.
- Compare accepted parameter ranges with documented broad physiological or modeling ranges.
- Distinguish:
  - `within_range`;
  - `out_of_range`;
  - `weakly_identified`;
  - `effective_only`;
  - `physiologically_interpretable`.
- Run constrained or penalized refits for high-priority cells/mechanism clusters.
- Compare unconstrained versus constrained fits by trace error, feature pass rate, held-out prediction, mechanism decomposition, and perturbation robustness.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Broad plausibility audit | Primary | Required for all accepted ensembles. |
| Effective-parameter plausibility | Primary | Use for reduced-model interpretation. |
| Constrained reruns | Targeted | Use for parameters repeatedly out of range or boundary-hit. |
| Penalty priors | Optional | Use when hard constraints degrade fit unrealistically. |

### How to verify

- Every reported parameter has a plausibility and identifiability status.
- Effective parameters are separated from raw parameters.
- Constrained rerun comparison states whether reviewer-facing mechanism conclusions persist.
- A parameter inside bounds is not automatically labeled interpretable if Step 03 shows weak identifiability.

### Gherkin specifications

```gherkin
@step08 @R4 @parameter-plausibility
Feature: accepted parameters are audited for plausibility and identifiability
  Scenario: each accepted parameter receives an interpretation status
    Given accepted cell ensembles and Step 03 identifiability results
    When parameter plausibility is audited
    Then each parameter is labeled within_range or out_of_range
    And each parameter is labeled identifiable, weakly_identified, or effective_only
    And physiologically_interpretable is true only when both plausibility and identifiability criteria support it
```

```gherkin
@step08 @R4 @constrained-rerun
Feature: constrained inference tests whether claims depend on implausible parameters
  Scenario: constrained and unconstrained accepted ensembles are compared
    Given a set of high-priority cells or mechanism clusters
    When constrained reruns are performed
    Then fit quality, prediction, and mechanism metrics are compared
    And claims that disappear under reasonable constraints are downgraded
```

### Notebook required

`analysis/08_parameter_plausibility_and_constrained_reruns.ipynb`

---

## Step 09 — Reviewer-facing figures, tables, and rebuttal traceability

**Pareto rank:** 9. Turns the computational pipeline into manuscript and response-letter material.

**Primary output:** `outputs/reviewer_figures/figure_manifest.csv`, `outputs/reviewer_figures/reviewer_traceability_table.csv`, `outputs/reviewer_figures/main_figure_sources.csv`, `outputs/reviewer_figures/supplement_figure_sources.csv`.

### Scientific objectives

1. **Resolve R7 strongly:** produce clear figures, units, axes, layouts, and tables.
2. **Resolve R1–R6 communication:** map every claim to the exact computational output that supports it.
3. **Avoid overclaiming:** separate full, partial, provisional, and unsupported claims in the response letter and manuscript.

### Technical objectives

- Build a traceability table mapping every output figure/table to reviewer critique IDs.
- Export manuscript-ready figures from Steps 00–08.
- Use consistent units and labels.
- Prefer composite panels and uncertainty bands over many individual traces.
- Include figure source CSVs for every plotted panel.
- Generate a claim table with:
  - claim text;
  - critique IDs addressed;
  - source notebook;
  - source output file;
  - claim strength;
  - unresolved limitation.

### Required figure families

| Figure family | Source steps | Purpose |
|---|---|---|
| Data/provenance/thresholds | 00, 02 | R2/R7: show dataset contract and empirical uncertainty. |
| Identifiability/effective parameters | 01, 03 | R1/R4: show non-separability, sloppiness, and effective reporting coordinates. |
| Cell-specific fit and held-out prediction | 04, 06 | R2/R4/R6: show robust accepted ensembles and predictive validation. |
| Mechanistic decomposition | 05 | R5: show Kir/gap/leak regimes. |
| Assumption sensitivity | 07 | R3: show robustness to gating/proxy/split choices. |
| Parameter plausibility | 08 | R4: show constrained interpretation. |

### Gherkin specifications

```gherkin
@step09 @R7 @traceability
Feature: every figure and table is traceable to reviewer objections
  Scenario: reviewer-facing outputs are assembled
    Given outputs from Steps 00 through 08
    When the figure manifest is generated
    Then every figure panel has a source file
    And every figure panel maps to one or more reviewer critique IDs
    And every claim has a claim strength and limitation field
```

```gherkin
@step09 @R7 @figure-quality
Feature: manuscript-ready figures have consistent units and layouts
  Scenario: figures are exported for manuscript use
    Given figure source tables
    When reviewer-facing figures are generated
    Then axes have units
    And panels have consistent labels
    And uncertainty bands are preferred over excessive overplotted traces
```

### Notebook required

`analysis/09_reviewer_figures_and_traceability.ipynb`

---

## Claim-strength vocabulary

All notebooks that produce reviewer-facing tables must use this vocabulary.

| Claim strength | Meaning |
|---|---|
| `debug_only` | Useful for implementation or troubleshooting, not reviewer-facing. |
| `provisional` | Suggestive result that requires a later step for final claim support. |
| `partial_response` | Addresses part of a reviewer objection but leaves stated limitations. |
| `reviewer_facing_supported` | Supported by data, acceptance criteria, mechanism analysis, and appropriate validation for the claim scope. |
| `downgraded_by_validation` | Initial result failed prediction, perturbation, plausibility, or assumption-sensitivity checks. |
| `unsupported` | Should not be used as a manuscript claim. |

## Minimal reviewer-facing claim path

The strongest defensible path for the central degeneracy claim is:

1. Step 03 shows that raw parameter multiplicity is not automatically degeneracy.
2. Step 04 builds accepted cell-level six-sweep ensembles with held-out prediction screens.
3. Step 05 shows whether accepted ensembles form mechanism-distinct candidate regimes or only continuous compensation manifolds.
4. Step 06 tests whether candidate regimes preserve predictions and K-buffering function under held-out currents and perturbations.
5. Step 07 tests whether the conclusion is robust to model assumptions.
6. Step 08 limits parameter interpretation to plausible and identifiable/effective coordinates.
7. Step 09 maps only supported claims into figures and response text.
The term `degeneracy` should be used only when mechanism-distinct accepted regimes preserve function under prediction or perturbation. Otherwise use `non-identifiability`, `sloppiness`, `parameter compensation`, or `compensation manifold`.
