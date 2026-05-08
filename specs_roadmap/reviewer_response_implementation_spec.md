# Reviewer-response implementation specification

This specification defines a test-first, notebook-validated implementation plan for the astrocytic potassium-buffering model revision. It is ordered by Pareto priority: each step should increase reviewer-facing credibility before lower-impact refinements are attempted.

The target scientific reframing is:

> Multiple Vm-compatible parameter sets are not, by themselves, evidence of biological degeneracy. The revised pipeline must first remove obvious structural non-separabilities, quantify practical identifiability and sloppiness, and then reserve the term degeneracy for accepted ensembles that are mechanistically distinct, physiologically interpretable, and predictive under held-out currents or perturbations.

## Reviewer critique taxonomy used by the plan

| ID | Reviewer objection | Required computational response |
|---|---|---|
| R1 | Degeneracy is not distinguished from structural non-identifiability, practical non-identifiability, sloppiness, or parameter compensation. | Operational definitions, effective parameters, FIM/profile-likelihood diagnostics, and mechanism mapping. |
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
4. **No silent provenance assumptions.** Any ambiguous trace source, threshold source, or objective mismatch must be represented as an explicit status field.
5. **Mechanisms are not inferred from raw parameters alone.** Mechanistic claims require hidden-current or flux summaries.
6. **Claims are graded as full or partial.** A step may partially answer a reviewer objection, but the notebook must state what remains unresolved.
7. **Brain region is a first-class biological factor.** The 37 ATF files include dorsal hippocampus (`DH`) and ventral hippocampus (`VH`) cells. Region must be parsed, audited, retained in every table, used in thresholds/model evaluation, and shown in reviewer-facing summaries. Region-blind pooling is allowed only as an explicitly labeled sensitivity or shrinkage fallback, not as the primary analysis.

## Coverage audit against the current recommendation

This table is part of the specification. It records whether the development plan fully covers each recommendation and what must be refined before the item can be treated as reviewer-facing.

| Recommendation from analysis | Current coverage | Required refinement in this spec |
|---|---|---|
| Fit one cell across all 6 sweeps jointly, with one shared cell parameter set. | Partially covered: Step 06 currently describes condition/current-level multi-current validation and leaves full six-sweep/cell fitting as a backlog item. | Promote full cell-specific six-sweep fitting to the main Step 06 objective. Historical single-current DB transfer remains debug/triage only. |
| Treat DH/VH brain region as a biological factor. | Partially covered: Step 02 already uses `condition × region × sweep`, and Step 06 reports by region, but region is not yet a formal design contract. | Add a region-aware experimental-design contract; audit DH/VH counts; preserve region in every output; stratify thresholds and predictive checks by region; use region-blind pooling only as a labeled sensitivity or shrinkage fallback. |
| Reparameterize raw parameters into effective combinations before interpretation. | Well covered by Step 01 and Step 03. | Keep `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, and `volume_ratio_wa_wo` as primary reporting coordinates. Add profile interpretation rules: clear valley, flat profile, boundary hit, broad valley. |
| Merge structural and practical identifiability using a soft STRIKE-GOLDD-inspired workflow. | Partially covered: Step 03 separates FIM/profile and lists symbolic structural checks as optional. | Rename Step 03 as a combined structural-inspection + practical-profile workflow. Avoid claiming full STRIKE-GOLDD unless implemented. |
| Use FIM/sloppiness diagnostics. | Covered by Step 03. | Keep FIM after effective-parameter reparameterization and run on representative accepted centers, not all trials. |
| Analyze accepted-fit geometry as continuous compensation manifold vs separated modes. | Partially covered by Step 04. | Add bootstrap cluster stability and interpolation tests between candidate modes. Interpret continuous connected sets as compensation, not degeneracy. |
| Mechanistic decomposition of accepted regimes. | Covered by Step 04. | Make clear that old DB-derived mechanisms are provisional until repeated with final six-sweep accepted ensembles. |
| Assumption sensitivity for gating form. | Partially covered: sigmoid, tanh, Hill, and soft-threshold are included. | Add hard-threshold and double-sigmoid variants; compare all with identical data splits, loss definitions, and evaluation metrics. |
| Proxy and compartment-split sensitivity. | Covered by Step 05. | Keep explicit ECS variant optional unless proxy validity fails; keep one-state intracellular variant as sensitivity. |
| Parameter plausibility and constrained reruns. | Covered by Step 07. | Distinguish `within_range`, `identifiable`, and `physiologically_interpretable`; a parameter inside bounds may still be weakly identified. |
| Population-level posterior predictive checks. | Partially covered by Step 06 and Step 09. | Add explicit feature-distribution posterior predictive checks for `region × condition × sweep` groups using accepted ensembles. |
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
- Compute finite-difference Jacobian and FIM around representative accepted centers.
- Report eigenvalue spectra on log scale and map stiff/sloppy modes to raw/effective parameters.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Equation-level structural inspection | Required | Use to define effective parameters before fitting interpretation. |
| Exact invariance demonstrations | Required where possible | Start with `d × pk`; add other product/ratio demonstrations if valid. |
| Profile likelihood | Main practical identifiability evidence | Use for representative conditions and key effective parameters. |
| FIM finite differences | Sloppiness diagnostic | Use after reparameterization, around representative accepted centers. |
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
    Given an accepted representative fit
    When one effective parameter is profiled and nuisance parameters are refit
    Then the profile curve is saved
    And the profile is classified as clear_valley, broad_valley, flat_unbounded, or boundary_hit
    And parameters with flat profiles are not interpreted as direct molecular estimates
```

```gherkin
@step03 @R1 @sloppiness @FIM
Feature: FIM sloppiness spectrum
  Scenario: Vm/features expose stiff and sloppy parameter combinations
    Given a verified representative accepted center
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


## Step 04 — Accepted ensembles and mechanistic decomposition

**Pareto rank:** 4. Converts non-identifiable fits into provisional mechanism-level evidence. Historical single-current DB ensembles are useful for triage and figure/method development; final reviewer-facing mechanism claims should be repeated on the six-sweep cell-specific accepted ensembles from Step 06.

**Primary output:** `outputs/mechanisms/accepted_fit_mechanisms.csv`, `outputs/mechanisms/mechanism_clusters.csv`, `outputs/mechanisms/representatives.csv`, `outputs/mechanisms/region_mechanism_enrichment.csv`.

### Scientific objectives

1. **Resolve R5 strongly:** show whether accepted parameter regimes correspond to distinct Kir/gap/leak buffering balances.
2. **Resolve R1 partially:** distinguish continua from separated modes in mechanism space.
3. **Resolve R4 partially:** interpret accepted fits through fluxes rather than raw parameters.
4. **Resolve R2/R5 partially:** test whether candidate mechanism regimes are enriched differently in DH and VH without claiming paired or animal-level effects.

### Technical objectives

- Apply condition-specific thresholds from Step 02 to top-N or all complete trials.
- Simulate accepted fits with hidden outputs.
- Compute flux summaries: Kir integral, gap integral, leak integral, K_o peak/final/recovery, proxy validity.
- Cluster in effective/mechanism space, not raw parameter space alone.
- Estimate whether the accepted set forms a continuous compensation manifold or separated modes.
- Test cluster stability by bootstrap resampling.
- Test interpolations between candidate modes in log/effective-parameter space.
- Select representative fits that preserve function while maximizing mechanism diversity.
- Report mechanism-cluster occupancy and enrichment by `region × condition`, with cell-level counts.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Best-trial only | Debug only | Not reviewer-facing. |
| Top-N accepted ensemble | Main post-fit analysis | Use once thresholds are condition-specific. |
| Mechanism-space clustering | Main clustering | Prefer over raw-parameter UMAP for claims. |
| Bootstrap cluster stability | Required for mode claims | A cluster is reviewer-facing only if it is stable under resampling. |
| Interpolation between candidate modes | Required for separated-mode claims | If interpolated points also fit, call the structure compensation rather than separated degeneracy. |
| Representative maximin selection | Figure construction | Choose same-function but mechanism-diverse examples. |

### How to verify

- Accepted ensemble size is reported per condition/current.
- Each accepted fit has flux and proxy metrics.
- Clusters differ in mechanism metrics while preserving functional metrics.
- Cluster summaries include region and condition occupancy, and no cluster claim is based on pooled regions alone.
- Continuous accepted sets are labeled `compensation_manifold`; only stable separated sets with distinct flux decompositions are labeled `candidate_degenerate_regimes`.

### Gherkin specifications

```gherkin
@step04 @R5 @accepted-ensemble
Feature: accepted fits are translated into mechanism summaries
  Scenario: every accepted fit has hidden-current metrics
    Given accepted top-N fits for a condition/current
    When the mechanism pipeline simulates them
    Then each fit has I_Kir, I_kgap, I_leak, K_o, gap/Kir ratio, and proxy validity
    And failed simulations are reported without corrupting the ensemble
```

```gherkin
@step04 @R1 @R5 @mechanism-diversity
Feature: same-function but different-mechanism representatives
  Scenario: representatives preserve function while maximizing mechanism distance
    Given an accepted ensemble with functional and mechanism columns
    When representatives are selected by maximin mechanism distance
    Then selected representatives remain within functional tolerances
    And their mechanism distances exceed the configured diversity threshold
```


```gherkin
@step04 @R1 @geometry
Feature: accepted-fit geometry separates compensation from modes
  Scenario: interpolated parameters test whether two clusters are disconnected
    Given two candidate accepted clusters in effective-parameter space
    When representative centers are interpolated in log/effective coordinates
    Then each interpolated point is simulated and scored
    And the pair is classified as compensation_manifold when interpolations remain accepted
    And the pair is classified as separated_modes only when interpolations pass through poor-fit regions
```

```gherkin
@step04 @R2 @R5 @region-mechanisms
Feature: mechanism regimes are summarized by brain region
  Scenario: mechanism-cluster occupancy is reported separately for DH and VH
    Given accepted cell-level ensembles with mechanism-cluster labels
    When mechanism enrichment is summarized
    Then the output reports counts and fractions by region, condition, and cluster
    And DH/VH differences are reported as population-level cell effects
    And no phenotype claim is made from region enrichment alone
```

### Notebook required

`analysis/04_mechanistic_decomposition.ipynb`

---

## Step 05 — Assumption sensitivity: gating, proxy, and compartment split

**Pareto rank:** 5. Addresses model-assumption criticism after the main inference pipeline is stable.

**Primary output:** `outputs/assumption_sensitivity/model_comparison.csv`, `outputs/assumption_sensitivity/gating_family_comparison.csv`, `outputs/assumption_sensitivity/proxy_validity_by_ensemble.csv`, `outputs/assumption_sensitivity/compartment_split_sensitivity.csv`.

### Scientific objectives

1. **Resolve R3 strongly:** show whether conclusions are brittle to gating form, proxy choice, or local/syncytial split.
2. **Resolve R6 partially:** quantify where model predictions diverge under assumption changes.

### Technical objectives

- Implement separate model-family fits: sigmoid, Hill, soft-threshold/tanh, hard-threshold, and double-sigmoid.
- Keep model-family comparison outside the Optuna parameter search; do not mix `switching_function` as a categorical parameter inside one primary inference run.
- Compare fit quality, accepted ensemble size, CV error, prediction bands, and mechanism clusters by region and condition.
- Quantify `corr(ΔK_a,t, K_o)` and lag across accepted ensembles.
- Implement a one-intracellular-state sensitivity model or reduced split variant.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Sigmoid gating | Baseline manuscript form | Retain only if predictive and mechanism conclusions are not uniquely dependent on it. |
| Hill gating | Smooth saturating alternative | Compare with equal data splits and loss definitions. |
| Soft-threshold/tanh gating | Smooth threshold alternative | Compare with equal data splits and loss definitions. |
| Hard-threshold gating | Stress test | Use to determine whether conclusions depend on smooth gating. |
| Double-sigmoid gating | Flexible two-transition alternative | Use to test whether one threshold is too restrictive. |
| Same parameter count or penalized-complexity comparison | Primary | Prefer CV/predictive performance over in-sample loss alone. |
| Proxy validity metrics | Required | Report failures, not only successes. |
| Minimal ECS extension | Optional | Use if proxy validity frequently fails. |
| Single-state intracellular model | Sensitivity | Use to check whether split creates artificial degeneracy. |

### How to verify

- Model-family comparison uses identical data splits, region labels, and loss definitions.
- Gating variants are implemented as separate model-family runs and recorded in the comparison table.
- Proxy validity is quantified per accepted fit and summarized by condition/current/cluster.
- Any qualitative degeneracy claim is reported as robust or not robust across variants.

### Gherkin specifications

```gherkin
@step05 @R3 @gating-comparison
Feature: alternative gating functions are compared fairly
  Scenario: sigmoid, Hill, soft-threshold/tanh, hard-threshold, and double-sigmoid variants use the same protocol
    Given identical data splits, seeds, priors, and loss definitions
    When each gating family is fit and evaluated
    Then the comparison table reports in-sample loss, CV error, accepted ensemble size, prediction-band width, and mechanism persistence
    And no primary conclusion is based only on the best in-sample loss
```

```gherkin
@step05 @R3 @gating-implementation
Feature: gating variants are separate model-family runs
  Scenario: switching_function is not mixed as a categorical search parameter in the main inference
    Given the assumption-sensitivity configuration
    When the gating-family jobs are created
    Then sigmoid, Hill, soft-threshold/tanh, hard-threshold, and double-sigmoid are separate runs
    And each run writes its model_family identifier to the output table
```

```gherkin
@step05 @R3 @proxy-validity
Feature: intracellular K proxy validity is quantified rather than assumed
  Scenario: proxy validity can succeed or fail by condition and mechanism
    Given accepted simulations with K_o and ΔK_a,t outputs
    When proxy validity is computed
    Then Pearson/Spearman correlation, lag, scaled RMSE, and validity class are reported
    And the manuscript does not claim universal proxy validity when failures are present
```

### Notebook required

`analysis/05_assumption_sensitivity.ipynb`


## Step 06 — Cell-specific six-sweep fitting and predictive validation

**Pareto rank:** 6. The strongest response to robustness criticism and the main replacement for the old single-current fitting logic.

**Primary output:** `outputs/cross_validation/cell_six_sweep_fit_summary.csv`, `outputs/cross_validation/leave_one_sweep_out.csv`, `outputs/cross_validation/prediction_bands.csv`, `outputs/cross_validation/posterior_predictive_feature_checks.csv`.

### Scientific objectives

1. **Resolve R6 strongly:** show whether accepted parameter regimes predict held-out sweeps/currents.
2. **Resolve R2 strongly:** use the 37 independent ATF cells as the fitting/evaluation unit rather than treating conditions or regions as single representative traces.
3. **Resolve R1 partially:** identify whether candidate mechanism modes remain functionally equivalent outside fitted traces.
4. **Resolve R5 partially:** test whether distinct mechanisms diverge under held-out conditions.

### Technical objectives

- Build a cell-specific six-sweep objective with one shared biological/effective parameter set per cell.
- Carry each cell's `region` label through fitting, validation, posterior predictive checks, and accepted-ensemble summaries.
- Use a monotone ordered stimulus mapping across the six sweeps.
- Use a composite loss with trace, feature, binary-feature, prior, and failure-penalty terms.
- Fit all six sweeps jointly for each ATF cell, then produce an accepted-fit ensemble per cell.
- Fit 5 sweeps and predict the held-out sweep; rotate all six sweeps.
- Fit low sweeps and predict high sweeps; fit high sweeps and predict low sweeps.
- Produce prediction intervals from accepted ensembles.
- Perform population-level posterior predictive checks by comparing simulated feature distributions against empirical `region × condition × sweep` distributions.
- Report primary held-out errors separately for DH and VH before pooled summaries.
- Use region-specific thresholds from Step 02; use region-pooled thresholds only as explicitly labeled sensitivity/fallback.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Full six-sweep/cell fit using ATF cells | Main reviewer-facing inference | Required for final claims; must retain DH/VH labels. |
| Leave-one-sweep-out | Main validation | Required for R6. |
| Low-to-high/high-to-low extrapolation | Stronger stress test | Use as supplement. |
| Historical single-current DB post-hoc transfer | Debug/triage only | Not enough for final claims. |
| Condition-level multi-current fit | Optional fallback | Use only if cell-level fitting is computationally infeasible, and label claims as partial. |

### How to verify

- Every ATF cell has one six-sweep fitting record or an explicit failure reason, including its DH/VH region label.
- Every sweep appears once as held-out for each eligible cell.
- Prediction RMSE/features are reported per cell, region, condition, and sweep.
- Prediction bands are generated from accepted ensembles.
- Posterior predictive feature checks are generated for `region × condition × sweep` groups and region-pooled summaries are labeled as secondary.
- Failures are reported as limitations, not hidden.

### Gherkin specifications

```gherkin
@step06 @R2 @R6 @cell-six-sweep
Feature: cell-specific six-sweep model fitting
  Scenario: each ATF cell is fit with one shared parameter set across six sweeps
    Given an ATF-derived cell with six sweeps
    When the cell-specific fitting pipeline runs
    Then one shared biological/effective parameter set is optimized across all six sweeps
    And the cell's DH or VH region label is preserved in the fit output
    And sweep-specific stimulus amplitudes are monotone increasing
    And the output reports trace, feature, binary-feature, prior, and failure-penalty loss components
```

```gherkin
@step06 @R6 @cross-validation
Feature: leave-one-sweep-out predictive validation
  Scenario: each sweep is predicted from the other five sweeps
    Given a cell with six pump-current sweeps
    When the multi-sweep model is fit on five sweeps
    Then the held-out sweep trace and features are predicted
    And the error table reports trace loss and feature loss
    And prediction bands are saved
```

```gherkin
@step06 @R6 @extrapolation
Feature: low-to-high and high-to-low predictive checks
  Scenario: accepted mechanisms are stress-tested outside fitted sweep ranges
    Given accepted fits from low-sweep training data
    When high-sweep traces are predicted
    Then prediction errors are compared by mechanism cluster
    And clusters that diverge are flagged as non-equivalent outside the fitted regime
```

```gherkin
@step06 @R2 @R6 @posterior-predictive
Feature: population-level posterior predictive feature checks
  Scenario: accepted ensembles reproduce empirical feature distributions
    Given accepted cell-level ensembles and empirical ATF feature thresholds
    When simulated features are summarized by region, condition, and sweep
    Then empirical and simulated medians, IQRs, and coverage rates are reported
    And model failures to reproduce group variability are reported as limitations
```

```gherkin
@step06 @R2 @region-aware-errors
Feature: predictive errors are not hidden by region pooling
  Scenario: held-out prediction metrics are summarized separately for DH and VH
    Given leave-one-sweep-out results for fitted ATF cells
    When validation summaries are generated
    Then RMSE, feature loss, binary-feature accuracy, and coverage are reported by region, condition, and sweep
    And pooled summaries are labeled secondary
    And a region-specific failure cannot be hidden by a good pooled score
```

### Notebook required

`analysis/06_cell_six_sweep_predictive_validation.ipynb`


## Step 07 — Parameter plausibility and constrained inference

**Pareto rank:** 7. Turns parameter criticism into a controlled interpretation.

**Primary output:** `outputs/plausibility/parameter_distribution_audit.csv`, `outputs/plausibility/constrained_vs_unconstrained.csv`.

### Scientific objectives

1. **Resolve R4 strongly:** distinguish physiological parameters from effective model-reduction parameters.
2. **Resolve R1 partially:** show whether degeneracy-like structure persists under constrained priors.

### Technical objectives

- Define literature/working ranges for raw and effective parameters.
- Report median/IQR/range per region/condition/sweep/cluster, with pooled summaries marked secondary.
- Join plausibility tables with profile/FIM classifications from Step 03.
- Classify each parameter as `within_range`, `out_of_range`, `effective_only`, `identifiable`, `weakly_identifiable`, or `physiologically_interpretable`.
- Rerun constrained inference or apply penalties.
- Compare fit quality, CV error, accepted ensemble size, and mechanism clusters by region and condition.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Post-hoc plausibility audit | Immediate | Required for all accepted fits. |
| Penalized inference | Main constrained analysis | Prefer if hard bounds cause solver instability. |
| Hard bounded refit | Sensitivity | Use for parameters with clear physical ranges. |

### How to verify

- Every reported parameter has units or is explicitly labeled dimensionless/effective.
- Out-of-range parameters are flagged.
- Claims are based on effective combinations when raw values are not separately identifiable.
- Plausibility summaries are stratified by DH/VH region when cell-level fits are available.
- A parameter inside its biological range is not called physiologically interpretable unless Step 03 also supports identifiability.

### Gherkin specifications

```gherkin
@step07 @R4 @plausibility
Feature: parameter plausibility audit
  Scenario: every accepted parameter is classified against its intended interpretation
    Given accepted fits and a parameter-range specification
    When the plausibility audit runs
    Then each raw and effective parameter is labeled within_range, out_of_range, or effective_only
    And out-of-range raw parameters are not interpreted as direct physiology
```

```gherkin
@step07 @R4 @constrained-inference
Feature: constrained inference tests whether conclusions survive physiological priors
  Scenario: constrained and unconstrained fits are compared
    Given an unconstrained accepted ensemble
    When inference is rerun with priors or penalties
    Then fit quality, CV error, accepted ensemble size, and mechanism clusters are compared
    And loss of the degeneracy claim is reported if it occurs
```

### Notebook required

`analysis/07_parameter_plausibility.ipynb`

---

## Step 08 — Perturbation robustness

**Pareto rank:** 8. Turns mechanism differences into testable predictions.

**Primary output:** `outputs/perturbations/perturbation_sweep_summary.csv`, `outputs/perturbations/homeostasis_stability_by_cluster.csv`.

### Scientific objectives

1. **Resolve R6 strongly:** test robustness beyond fitted traces.
2. **Resolve R5 strongly:** show how mechanism clusters respond differently to perturbations.
3. **Resolve R3 partially:** evaluate bath coupling and stimulus assumptions.

### Technical objectives

- Perturb `epsilon`, baseline `K_o`, stimulus duration, stimulus amplitude, and selected physiological/effective parameters.
- Simulate accepted representatives and ensembles.
- Classify homeostasis stability using Vm final error, K_o peak, K_o recovery, and finite-state criteria.
- Compare cluster-level perturbation responses by region and condition where cell-level ensembles are available.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Representative perturbations | First visual diagnostic | Use for figures. |
| Full accepted-ensemble perturbations | Reviewer-facing robustness | Use for statistics. |
| One-at-a-time perturbations | Interpretability | Required initially. |
| Latin hypercube perturbations | Optional | Use if interactions are important. |

### How to verify

- Each perturbation has baseline and perturbed simulations.
- Stability criteria are explicit.
- Cluster divergence is quantified.
- Perturbation robustness summaries report DH and VH separately before pooled results.

### Gherkin specifications

```gherkin
@step08 @R6 @perturbations
Feature: accepted mechanisms are stress-tested under perturbations
  Scenario: perturbation sweeps produce stability classifications
    Given accepted representative fits and perturbation definitions
    When the perturbation pipeline runs
    Then each simulation has finite status, flux summary, proxy validity, and homeostasis stability
    And cluster-level robustness is summarized
```

```gherkin
@step08 @R5 @mechanism-prediction
Feature: mechanism clusters generate testable predictions
  Scenario: distinct mechanisms diverge under perturbation
    Given two accepted clusters with similar fitted function
    When bath coupling or stimulus duration is perturbed
    Then predicted K_o recovery or Vm recovery differs beyond tolerance
    And this difference is reported as a testable model prediction
```

### Notebook required

`analysis/08_perturbation_robustness.ipynb`

---

## Step 09 — Manuscript figures, units, and reviewer-facing outputs

**Pareto rank:** 9. Required for resubmission clarity after analyses are correct.

**Primary output:** `outputs/figures/main_*.png`, `outputs/figures/supplement_*.png`, `outputs/manuscript_tables/*.csv`.

### Scientific objectives

1. **Resolve R7 strongly:** replace cluttered traces with composite panels, bands, and summaries.
2. **Resolve R1-R6 presentation:** make every claim traceable to a table, diagnostic, and figure.

### Technical objectives

- Generate one main figure for definitions/identifiability/mechanisms.
- Generate one predictive validation figure.
- Generate one mechanism-decomposition figure.
- Generate supplementary figures for thresholds, parameter plausibility, assumption sensitivity, perturbations.
- Use region-faceted panels or matched DH/VH subpanels for ATF feature, fit, validation, and posterior predictive figures.
- Enforce units and axis labels.
- Generate a `figure_traceability_table.csv` mapping each figure/table to reviewer critique IDs, source notebook, and source output files.

### Approaches to compare

| Approach | Use | Selection rule |
|---|---|---|
| Spaghetti trace plots | Avoid except debug | Replace with bands and representative traces. |
| Composite grid panels | Main figure style | Use consistent axes and units. |
| Tables with pass/fail flags | Supplement | Use for traceability. |

### How to verify

- Figure-generation notebook runs without manual edits.
- Every figure has units, labels, legends, and panel titles.
- Figures that use ATF cells identify whether panels are DH, VH, or pooled sensitivity summaries.
- Tables can be regenerated from source outputs.
- Figure traceability table maps every reviewer-facing panel to at least one critique ID.

### Gherkin specifications

```gherkin
@step09 @R7 @figures
Feature: publication figures are generated reproducibly
  Scenario: every reviewer-facing figure has labeled units and traceable source data
    Given analysis output tables
    When the figure notebook runs
    Then every figure is written to outputs/figures
    And every axis has a non-empty label with units where applicable
    And every panel source table is recorded
    And ATF-derived panels explicitly state DH, VH, or pooled sensitivity scope
```

### Notebook required

`analysis/09_manuscript_figures.ipynb`

---

# Additional backlog from the technical suggestions

The following items are lower priority than the steps above but should be added if time allows.

| Backlog item | Related critiques | When to add |
|---|---|---|
| Full STRIKE-GOLDD or symbolic structural-identifiability report | R1 | Optional after Step 03; start with exact `d × pk` proof and only claim formal structural identifiability if implemented. |
| Bayesian or approximate posterior sampling | R1, R6 | After accepted-ensemble pipeline is stable. |
| Disk caching for simulations | Implementation | After post-fit and perturbation sweeps become slow. |
| Parallel postfit/perturbation workers | Implementation | After cache keys are validated. |
| Model variant with explicit ECS compartment | R3 | Only if proxy validity often fails. |
| Single-state intracellular model variant | R3 | Add after gating-family comparison. |
| Manuscript equation cleanup and notation table | R7 | In parallel with Step 09. |
| Region-aware Methods paragraph and limitations text | R2, R7 | In parallel with Step 02/06; state DH/VH counts, unpaired design, and no animal/slice IDs. |

# Minimal CI contract

Add these commands to the repository CI as soon as the first two steps are committed:

```bash
pytest -q tests/test_00_data_provenance_audit.py
pytest -q tests/test_01_postfit_sqlite_pipeline.py
```

As later steps are implemented, add one test file and one notebook smoke execution per step:

```bash
pytest -q tests/test_02_atf_thresholds.py
pytest -q tests/test_02_region_design_contract.py
pytest -q tests/test_03_combined_identifiability.py
pytest -q tests/test_04_mechanistic_decomposition.py
pytest -q tests/test_05_assumption_sensitivity.py
pytest -q tests/test_06_cell_six_sweep_validation.py
pytest -q tests/test_07_parameter_plausibility.py
pytest -q tests/test_08_perturbation_robustness.py
pytest -q tests/test_09_figures.py
```
