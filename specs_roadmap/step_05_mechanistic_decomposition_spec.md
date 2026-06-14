# Step 05 — Mechanistic decomposition of accepted cell ensembles

## Purpose and reviewer-response scope

Step 05 converts Step 04 cell-specific accepted ensembles into mechanism-level evidence. It is the first reviewer-response step that can address whether multiple voltage-compatible fits correspond to different buffering balances rather than simply to structural non-identifiability, sloppiness, or continuous parameter compensation.

This step is intentionally conservative:

- it may identify **candidate mechanism regimes pending validation**;
- it may identify **compensation manifolds** when accepted candidates remain connected through effective-parameter interpolation;
- it must not use the stronger phrase **candidate degenerate regimes** until Step 06 predictive or perturbation robustness checks support the claim.

## Inputs

Primary input:

- `outputs/cell_fits/accepted_cell_ensembles.csv` from the model-aligned Step 04 path.

Supported fallback input for the newer multi-sweep runner:

- `outputs/step04_cell_specific_multisweep/accepted_candidates.csv`.

The loader must preserve the biological and fitting contract fields from Step 04:

- `file_id`
- `region`
- `condition`
- `candidate_id`
- acceptance/status fields (`accepted_all6`, `cell_reviewer_facing`, or `accepted` depending on Step 04 source)
- effective coordinates: `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`
- conductance/gating/nuisance coordinates: `gki` or `g_kir`, `gl_a`, `zth`, `zs`, `eps`, optional `k_bath_gain`, and `switching_function`.

## Outputs

All reviewer-facing outputs are written under `outputs/mechanisms/`:

| Output | Required contents |
|---|---|
| `accepted_fit_mechanisms.csv` | One row per `candidate × sweep` with simulation status, hidden-current integrals/peaks, K_o summaries, proxy-validity metrics, and retained Step 04 metadata. |
| `mechanism_clusters.csv` | One row per accepted candidate with aggregate mechanism metrics, effective coordinates, cluster label, cluster stability, and claim-scope status. |
| `representatives.csv` | Maximin mechanism-diverse representatives that retain accepted Step 04 functional status. |
| `region_mechanism_enrichment.csv` | Counts/fractions by `region × condition × mechanism_cluster`, with small-stratum flags. |
| `geometry_classification.csv` | Pairwise/interpolation diagnostics classifying accepted-set geometry as `compensation_manifold`, `separated_modes_pending_validation`, or `insufficient_evidence`. |
| `bootstrap_cluster_stability.csv` | Bootstrap cell-level resampling stability statistics. |
| `claim_scope_table.csv` | Explicit language allowed for manuscript claims before Step 06. |
| `analysis_summary.json` | Configuration, input source, row counts, and conservative headline status. |

## Scientific objectives

1. Compute hidden-current and K_o summaries for every accepted cell candidate across the six ordered pump-current sweeps.
2. Decompose candidate fits into Kir, gap-junction, and leak buffering fractions and ratios.
3. Quantify whether `ΔK_a,t` is a reliable proxy for `K_o` in each accepted candidate.
4. Cluster candidates in effective/mechanism space rather than raw parameter space alone.
5. Test whether clusters are stable under cell-level bootstrap resampling.
6. Test whether representative cluster centers are connected by accepted effective-parameter interpolations.
7. Select same-function but mechanism-diverse representatives for Step 06 perturbation and predictive validation.
8. Report cluster occupancy separately by DH/VH region and condition.

## Technical design

### Parameter reconstruction

Step 04 stores effective coordinates, not all raw ODE parameters. Step 05 reconstructs simulator-ready flat parameters using the canonical effective-coordinate identities:

- `P_gap_eff = d × pk`; Step 05 sets `d = 1` and `pk = P_gap_eff` for simulation because only the product is identifiable in this model.
- `gamma_t_eff = gt × Sig_a / (w_a × F)`; Step 05 uses canonical `w_a = 2000`, `Sig_a = 1600`, and `F = 96485` to reconstruct `gt`.
- `gamma_s_eff = gs × Sig_a / (w_a × F)` using the same constants.
- `volume_ratio_wa_wo = w_a / wo`; Step 05 reconstructs `wo = w_a / volume_ratio_wa_wo`.

This reconstruction must be documented as an effective-parameter simulation convention, not evidence that `d`, `pk`, `gt`, `gs`, or `wo` are individually identifiable.

### Hidden-output simulation

For each accepted candidate and current in `{50, 75, 100, 125, 150, 175}` nA, Step 05 calls the canonical model hidden-output API with a configurable coarse time grid. Failed simulations produce rows with `simulation_status = failed` and `failure_reason`; they must not be silently dropped.

### Flux and proxy summaries

For each candidate-sweep simulation, compute:

- `I_Kir_integral`, `I_Kir_peak_abs`
- `I_kgap_integral`, `I_kgap_peak_abs`
- `I_leak_integral`
- `gap_to_kir_integral_ratio`
- `gap_fraction`, `kir_fraction`, `leak_fraction`
- `K_o_peak`, `K_o_final`, `K_o_recovery_error`
- proxy-validity metrics comparing `DK_a` against `K_o`: Pearson r, Spearman r, scaled RMSE, and class.

### Clustering and stability

Candidate-level clustering uses standardized mechanism/effective columns:

- `log10_P_gap_eff`
- `log10_gamma_t_eff`
- `log10_gamma_s_eff`
- `log10_volume_ratio_wa_wo`
- mean `gap_fraction`
- mean `kir_fraction`
- mean `leak_fraction`
- log mean `gap_to_kir_integral_ratio`
- mean `K_o_recovery_error`

The default cluster count is bounded by the number of available cells/candidates and is recorded in the summary. Bootstrap stability resamples cells, reclusters/resummarizes candidate assignments against the original centers, and reports a mean adjusted-Rand-like pairwise coassignment score. Small ensembles must be labeled `insufficient_evidence` rather than overinterpreted.

### Geometry classification

Representative cluster centers are interpolated in log/effective coordinates. If interpolated points remain within the observed accepted functional envelope and are simulatable, the pair is classified as `compensation_manifold`. If interpolation passes through poor functional regions, it may be classified as `separated_modes_pending_validation`. If there are too few clusters/cells, classify as `insufficient_evidence`.

### Representative selection

Representatives are selected by maximin distance in standardized mechanism space after filtering to Step 04 accepted/reviewer-facing candidates. Representatives must preserve functional status and expose their `representative_rank`, `selection_reason`, and `claim_scope`.

## Gherkin specifications

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
    And their mechanism distances exceed the configured diversity threshold when enough candidates exist
```

```gherkin
@step05 @R1 @geometry
Feature: accepted-fit geometry separates compensation from modes
  Scenario: interpolated parameters test whether two clusters are disconnected
    Given two candidate accepted clusters in effective-parameter space
    When representative centers are interpolated in log/effective coordinates
    Then each interpolation diagnostic is reported with a simulation status
    And the pair is classified as compensation_manifold when interpolations remain accepted-like
    And separated modes remain pending_validation until Step 06 robustness checks are passed
```

```gherkin
@step05 @R2 @R5 @region-mechanisms
Feature: mechanism regimes are summarized by brain region
  Scenario: mechanism-cluster occupancy is reported separately for DH and VH
    Given accepted cell-level ensembles with mechanism-cluster labels
    When mechanism enrichment is summarized
    Then counts and fractions are reported by region, condition, and cluster
    And DH/VH differences are reported as population-level cell effects
    And no phenotype claim is made from region enrichment alone
```

## Tests required

- Bootstrap tests:
  - Step 04 accepted-ensemble loading preserves `file_id`, `region`, `condition`, and effective-coordinate columns.
  - Simulator parameter reconstruction preserves effective coordinates.
  - Flux summaries contain hidden-current, K_o, and proxy-validity columns.
- Acceptance tests:
  - Running Step 05 writes all required CSV/JSON outputs.
  - Mechanism outputs never drop `region` or `condition`.
  - Representatives are Step 04 accepted/reviewer-facing candidates.
  - Claim-scope statuses remain conservative before Step 06.
- Integration tests:
  - The notebook `analysis/05_mechanistic_decomposition.ipynb` executes from repository root.
  - Region/condition enrichment and geometry classification are coherent with cluster outputs.
- Performance tests:
  - A small Step 05 run over the cached Step 04 demo ensemble finishes within a practical runtime budget.
  - A tuning comparison records elapsed time and status for coarse and default simulation grids.

## Notebook contract

`analysis/05_mechanistic_decomposition.ipynb` must include an Open-in-Colab badge at the top and must run without Google Drive dependencies. It must demonstrate:

1. accepted ensemble inventory;
2. candidate-sweep flux decomposition table;
3. mechanism-space clustering plot;
4. flux decomposition visual for representative candidates;
5. compensation-vs-separated-mode interpolation diagnostics;
6. bootstrap stability table;
7. DH/VH and condition occupancy table;
8. explicit claim-scope table stating that predictive/perturbation validation is pending Step 06.
