# Step 03 — Combined identifiability screen: soft structural inspection, profile likelihood, and FIM

## Purpose and reviewer-response scope

Step 03 implements the reviewer-response plan's Pareto-rank-3 objective: separate structural non-separability, practical non-identifiability, sloppiness, and only then candidate biological degeneracy. The step is intentionally named a **structural-identifiability-informed practical screen**. It is not a full STRIKE-GOLDD proof unless a future symbolic backend is added and documented.

Primary reviewer critiques addressed:

- **R1:** distinguish degeneracy from structural non-identifiability, practical non-identifiability, sloppiness, and compensation.
- **R4:** report which raw or effective coordinates Vm can constrain.
- **R7:** provide clean tables/figures and claim-boundary language for manuscript revision.

## Inputs

- Representative accepted SQLite fits from `data/1_Initial_xp_fit/`:
  - `CONTROL_75nA.db`
  - `MFA_100nA.db`
  - `BARIUM_100nA.db`
- The refactored local model in `src/astro_model.py`.
- Step 01 effective-parameter helpers in `src/postfit_sqlite.py`.

The Step 03 diagnostics are centered on representative single-current fits because Step 06 will replace this with six-sweep cell-specific accepted ensembles. Step 03 outputs must therefore avoid phenotype or degeneracy claims.

## Required outputs

Write all tables under `outputs/identifiability/`:

| File | Required content |
|---|---|
| `effective_parameter_map.csv` | Raw/effective coordinate map, structural classification, expressions, and reviewer interpretation. |
| `structural_invariance_diagnostics.csv` | Exact or near-exact product-invariance checks, starting with `d × pk`. |
| `profile_likelihoods.csv` | Profile grid values, fixed values, best loss after nuisance refit, and profile class. |
| `profile_summary.csv` | One row per profiled coordinate with min/max loss and profile interpretation. |
| `fim_spectrum.csv` | FIM eigenvalue spectrum, log eigenvalues, stiff/sloppy labels, dominant parameters. |
| `fim_mode_loadings.csv` | Parameter loadings for every FIM mode and representative center. |
| `analysis_summary.csv` | Runtime, representative-center count, numerical settings, and explicit claim boundary. |

## Effective-parameter contract

The following effective coordinates are primary reporting coordinates:

- `P_gap_eff = d * pk`
- `gamma_t_eff = gt * Sig_a / (w_a * F)`
- `gamma_s_eff = gs * Sig_a / (w_a * F)`
- `volume_ratio_wa_wo = w_a / wo`

The map must classify:

- `d` and `pk` as `effective_combination_member` of `P_gap_eff`.
- `gt` and `gs` as `effective_combination_member` of their gamma effective coordinates.
- `wo` and fixed `w_a` as members of `volume_ratio_wa_wo`.
- the four effective coordinates as `primary_interpretable`.
- raw candidates such as `gki`, `gl_a`, `ca`, `eps`, and `K_bath_value_middle` as direct candidates only if profile/FIM diagnostics support local interpretability.

## Structural-inspection diagnostics

The required exact-invariance demonstration is the gap-current product:

1. Choose one representative state and time in the stimulation window.
2. Create paired parameter dictionaries that scale `d` and inversely scale `pk`.
3. Confirm that `P_gap_eff`, `I_kgap`, and the full RHS vector are unchanged within floating-point tolerance.
4. Save the result to `structural_invariance_diagnostics.csv`.

This demonstrates an equation-level structural confounding but must not be described as a complete structural-identifiability proof.

## Profile-likelihood diagnostics

For each key effective coordinate:

- fix a grid of multiplicative values around the representative center;
- perform a lightweight nuisance refit rather than directly comparing unadjusted traces;
- save every grid point and a shape classification.

The implemented nuisance refit is least-squares affine Vm calibration (`scale × Vm + offset`) so tests and notebooks remain fast and deterministic. This is a local practical-identifiability screen; future steps may replace it with full nuisance-parameter optimization.

Allowed profile classes:

- `clear_valley`
- `broad_valley`
- `flat_unbounded`
- `boundary_hit`

Interpretation rules:

- `clear_valley`: locally identifiable around the representative center.
- `broad_valley`: weakly constrained and not a direct molecular estimate.
- `flat_unbounded`: practical non-identifiability in this diagnostic.
- `boundary_hit`: the tested grid is insufficient or the optimum lies outside the local range.

## FIM/sloppiness diagnostics

The FIM must be computed after effective reparameterization using finite differences in log-coordinate space. The default coordinate set is:

- `P_gap_eff`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`
- `gki`
- `gl_a`
- `ca`
- `eps`
- `K_bath_value_middle`

Requirements:

- evaluate at the representative accepted centers;
- symmetrize and ridge-stabilize the FIM;
- write finite eigenvalues and parameter loadings;
- label modes as `stiff` or `sloppy` by relative eigenvalue;
- state that sloppy directions are not equivalent to biological degeneracy.

## Tests to add

### Bootstrap tests

- Required input SQLite files exist and are readable.
- Effective-parameter map contains all primary effective coordinates.
- `d`/`pk` are mapped to `P_gap_eff`.
- The exact product-invariance check gives near-zero differences.

### Acceptance tests

- Running the Step 03 pipeline writes all required CSV files.
- Profiles exist for all primary effective coordinates and use only allowed classes.
- FIM eigenvalues are finite, non-negative after stabilization, and include stiff/sloppy annotations.
- Mode-loading rows contain both raw and effective coordinates.

### Integration tests

- The notebook `analysis/03_combined_identifiability_profiles_fim.ipynb` executes from the repository root.
- Notebook execution produces the required Step 03 outputs.

### Performance/tuning tests

- Compare two FIM finite-difference step sizes on the same representative center.
- Save/verify a tuning table with elapsed time, eigenvalue rank, and condition-number estimate.
- The default configuration must complete quickly enough for CI-scale smoke execution.

## Notebook requirements

Create `analysis/03_combined_identifiability_profiles_fim.ipynb` with:

1. an Open in Colab badge at the top;
2. local/Colab setup that can access `src/`, `data/`, and relative outputs;
3. execution of `run_step03_identifiability_screen`;
4. display of `effective_parameter_map.csv`;
5. display of exact `d × pk` invariance diagnostics;
6. at least one profile-likelihood plot per primary effective coordinate;
7. FIM eigenvalue spectrum plot;
8. stiff/sloppy mode-loading table;
9. reviewer-facing claim-boundary text explaining that this is a structural-inspection screen, not a full STRIKE-GOLDD proof.

## Completion criteria

Step 03 is complete when:

- all Step 03 bootstrap, acceptance, integration, and performance tests pass;
- the Step 03 notebook has been executed and committed;
- all outputs listed above are present under `outputs/identifiability/`;
- no notebook or output text claims biological degeneracy from flat profiles or sloppy FIM modes.
