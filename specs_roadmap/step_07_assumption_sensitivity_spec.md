# Step 07 — Assumption sensitivity: gating, proxy, and compartment split

## Purpose

Step 07 implements the reviewer-response assumption-sensitivity layer requested in `reviewer_response_implementation_spec.md`.  It asks whether Step 04–06 conclusions are stable when the core modeling assumptions are changed, without silently upgrading weak evidence into a biological-degeneracy claim.

## Reviewer objections addressed

| Reviewer objection | Step 07 response | Claim strength |
|---|---|---|
| R3: sigmoid gating, intracellular-K proxy, and local/syncytial split are under-justified. | Compare six gating families, quantify `ΔK_a,t` versus simulated `K_o`, and compare two-state versus one-state intracellular proxy behavior. | Strong for assumption audit; conservative for biology. |
| R6: predictions may be brittle beyond the fitted traces. | Score assumption variants under identical candidate/current/time-grid contracts. | Partial; Step 06 remains primary predictive validation. |
| R1/R5: mechanism regimes may be artifacts of parameterization. | Mechanism persistence is reported only when configured assumption variants pass the same contract. | Partial; final claims require later plausibility/statistical steps. |

## Inputs

Step 07 uses existing repository outputs and model APIs:

- Step 04 accepted cell-specific ensemble rows, loaded through the Step 06 input contract.
- Step 05 mechanism labels merged on `file_id`, `region`, `condition`, and `candidate_id`.
- Step 06 held-out metrics exposed on candidate rows.
- The canonical simulator in `src.astro_model` with the same current protocols and time grid for each compared family.

No Google Drive paths, external downloads, or hidden notebook state are allowed.

## Compared assumptions

### Gating-family comparison

The configured model-family panel is:

1. `sigmoid` baseline;
2. `tanh`;
3. `hill`;
4. `soft_threshold`;
5. `hard_threshold`;
6. `double_sigmoid`.

Each family is evaluated on identical candidates, currents, time grid, candidate-level Step 04/06 metrics, and mechanism labels.  The primary contract identifier is `step07_same_candidates_currents_timegrid_loss_v1`.

### Intracellular K as ECS proxy

For each accepted candidate/current, Step 07 simulates hidden outputs and compares the local intracellular potassium proxy `ΔK_a,t` against simulated extracellular potassium `K_o` using:

- Pearson correlation;
- Spearman correlation;
- scaled linear RMSE;
- best lag in samples over a small lag window;
- an explicit `proxy_validity_status` and `explicit_ecs_variant_required` flag.

### Local/syncytial compartment split

Step 07 compares the two-state local proxy `ΔK_a,t` with a one-state aggregate sensitivity proxy `ΔK_a,t + K_a,s`.  This is a scored sensitivity check, not a replacement ODE fit.  Rows report whether mechanism structure persists or whether split dependence remains unresolved.

## Outputs

All primary outputs are written under `outputs/assumption_sensitivity/`.

| File | Purpose |
|---|---|
| `model_comparison.csv` | Family-level fit/prediction/mechanism stability summary. |
| `gating_family_comparison.csv` | Candidate/current/family rows using identical contract definitions. |
| `proxy_validity_by_ensemble.csv` | Proxy metrics by region, condition, sweep, current, candidate, and mechanism cluster. |
| `compartment_split_sensitivity.csv` | One-state versus two-state proxy sensitivity rows. |
| `claim_scope_table.csv` | Conservative assumption-level claim scope; final degeneracy claims remain disabled. |
| `analysis_summary.json` | Machine-readable configuration, counts, and claim-scope text. |

## Scientific contract

1. **Fair comparison:** gating families must use the same candidates, currents, time grid, identity columns, and status columns.
2. **No silent failures:** simulation failures must produce rows with `simulation_status = failed` and a `failure_reason`.
3. **Proxy limitations are explicit:** if proxy correlation/RMSE criteria fail, `explicit_ecs_variant_required` is `True`.
4. **Split sensitivity remains conservative:** a one-state aggregate is a sensitivity score only; it does not replace cell-specific two-state inference.
5. **No final degeneracy language:** `final_degeneracy_claim_allowed_after_step07` remains `False` because parameter plausibility and final synthesis steps are still pending.

## Gherkin specifications

```gherkin
@step07 @R3 @gating-sensitivity
Feature: gating-family conclusions are compared under identical contracts
  Scenario: different gating forms are evaluated fairly
    Given accepted-fit contracts and data splits
    When each configured gating family is scored
    Then the output reports fit, prediction, and mechanism metrics with identical definitions
    And mechanism claims are marked robust only if they persist across configured families
```

```gherkin
@step07 @R3 @proxy-validity
Feature: intracellular K proxy validity is quantified
  Scenario: ΔK_a,t is compared with K_o
    Given accepted ensemble simulations with hidden states
    When proxy validity metrics are computed
    Then Pearson correlation, Spearman correlation, scaled RMSE, and lag are reported
    And failed proxy regimes require an explicit ECS variant or additional data
```

```gherkin
@step07 @R3 @compartment-split
Feature: local/syncytial split sensitivity is tested
  Scenario: one-state and two-state intracellular formulations are compared
    Given the same empirical candidates and acceptance contract
    When split and aggregate proxy variants are evaluated
    Then output reports whether accepted mechanism structure persists
```

## Tests required

### Bootstrap

- Step 07 input loading preserves `file_id`, `region`, `condition`, `candidate_id`, and mechanism labels.
- All configured gating families produce explicit simulation statuses under the same contract ID.
- Proxy metrics are finite for successful simulations or have auditable failure fields.

### Acceptance

- Running Step 07 writes all required CSV/JSON outputs.
- `model_comparison.csv` includes fit quality, held-out pass, same-contract, and mechanism-stability columns.
- `claim_scope_table.csv` never enables final degeneracy claims.

### Integration

- `analysis/07_assumption_sensitivity.ipynb` executes from the repository root.
- Gating, proxy, and split tables are cross-table coherent by region/condition/candidate.

### Performance

- A one-candidate coarse Step 07 run completes inside a practical runtime budget.
- `compare_step07_runtime_presets` records coarse/default elapsed times and a tuning recommendation.

## Notebook contract

`analysis/07_assumption_sensitivity.ipynb` must include an Open-in-Colab badge at the top, run from the repository root, write machine-readable outputs, and demonstrate:

1. accepted ensemble inventory;
2. gating-family comparison table and figure;
3. model-comparison summary;
4. proxy-validity table and figure;
5. compartment-split sensitivity table and figure;
6. conservative claim-scope table explaining which conclusions are robust versus model-dependent.
