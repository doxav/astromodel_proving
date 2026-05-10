# Step 08 — Parameter plausibility and constrained reruns

## Purpose

Step 08 implements the reviewer-response parameter-interpretability layer requested in `reviewer_response_implementation_spec.md`.  It converts accepted Step 04–06 candidate ensembles into a cautious audit that separates broad biophysical plausibility, practical identifiability, effective-coordinate interpretability, and mechanism-claim persistence under plausible constraints.

The step is deliberately conservative: an accepted fit inside a numerical search bound is not automatically biologically interpretable, and an effective parameter may be reviewer-facing even when the raw factors that generated it remain structurally confounded.

## Reviewer objections addressed

| Reviewer objection | Step 08 response | Claim strength |
|---|---|---|
| R4: Vm-only fitting weakly constrains ionic dynamics and may yield non-physiological parameters. | Audit raw/effective coordinates against documented broad ranges, Step 03 identifiability statuses, held-out prediction, and mechanism labels. | Strong for parameter interpretation guardrails. |
| R1: degeneracy is not separated from non-identifiability/sloppiness. | Assign `within_range`, `out_of_range`, `identifiable`, `weakly_identified`, and `effective_only` statuses before any mechanism claim can be called physiological. | Partial; final synthesis remains Step 09. |
| R5: mechanisms may not correspond to interpretable biology. | Compare unconstrained and plausibility-constrained candidates; downgrade claims that depend on implausible or weakly identified coordinates. | Partial; mechanisms remain candidate regimes until final reviewer-facing synthesis. |

## Inputs

Step 08 uses repository-local outputs only:

- Step 04 accepted cell-specific ensembles loaded through the Step 06 input contract.
- Step 05 mechanism labels for mechanism cluster and dominant mechanism annotations.
- Step 06 held-out prediction aggregates preserved on candidates.
- Step 03 effective-parameter and profile/FIM outputs when available.

No Google Drive paths, external downloads, or hidden notebook state are allowed.

## Parameter range policy

The audit uses broad plausibility ranges intended as reviewer-facing guardrails rather than narrow priors.  Each range row contains a `range_source` string and `range_basis` so the manuscript can distinguish literature/modeling guardrails from hard physiological facts.  Step 08 must include at least:

- raw coordinates: `gki`, `eps`, `gl_a`, `zth`, `zs`;
- effective coordinates: `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`.

Raw factors that are known to be structurally confounded are never upgraded to physiological mechanism claims solely because they are inside range.

## Interpretation statuses

Every candidate-parameter row must receive:

1. `plausibility_status`: `within_range`, `out_of_range`, or `missing_value`;
2. `identifiability_status`: `identifiable`, `weakly_identified`, `effective_only`, or `not_profiled`;
3. `physiologically_interpretable`: `True` only when the parameter is in range and identifiable/effective-coordinate evidence supports interpretation;
4. `interpretation_guardrail`: a human-readable caution explaining why a parameter is or is not reviewer-facing.

A parameter inside bounds remains non-interpretable if Step 03 labels it flat, broad, boundary-hit, weak, structurally confounded, or only interpretable through an effective combination.

## Constrained rerun comparison

Step 08 performs a targeted constrained-rerun screen by projecting high-priority candidate parameters into the broad plausibility ranges and rescoring the projected candidate under the same candidate/current/time-grid contract.  This is a lightweight constrained screen, not a replacement for the Step 04 optimizer.  Outputs must compare:

- unconstrained versus constrained fit-quality aggregates for every canonical current in `VALID_CURRENTS`;
- held-out prediction pass fraction;
- mechanism cluster/label persistence based on hidden-current flux fractions, not labels alone;
- maximum gap/Kir/leak flux-fraction drift under constraints;
- perturbation/robustness proxies when already available;
- which parameters changed due to the plausibility projection.

Claims that disappear or degrade materially under reasonable broad constraints must be downgraded.

## Outputs

All primary outputs are written under `outputs/parameter_plausibility/`.

| File | Purpose |
|---|---|
| `parameter_range_audit.csv` | Candidate × parameter plausibility and identifiability statuses. |
| `effective_parameter_plausibility.csv` | Effective-coordinate-only reviewer-facing interpretation table. |
| `constrained_rerun_comparison.csv` | Unconstrained versus constrained candidate comparison. |
| `interpretability_status.csv` | Candidate-level claim status by region, condition, and mechanism cluster. |
| `performance_benchmark.csv` | Runtime preset comparison and tuning recommendation. |
| `analysis_summary.json` | Machine-readable configuration, counts, and claim-scope text. |

## Scientific contract

1. **No silent interpretation:** every audited parameter has explicit plausibility, identifiability, and interpretability fields.
2. **Effective parameters are separated from raw parameters:** effective-coordinate results are exported separately and are the preferred reviewer-facing coordinates for structurally confounded terms.
3. **Inside range is insufficient:** a parameter cannot become physiologically interpretable if identifiability evidence is weak or absent.
4. **Constrained comparisons are auditable:** every constrained row states the current sweep, whether values changed, which values changed, hidden-current flux summaries, and whether the same mechanism/prediction conclusion persists.
5. **No final degeneracy language:** Step 08 may authorize parameter-interpretability claims, but final biological degeneracy claims remain pending Step 09 synthesis.

## Gherkin specifications

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
@step08 @R4 @effective-parameters
Feature: effective coordinates are reported separately from raw coordinates
  Scenario: structurally confounded coordinates are interpreted through effective combinations
    Given raw and effective accepted-candidate parameters
    When Step 08 writes plausibility outputs
    Then effective parameters are exported to `effective_parameter_plausibility.csv`
    And raw structurally confounded factors are not used as standalone physiological mechanisms
```

```gherkin
@step08 @R4 @constrained-rerun
Feature: constrained inference tests whether claims depend on implausible parameters
  Scenario: constrained and unconstrained accepted ensembles are compared
    Given high-priority cells or mechanism clusters
    When constrained screens are performed
    Then fit quality, prediction, and mechanism metrics are compared
    And claims that disappear under reasonable constraints are downgraded
```

## Tests required

### Bootstrap

- Plausibility-range definitions contain all required raw and effective parameters with finite lower/upper bounds.
- Step 08 input loading preserves `file_id`, `region`, `condition`, `candidate_id`, held-out metrics, and mechanism labels.
- Parameter audit rows assign plausibility, identifiability, and physiologically-interpretable statuses without null status fields.

### Acceptance

- Running Step 08 writes all required CSV/JSON outputs.
- `effective_parameter_plausibility.csv` contains only effective coordinates and never raw-coordinate rows.
- `interpretability_status.csv` states that final biological degeneracy claims remain disabled after Step 08.

### Integration

- Candidate-level interpretability summaries are coherent with parameter-level audit rows by `file_id`, `region`, `condition`, and `candidate_id`.
- Constrained-rerun comparison contains one row per requested current, changed-parameter provenance, flux-based mechanism-change fields, and conservative claim-persistence statuses.
- `analysis/08_parameter_plausibility_and_constrained_reruns.ipynb` executes from the repository root and saves an auditable executed copy.

### Performance

- A one-candidate coarse Step 08 run completes inside a practical runtime budget.
- `compare_step08_runtime_presets` records coarse/default elapsed times and a tuning recommendation.

## Notebook contract

`analysis/08_parameter_plausibility_and_constrained_reruns.ipynb` must include an Open-in-Colab badge at the top, run from the repository root, write machine-readable outputs, and demonstrate:

1. accepted ensemble inventory;
2. parameter plausibility/identifiability audit table and figure;
3. effective-parameter plausibility table;
4. constrained-rerun comparison table and figure;
5. candidate-level interpretability status table;
6. conservative claim-scope text explaining which parameter/mechanism statements are reviewer-facing versus downgraded.
