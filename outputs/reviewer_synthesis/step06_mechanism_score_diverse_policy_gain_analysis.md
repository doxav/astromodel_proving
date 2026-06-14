# Step 06 Mechanism-Score Diverse Policy Gain Analysis

## Objective

Test the minimal robust way to use Step 05 continuous mechanism scores in Step 06 candidate selection, without changing Step 04 fitting or overloading Step 05 phenotype labels.

## Strategies Considered

| Strategy | Code/notebook impact | Reviewer value | Decision |
| --- | --- | --- | --- |
| Keep `best_per_cell` only | No change | Conservative canonical screen, but underuses accepted alternatives | Keep as canonical default. |
| Use `top_k_per_cell` | Already present | Tests whether more accepted candidates improve support | Keep as sensitivity. |
| Use `mechanism_diverse_per_cell` | Already present | Tests one representative per Step 05 mechanism cluster per cell | Keep as best current expanded-scope sensitivity. |
| Add `mechanism_score_diverse_per_cell` | Small Step 06 policy addition | Directly tests log effective coordinates plus Step 05 continuous scores | Added as sensitivity, not default. |
| Add hard phenotype penalties to Step 04 Optuna | Large Step 04/05 coupling | Risky because tags are provisional and some are threshold-sensitive | Rejected. |

## Implemented Design

The new Step 06 policy is `mechanism_score_diverse_per_cell`.

It keeps the best-quality accepted candidate as the seed for each cell, then selects additional candidates by maximin distance over:

- log10 `P_gap_eff`
- log10 `gamma_t_eff`
- log10 `gamma_s_eff`
- log10 `volume_ratio_wa_wo`
- `dKs_activation_score_mean`
- `long_range_distribution_fraction_mean`
- `voltage_coupling_score_mean`
- `kir_current_score_mean`
- log10 recruited-surface score, computed as `gamma_s_eff * dKs_activation_score_mean`

Stable phenotype labels are used only as a tie-breaker after continuous-score distance. The policy does not change the canonical Step 06 default.

## Regenerated Results

Evidence files:

- `outputs/reviewer_synthesis/step06_candidate_scope_sensitivity_summary.csv`
- `outputs/reviewer_synthesis/step06_candidate_scope_sensitivity_robustness.csv`
- `outputs/reviewer_synthesis/step06_candidate_scope_selection_inventory.csv`
- `outputs/reviewer_synthesis/step06_mechanism_score_diverse_selection_summary.csv`

| Candidate policy | Candidates | Robustness rows | Predictive-supported rows | Prediction-limited rows | Mean biological score | Mean perturbation robust fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_per_cell` | 35 | 8 | 3 | 5 | 0.617 | 0.413 |
| `top_k_per_cell` | 99 | 13 | 8 | 5 | 0.648 | 0.523 |
| `mechanism_diverse_per_cell` | 84 | 15 | 10 | 5 | 0.665 | 0.557 |
| `mechanism_score_diverse_per_cell` | 99 | 14 | 7 | 7 | 0.644 | 0.505 |

The new policy did select broader mechanism-score/phenotype coverage. For example, it selected 99 candidates across 35 cells and covered 4-5 phenotype labels in the main treatment strata. However, this broader continuous-score novelty did not improve Step 06 validation compared with the existing mechanism-cluster-diverse policy.

## Reviewer Challenge

The reviewer-facing question is not whether the policy can find different-looking candidates. It can. The relevant question is whether those candidates improve predictive and perturbation support without weakening claim discipline.

On that stronger criterion, `mechanism_score_diverse_per_cell` is not the best design:

- It has fewer predictive-supported rows than `mechanism_diverse_per_cell` (`7/14` versus `10/15`).
- It has lower mean perturbation robustness (`0.505` versus `0.557`).
- It increases prediction-limited rows (`7` versus `5`).
- It does not change the final claim gate because assumption and parameter plausibility checks still constrain degeneracy wording.

This is a useful negative control: continuous Step 05 scores are biologically meaningful diagnostics, but maximizing novelty in those scores can select less robust perturbation candidates.

## Claim Progression

Claim progression is limited but real:

- **R6:** strengthened as a sensitivity audit, because Step 06 now explicitly tests whether continuous mechanism-score novelty improves predictive/perturbation support.
- **R5:** modestly strengthened, because the analysis shows mechanism diversity should be selected by stable Step 05 cluster representatives rather than raw phenotype novelty.
- **Final biological degeneracy:** no upgrade. The new policy does not outperform the existing mechanism-diverse policy and does not satisfy the later Step 07/08/09 gates.

## Recommended Notebook Position

Keep Step 06 canonical output as `best_per_cell`.

Keep sensitivity ordering:

1. `best_per_cell`
2. `top_k_per_cell`
3. `mechanism_diverse_per_cell`
4. `mechanism_score_diverse_per_cell`

Use `mechanism_diverse_per_cell` as the strongest expanded-scope result for reviewer discussion. Use `mechanism_score_diverse_per_cell` as a negative-control sensitivity showing that more mechanistic novelty is not automatically better and that claims remain conservative.
