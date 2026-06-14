# Step 04 Optimizer Diversity Comparison

## Question

Can Step 04 prove useful degeneracy by finding 2+ biologically/contract-acceptable fits per cell that differ in effective parameters, and is the limiting factor simply that Optuna/SciPy was not run long enough?

## Definitions Used

Effective parameters were taken from the repository's shared parameter-space definition and cross-checked against Step 04 and Step 08:

- `P_gap_eff`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`

Raw/nuisance coordinates such as `gki`, `eps`, `gl_a`, `zth`, and `zs` were excluded from the primary degeneracy-distance metric because raw-only variation does not prove distinguishable effective mechanisms.

Accepted candidates were required to satisfy the Step 04 all-six-sweep contract:

- `mean_trace_rmse_mV <= 18`
- `mean_weighted_pass_fraction >= 0.30`
- `n_failures == 0`

Reviewer-facing cell support additionally uses the Step 04 held-out-current gate. Effective diversity was scored by greedy clustering in log10 effective-parameter space, with the main threshold at Euclidean distance `0.5` and sensitivity checks at `0.3`, `0.7`, and `1.0`.

## Evidence Files

- Full optimizer summary: `outputs/reviewer_synthesis/step04_optimizer_historical_diversity_summary.csv`
- Per-cell diversity: `outputs/reviewer_synthesis/step04_optimizer_diversity_by_cell.csv`
- Per-stratum diversity: `outputs/reviewer_synthesis/step04_optimizer_diversity_by_stratum.csv`
- Controlled missing-method probes: `outputs/reviewer_synthesis/step04_optimizer_method_probe_summary.csv`
- Parameter plausibility failures: `outputs/reviewer_synthesis/step04_accepted_parameter_plausibility_failures.csv`

## Method Comparison

| Method/evidence | Scope | Holdout? | Accepted candidates | Cells with 2+ effective clusters at 0.5 | Reviewer-facing cells | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Hybrid TPE, `acceptance_margin`, current canonical `outputs/cell_fits` | 37 cells | yes | 2110 | 33 | 33 | Best current evidence. Generalizes across all non-VH-control reviewer-facing strata. |
| Hybrid TPE, `acceptance_margin`, pre-targeted full run | 37 cells | yes | 2098 | 28 | 26 | Same method before targeted VH MFA repair; shows method was close but under-searched VH MFA. |
| Hybrid TPE, metric-scalar 2-per-group | 12 cells | yes | 21 | 7 | 7 | Useful but lower scope; not a replacement for canonical. |
| Hybrid TPE objective variants, 12-cell no-holdout screens | 12 cells | no | 18-21 | 5-6 | 0 | Good exploratory seed generators, but not reviewer-facing without heldout. |
| Controlled Optuna scalar random, 6-cell probe | 6 cells | no | 68 | 4 | 0 | Finds effective alternatives but no reviewer-facing validation and no advantage over hybrid. |
| Controlled Optuna multi/NSGA-II, 6-cell probe | 6 cells | no | 18 | 4 | 0 | Works, but did not outperform simpler scalar/random search in this probe. |
| Least-squares-only smoke/perf runs | 1-2 cells | yes | 1-2 | 0 | 1-2 | Baseline fit finder, not a diversity finder. |

## Current Canonical Step 04 Diversity

Current canonical Step 04 has 37 fitted cells, 35 cells with at least one all-six accepted candidate, and 33 reviewer-facing cells. Among accepted effective-plausible candidates, 33 cells have at least two distinct effective-parameter clusters at the main 0.5 log-distance threshold. The same count remains 33 at thresholds 0.3 and 0.7, so this result is not threshold-fragile.

By stratum:

| Region | Condition | Accepted cells represented | Cells with 2+ effective clusters | Median effective clusters | Main limitation |
| --- | --- | ---: | ---: | ---: | --- |
| DH | CONTROL | 7 | 5 | 2.0 | Sparse but mostly reviewer-facing. |
| DH | MFA | 6 | 6 | 62.0 | Strong effective diversity. |
| DH | MFA_BA | 6 | 6 | 66.5 | Strong effective diversity. |
| VH | CONTROL | 2 accepted, 0 reviewer-facing | 2 | 2.5 | Heldout/feature-contract mismatch, not lack of effective alternatives. |
| VH | MFA | 7 | 7 | 3.0 | Rescued by targeted high-budget hybrid reruns. |
| VH | MFA_BA | 7 | 7 | 57.0 | Strong effective diversity. |

## Biological/Plausibility Caveat

If the audit requires all raw plus effective parameters to be inside the broad Step 08 ranges, diversity collapses: only 5 cells retain 2+ all-audited-plausible effective clusters. This is not primarily an optimizer-count issue. It is driven by raw/nuisance guardrails, especially `zth`, plus `gki`, `gl_a`, and `zs`.

That distinction matters:

- Effective-space multiplicity is already present in the canonical accepted pool.
- Raw-parameter physiological interpretability is still weak and belongs to Step 08/constrained rerun logic.
- Counting raw-only variation would overstate degeneracy.

## Conclusion

More optimization was a real local root cause for the old VH MFA gap, and the targeted high-budget hybrid repair fixed that stratum. It is not the main global root cause for proving effective-parameter degeneracy now. The current hybrid TPE `acceptance_margin` run already finds 2+ distinct effective-parameter accepted alternatives for the reviewer-facing cells where Step 04 has support.

The best Step 04 approach to keep is:

1. Hybrid TPE with `acceptance_margin`, current numerical-health handling, full target scope, heldout enabled.
2. Post-hoc effective-space maximin selection from accepted candidates, so Step 05/06 receive distinct effective candidates rather than raw-parameter duplicates. This has now been implemented as the additive Step 04 artifact `outputs/cell_fits/effective_diverse_cell_ensembles.csv`; the full accepted ensemble remains unchanged.

The best secondary approach is not NSGA-II as a replacement. It is objective-diversified hybrid screening, such as metric/trace-shape/balanced-residual hybrid runs, used only as a seed source and only if candidates pass the same all-six and heldout contracts before merge.

Do not run a blind massive full rerun as the next default action. The stronger next improvement is to make the notebook/reporting explicitly separate:

- effective-space accepted alternatives, which Step 04 now supports for 33 cells;
- reviewer-facing heldout support, which excludes VH CONTROL;
- raw physiological plausibility, which remains a Step 08 constraint/interpretability problem;
- final biological degeneracy wording, which remains gated by Steps 05-09.
