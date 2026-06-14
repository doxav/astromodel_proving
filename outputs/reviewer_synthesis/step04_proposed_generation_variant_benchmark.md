# Step 04 proposed generation variant benchmark

## Scope
This benchmark evaluates the user-proposed generation design on 12 representative cells using 10 workers on a 20-vCPU machine. Holdout is disabled here to isolate all-six-sweep candidate generation yield, runtime, and effective-space diversity.

Target cells: `DH_2_CONTROL`, `3_DH_1_CONTROL`, `2_VH_1_CONTROL`, `1_VH_2_CONTROL`, `1_DH_1_CONTROL`, `DH_1_MFA`, `DH_2_MFA`, `VH_1_MFA`, `VH_6_MFA`, `DH_1_MFA_BA`, `VH_4_MFA_BA`, `VH_7_OG_MFA_BA`.

## Strategies

- `random_acceptance_160_archive`: Pure Optuna random acceptance-margin retained archive, 160 trials.
- `random_acceptance_240_archive`: Pure Optuna random acceptance-margin retained archive, 240 trials.
- `scipy_seed1_random160_pre60_post1`: One SciPy prefit seed, random Optuna acceptance archive, one small post-refine.
- `nsga2_multi_160_archive`: NSGA-II multi-objective acceptance-margin Pareto archive, 160 trials.
- `tpe_refine_candidate_only_12_nfev4`: Bounded feasibility check: one SciPy seed; TPE proposes 12 trials; every trial is SciPy-refined for 4 function evaluations, but only raw TPE trials are told to Optuna.
- `scipy_seed1_tpe160_pre60_post1`: One SciPy prefit seed, TPE Optuna acceptance archive, one small post-refine.
- `tpe_refine_feedback_history_12_nfev4`: Bounded feasibility check: one SciPy seed; TPE proposes 12 trials; every trial is SciPy-refined for 4 function evaluations, and each refined point/score is added to Optuna history.

## Ranked Summary

| selection_strategy                   | elapsed_s | n_candidate_rows | n_accepted_rows | n_accepted_cells | cells_with_3plus_effective_clusters_t0_5 | median_accepted_effective_clusters_t0_5 | effective_diverse_top5_rows | median_top5_min_pairwise_effective_log_distance | accepted_cells_per_minute |
| ------------------------------------ | --------- | ---------------- | --------------- | ---------------- | ---------------------------------------- | --------------------------------------- | --------------------------- | ----------------------------------------------- | ------------------------- |
| random_acceptance_160_archive        | 125.278   | 1920             | 754             | 10               | 10                                       | 75.5                                    | 47                          | 3.879                                           | 4.789                     |
| random_acceptance_240_archive        | 204.315   | 2880             | 1145            | 10               | 10                                       | 115.0                                   | 47                          | 3.879                                           | 2.937                     |
| scipy_seed1_random160_pre60_post1    | 204.664   | 1944             | 766             | 10               | 10                                       | 75.5                                    | 49                          | 3.879                                           | 2.932                     |
| nsga2_multi_160_archive              | 76.751    | 126              | 74              | 10               | 9                                        | 5.0                                     | 42                          | 1.63                                            | 7.818                     |
| tpe_refine_candidate_only_12_nfev4   | 182.128   | 300              | 133             | 9                | 7                                        | 18.0                                    | 34                          | 3.13                                            | 2.965                     |
| scipy_seed1_tpe160_pre60_post1       | 187.263   | 1944             | 748             | 8                | 8                                        | 81.0                                    | 37                          | 3.684                                           | 2.563                     |
| tpe_refine_feedback_history_12_nfev4 | 188.601   | 300              | 117             | 8                | 5                                        | 18.0                                    | 28                          | 1.785                                           | 2.545                     |

## Interpretation Notes

- `random_acceptance_*_archive` measures the retained Optuna random archive baseline.
- `scipy_seed1_*` measures whether a single stronger SciPy prefit seed improves Optuna archive yield.
- `tpe_refine_*_12_nfev4` is a bounded feasibility check for refining every TPE proposal; the heavier 60-trial/8-nfev path was too slow to keep in the default fast benchmark.
- `tpe_refine_feedback_history_12_nfev4` directly tests whether adding the SciPy-refined point/score back into Optuna history improves generation.
- `nsga2_multi_160_archive` measures whether a multi-objective Pareto archive provides a fast auxiliary source of accepted candidates.
- Effective-space maximin selection is applied after generation to measure downstream usable diversity.

## Recommendation

`random_acceptance_160_archive` is the best default Step 04 generation upgrade for speed, accepted-cell breadth, and effective-space diversity. `random_acceptance_240_archive` is the best larger-pool option when a slower run is acceptable. NSGA-II is useful as an auxiliary quality-oriented source, not as the main diverse-candidate source. Per-trial SciPy refinement and refined-point Optuna feedback should not be promoted: the bounded versions were slower and produced fewer accepted/diverse cells than random archives.
