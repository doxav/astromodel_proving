# Step 04 proposed generation variant benchmark

## Scope
This benchmark evaluates the user-proposed generation design on 12 representative cells using the requested worker count. Holdout is disabled here to isolate all-six-sweep candidate generation yield, runtime, and effective-space diversity.

Target cells: `DH_2_CONTROL`, `3_DH_1_CONTROL`, `2_VH_1_CONTROL`, `1_VH_2_CONTROL`, `1_DH_1_CONTROL`, `DH_1_MFA`, `DH_2_MFA`, `VH_1_MFA`, `VH_6_MFA`, `DH_1_MFA_BA`, `VH_4_MFA_BA`, `VH_7_OG_MFA_BA`.

## Strategies

- `random_acceptance_160_archive`: Pure Optuna random acceptance-margin retained archive, 160 trials.
- `nsga2_trace_feature_160_archive_preseed_tpe160`: NSGA-II trace+feature Pareto archive, 160 trials, pre-seeded with already found accepted candidates from `scipy_seed1_tpe160_pre60_post1`.
- `nsga2_multi_160_archive`: NSGA-II multi-objective acceptance-margin Pareto archive, 160 trials.
- `nsga2_trace_feature_160_archive`: NSGA-II acceptance-margin Pareto archive using only trace and feature objectives, 160 trials.
- `nsga2_multi_160_archive_refine_all_nfev15`: Append raw NSGA-II candidates with SciPy-refined version of every generated candidate, max_nfev=15.
- `nsga2_multi_160_archive_refine_all_nfev30`: Append raw NSGA-II candidates with SciPy-refined version of every generated candidate, max_nfev=30.
- `scipy_seed1_tpe160_pre60_post1`: One SciPy prefit seed, TPE Optuna acceptance archive, one small post-refine.

## Ranked Summary

| selection_strategy                             | elapsed_s | n_candidate_rows | n_accepted_rows | n_accepted_cells | cells_with_3plus_effective_clusters_t0_5 | median_accepted_effective_clusters_t0_5 | effective_diverse_top5_rows | median_top5_min_pairwise_effective_log_distance | accepted_cells_per_minute |
| ---------------------------------------------- | --------- | ---------------- | --------------- | ---------------- | ---------------------------------------- | --------------------------------------- | --------------------------- | ----------------------------------------------- | ------------------------- |
| random_acceptance_160_archive                  | 115.804   | 1920             | 754             | 10               | 10                                       | 75.5                                    | 47                          | 3.879                                           | 5.181                     |
| nsga2_trace_feature_160_archive_preseed_tpe160 | 64.39     | 109              | 65              | 10               | 9                                        | 6.0                                     | 38                          | 1.106                                           | 9.318                     |
| nsga2_multi_160_archive                        | 71.186    | 126              | 74              | 10               | 9                                        | 5.0                                     | 42                          | 1.63                                            | 8.429                     |
| nsga2_trace_feature_160_archive                | 71.702    | 104              | 60              | 10               | 9                                        | 5.5                                     | 38                          | 1.475                                           | 8.368                     |
| nsga2_multi_160_archive_refine_all_nfev15      | 199.727   | 252              | 148             | 10               | 9                                        | 7.0                                     | 48                          | 0.817                                           | 3.004                     |
| nsga2_multi_160_archive_refine_all_nfev30      | 204.591   | 252              | 148             | 10               | 9                                        | 7.0                                     | 48                          | 0.817                                           | 2.933                     |
| scipy_seed1_tpe160_pre60_post1                 | 173.849   | 1944             | 748             | 8                | 8                                        | 81.0                                    | 37                          | 3.684                                           | 2.761                     |

## Interpretation Notes

- `random_acceptance_*_archive` measures the retained Optuna random archive baseline.
- `nsga2_trace_feature_160_archive` tests NSGA-II with reduced objectives (trace + feature) only.
- `nsga2_trace_feature_160_archive_preseed_tpe160` tests pre-seeding NSGA-II with already accepted candidates from `scipy_seed1_tpe160_pre60_post1`.
- `scipy_seed1_*` measures whether a single stronger SciPy prefit seed improves Optuna archive yield.
- `nsga2_multi_160_archive` measures whether a multi-objective Pareto archive provides a fast auxiliary source of accepted candidates.
- Effective-space maximin selection is applied after generation to measure downstream usable diversity.

## Recommendation

`random_acceptance_160_archive` remains the speed-first benchmark default, while `nsga2_trace_feature_160_archive` is the cleaner NSGA-II baseline for reduced-objective experiments. `nsga2_trace_feature_160_archive_preseed_tpe160` is the proposed pre-seeding experiment and is now benchmarked under identical cell coverage.
