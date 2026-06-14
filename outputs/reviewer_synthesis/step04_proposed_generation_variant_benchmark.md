# Step 04 proposed generation variant benchmark

## Scope
This benchmark evaluates the user-proposed generation design on 12 representative cells using the requested worker count. Holdout is disabled here to isolate all-six-sweep candidate generation yield, runtime, and effective-space diversity.

Target cells: `DH_2_CONTROL`, `3_DH_1_CONTROL`, `2_VH_1_CONTROL`, `1_VH_2_CONTROL`, `1_DH_1_CONTROL`, `DH_1_MFA`, `DH_2_MFA`, `VH_1_MFA`, `VH_6_MFA`, `DH_1_MFA_BA`, `VH_4_MFA_BA`, `VH_7_OG_MFA_BA`.

## Strategies

- `random_acceptance_160_archive`: Pure Optuna random acceptance-margin retained archive, 160 trials.
- `scipy_seed1_tpe160_pre60_post1`: One SciPy prefit seed, TPE Optuna acceptance archive, one small post-refine.
- `nsga2_trace_feature_160_archive_adaptive_500`: NSGA-II trace+feature Pareto archive with adaptive extension (target 160, batches of 100 up to 500) and extended trainable parameters.
- `nsga2_trace_feature_metric_multi_160_archive`: NSGA-II metric_scalar Pareto archive with trace+feature objective names (trace/feature), 160 trials.
- `nsga2_multi_160_archive`: NSGA-II multi-objective acceptance-margin Pareto archive, 160 trials.
- `nsga2_multi_160_archive_refine_all_nfev15`: Append raw NSGA-II candidates with SciPy-refined version of every generated candidate, max_nfev=15.
- `nsga2_multi_160_archive_refine_all_nfev30`: Append raw NSGA-II candidates with SciPy-refined version of every generated candidate, max_nfev=30.
- `nsga2_trace_feature_160_archive`: NSGA-II acceptance-margin Pareto archive using only trace and feature objectives, 160 trials.
- `nsga2_trace_feature_160_archive_feature_focused`: NSGA-II trace+feature Pareto archive with stronger feature-weighting in scalar ranking.
- `nsga2_trace_feature_160_archive_preseed_tpe160`: NSGA-II trace+feature Pareto archive, 160 trials, pre-seeded with already found accepted candidates from `scipy_seed1_tpe160_pre60_post1`.
- `nsga2_trace_feature_160_archive_strict`: NSGA-II acceptance-margin trace+feature Pareto archive with strict acceptance contract and adaptive extension (target 160, batches of 100 up to 500 if <3 accepted).

## Ranked Summary

| selection_strategy                              | elapsed_s | n_candidate_rows | n_accepted_rows | n_accepted_cells | cells_with_3plus_effective_clusters_t0_5 | median_accepted_effective_clusters_t0_5 | effective_diverse_top5_rows | median_top5_min_pairwise_effective_log_distance | accepted_cells_per_minute |
| ----------------------------------------------- | --------- | ---------------- | --------------- | ---------------- | ---------------------------------------- | --------------------------------------- | --------------------------- | ----------------------------------------------- | ------------------------- |
| random_acceptance_160_archive                   | 125.628   | 1920             | 1105            | 10               | 10                                       | 126.0                                   | 50                          | 4.361                                           | 4.776                     |
| scipy_seed1_tpe160_pre60_post1                  | 157.205   | 1944             | 1210            | 10               | 10                                       | 76.0                                    | 48                          | 2.688                                           | 3.817                     |
| nsga2_trace_feature_160_archive_adaptive_500    | 713.448   | 161              | 132             | 10               | 10                                       | 10.0                                    | 49                          | 1.537                                           | 0.841                     |
| nsga2_trace_feature_metric_multi_160_archive    | 61.879    | 68               | 51              | 10               | 9                                        | 4.0                                     | 38                          | 2.085                                           | 9.696                     |
| nsga2_multi_160_archive                         | 63.104    | 113              | 76              | 10               | 9                                        | 6.5                                     | 44                          | 2.139                                           | 9.508                     |
| nsga2_multi_160_archive_refine_all_nfev15       | 186.171   | 226              | 144             | 10               | 9                                        | 8.0                                     | 47                          | 2.138                                           | 3.223                     |
| nsga2_multi_160_archive_refine_all_nfev30       | 219.219   | 226              | 144             | 10               | 9                                        | 8.0                                     | 47                          | 2.138                                           | 2.737                     |
| nsga2_trace_feature_160_archive                 | 59.494    | 76               | 54              | 10               | 8                                        | 4.0                                     | 34                          | 1.814                                           | 10.085                    |
| nsga2_trace_feature_160_archive_feature_focused | 62.67     | 76               | 54              | 10               | 8                                        | 4.0                                     | 34                          | 1.814                                           | 9.574                     |
| nsga2_trace_feature_160_archive_preseed_tpe160  | 72.038    | 79               | 57              | 10               | 8                                        | 4.0                                     | 34                          | 1.675                                           | 8.329                     |
| nsga2_trace_feature_160_archive_strict          | 729.594   | 161              | 0               | 0                | 0                                        | 0.0                                     | 0                           |                                                 |                           |

## Interpretation Notes

- `random_acceptance_*_archive` measures the retained Optuna random archive baseline.
- `nsga2_trace_feature_160_archive` tests NSGA-II with reduced objectives (trace + feature) only.
- `nsga2_trace_feature_160_archive_preseed_tpe160` tests pre-seeding NSGA-II with already accepted candidates from `scipy_seed1_tpe160_pre60_post1`.
- `nsga2_trace_feature_160_archive_feature_focused` increases feature influence in candidate ranking.
- `nsga2_trace_feature_160_archive_adaptive_500` enables adaptive extension (0.3/0.30 thresholds) with a 500-trial cap and expanded trainable space when beyond 100 trials.
- `nsga2_trace_feature_160_archive_adaptive_strict` applies strict acceptance thresholds (0.9 mV, 0.5) together with adaptive extension and expanded trainable parameters.
- `nsga2_trace_feature_160_archive_strict` adds stricter acceptance thresholds (`trace_rmse<=0.9`, `feature>=0.5`) and adaptive trial extension until >=3 accepted per cell, max 500.
- `nsga2_trace_feature_metric_multi_160_archive` uses Pareto objective components from `metric_scalar` with trace+feature names only.
- `nsga2_trace_feature_metric_multi_160_archive_strict` combines the strict contract with `metric_scalar`.
- `scipy_seed1_*` measures whether a single stronger SciPy prefit seed improves Optuna archive yield.
- `nsga2_multi_160_archive` measures whether a multi-objective Pareto archive provides a fast auxiliary source of accepted candidates.
- Effective-space maximin selection is applied after generation to measure downstream usable diversity.

## Recommendation

`nsga2_trace_feature_160_archive` is now the requested default baseline for this benchmark family. `nsga2_trace_feature_160_archive_adaptive_500` is the principal quality-depth follow-up (it adds extra parameters if trial count grows and can extend to 500 trials when coverage is weak). `nsga2_trace_feature_160_archive_feature_focused` provides a focused ranking variant, and `nsga2_trace_feature_160_archive_preseed_tpe160` remains the nearest pre-seeding extension. `random_acceptance_160_archive` stays as the speed-first fallback.
