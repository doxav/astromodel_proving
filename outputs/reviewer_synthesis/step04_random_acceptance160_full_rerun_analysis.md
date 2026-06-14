# Step 04 random_acceptance_160_archive full rerun analysis

Comparison target: new full-scope Step 04 run using the notebook preset `random_acceptance_160_archive` versus the backed-up pre-rerun `outputs/reviewer_synthesis/cell_fits_pre_random_acceptance160_20260614_164228`.

## Headline

- The notebook now ran the full 37-cell scope with `optuna_scalar`, random sampler, `acceptance_margin`, 160 trials, 160 retained candidates per cell, and held-out screening enabled.
- Candidate acceptance improved in count and rate: 2110 / 6230 (33.9%) before versus 2827 / 5920 (47.8%) now.
- Effective-diverse downstream rows increased from 97 to 158, mainly because the per-cell selected cap is now 5 instead of 3 and the accepted pool is larger.
- Reviewer-facing held-out support regressed from 33 to 20 cells. This is the decisive negative result for reviewer acceptance.
- Therefore `random_acceptance_160_archive` is better as a fast broad candidate generator, but it is not a replacement for the prior hybrid/TPE plus targeted high-budget merge if the target metric is held-out reviewer-facing cell coverage.

## Run-Level Comparison

| run                                        | backend       | sampler | optuna_n_trials | candidate_top_k | n_fit_points | max_nfev_all6 | n_candidates | n_accepted_candidates | acceptance_rate_percent | n_effective_diverse_candidates | n_reviewer_facing_cells | median_holdout_pass_count |
| ------------------------------------------ | ------------- | ------- | --------------- | --------------- | ------------ | ------------- | ------------ | --------------------- | ----------------------- | ------------------------------ | ----------------------- | ------------------------- |
| previous_hybrid_tpe_targeted_merge         | hybrid        | tpe     | 100             | 500             | 40           | 60            | 6230         | 2110                  | 33.868                  | 97                             | 33                      | 6.000                     |
| random_acceptance_160_archive_full_heldout | optuna_scalar | random  | 160             | 160             | 24           | 20            | 5920         | 2827                  | 47.753                  | 158                            | 20                      | 6.000                     |

## Stratum Delta

| region | condition | cells_current | reviewer_facing_cells_current | reviewer_facing_cells_previous | reviewer_facing_cells_delta | accepted_candidates_current | accepted_candidates_previous | accepted_candidates_delta | median_holdout_pass_count_current | median_holdout_pass_count_previous | median_holdout_pass_count_delta |
| ------ | --------- | ------------- | ----------------------------- | ------------------------------ | --------------------------- | --------------------------- | ---------------------------- | ------------------------- | --------------------------------- | ---------------------------------- | ------------------------------- |
| DH     | CONTROL   | 7             | 0                             | 7                              | -7                          | 28                          | 17                           | 11                        | 2.000                             | 3.000                              | -1.000                          |
| DH     | MFA       | 6             | 6                             | 6                              | 0                           | 876                         | 653                          | 223                       | 6.000                             | 6.000                              | 0.000                           |
| DH     | MFA_BA    | 6             | 6                             | 6                              | 0                           | 858                         | 662                          | 196                       | 6.000                             | 6.000                              | 0.000                           |
| VH     | CONTROL   | 4             | 0                             | 0                              | 0                           | 0                           | 17                           | -17                       | 0.000                             | 1.000                              | -1.000                          |
| VH     | MFA       | 7             | 1                             | 7                              | -6                          | 103                         | 20                           | 83                        | 0.000                             | 5.000                              | -5.000                          |
| VH     | MFA_BA    | 7             | 7                             | 7                              | 0                           | 962                         | 741                          | 221                       | 6.000                             | 6.000                              | 0.000                           |

## Reviewer-Facing Cell Changes

- Retained reviewer-facing cells: 20
- Lost reviewer-facing cells: 13
- Gained reviewer-facing cells: 0

### Lost Cells

| file_id        | region_current | condition_current | holdout_pass_count_current | holdout_pass_count_previous | n_accepted_candidates_current | n_accepted_candidates_previous | best_trace_rmse_mV_current | best_trace_rmse_mV_previous | best_weighted_pass_fraction_current | best_weighted_pass_fraction_previous | accepted_effective_cluster_count_current | accepted_effective_cluster_count_previous |
| -------------- | -------------- | ----------------- | -------------------------- | --------------------------- | ----------------------------- | ------------------------------ | -------------------------- | --------------------------- | ----------------------------------- | ------------------------------------ | ---------------------------------------- | ----------------------------------------- |
| 1_DH_1_CONTROL | DH             | CONTROL           | 2                          | 3                           | 4                             | 2                              | 13.516                     | 7.599                       | 0.333                               | 0.359                                | 4                                        | 2                                         |
| 1_DH_2_CONTROL | DH             | CONTROL           | 2                          | 3                           | 4                             | 4                              | 7.783                      | 13.977                      | 0.333                               | 0.359                                | 4                                        | 4                                         |
| 2_DH_1_CONTROL | DH             | CONTROL           | 2                          | 3                           | 4                             | 2                              | 6.021                      | 12.449                      | 0.333                               | 0.359                                | 4                                        | 2                                         |
| 3_DH_1_CONTROL | DH             | CONTROL           | 2                          | 3                           | 4                             | 3                              | 8.862                      | 15.033                      | 0.333                               | 0.359                                | 4                                        | 2                                         |
| 3_DH_2_CONTROL | DH             | CONTROL           | 2                          | 3                           | 4                             | 2                              | 8.622                      | 14.757                      | 0.333                               | 0.359                                | 4                                        | 2                                         |
| DH_1_CONTROL   | DH             | CONTROL           | 2                          | 3                           | 4                             | 3                              | 6.933                      | 14.380                      | 0.377                               | 0.360                                | 4                                        | 3                                         |
| DH_2_CONTROL   | DH             | CONTROL           | 2                          | 3                           | 4                             | 1                              | 8.161                      | 6.691                       | 0.377                               | 0.321                                | 4                                        | 1                                         |
| VH_1_MFA       | VH             | MFA               | 0                          | 5                           | 15                            | 2                              | 5.163                      | 5.402                       | 0.358                               | 0.319                                | 15                                       | 2                                         |
| VH_2_MFA       | VH             | MFA               | 0                          | 5                           | 15                            | 3                              | 6.343                      | 5.607                       | 0.358                               | 0.350                                | 15                                       | 3                                         |
| VH_3_MFA       | VH             | MFA               | 0                          | 5                           | 15                            | 3                              | 7.801                      | 7.389                       | 0.358                               | 0.358                                | 15                                       | 3                                         |
| VH_4_MFA       | VH             | MFA               | 0                          | 4                           | 15                            | 3                              | 6.562                      | 5.172                       | 0.358                               | 0.329                                | 15                                       | 3                                         |
| VH_5_MFA       | VH             | MFA               | 0                          | 3                           | 15                            | 3                              | 5.731                      | 4.233                       | 0.358                               | 0.308                                | 15                                       | 3                                         |
| VH_6_MFA       | VH             | MFA               | 0                          | 5                           | 15                            | 3                              | 4.715                      | 3.989                       | 0.358                               | 0.335                                | 15                                       | 3                                         |

### Gained Cells

_None._

## Effective-Space Diversity

| run                                        | cells_with_accepted_candidates | cells_with_accepted_effective_clusters_ge3 | cells_with_effective_diverse_candidates | effective_diverse_candidates | median_accepted_effective_cluster_count | median_effective_diverse_cluster_count | median_effective_diverse_min_distance |
| ------------------------------------------ | ------------------------------ | ------------------------------------------ | --------------------------------------- | ---------------------------- | --------------------------------------- | -------------------------------------- | ------------------------------------- |
| previous_hybrid_tpe_targeted_merge         | 35                             | 27                                         | 35                                      | 97                           | 59.000                                  | 3.000                                  | 3.963                                 |
| random_acceptance_160_archive_full_heldout | 33                             | 33                                         | 33                                      | 158                          | 136.000                                 | 5.000                                  | 4.504                                 |

Interpretation: the new run increases the size of the accepted pool and selected effective-diverse pool. However, the diversity increase is post-acceptance and does not rescue held-out reviewer-facing coverage. The accepted effective cluster counts are therefore useful for Step 05/06 exploration, but not sufficient evidence that Step 04 generalization improved.

## Root-Cause Interpretation

1. The new preset deliberately trades local refinement budget for archive breadth: `n_fit_points=24`, `n_starts=4`, `max_nfev_all6=20`, random 160-trial archive. The previous run used `n_fit_points=40`, `n_starts=8`, `max_nfev_all6=60`, hybrid/TPE generation, and strict targeted high-budget replacement for VH/MFA rows.
2. The broader random archive finds many more all-six accepted candidates, but its best candidate per cell is often worse for held-out pass count. This indicates that all-six acceptance breadth and leave-one-current generalization are not equivalent objectives.
3. The largest reviewer-facing losses are not random noise: DH CONTROL drops from 7/7 to 0/7 reviewer-facing cells and VH MFA drops from 7/7 to 1/7. DH MFA, DH MFA_BA, and VH MFA_BA remain fully reviewer-facing, so the new preset is stratum-sensitive rather than uniformly poor.
4. The held-out failures are mostly feature-contract failures, not trace failures. In the current run, VH MFA has low held-out trace RMSE but 39/42 held-out sweeps fail the feature threshold; DH CONTROL consistently keeps only sweeps 1-2 above the feature threshold and loses the previous third held-out pass.
5. The stale `targeted_step04_merge_manifest.csv` from the prior targeted merge was removed from `outputs/cell_fits` and from the new run ZIP after verifying that it is preserved in the pre-rerun backup and is not part of the current artifact manifest.

## Decision

Do not promote `random_acceptance_160_archive` as the sole canonical Step 04 reviewer-facing configuration. It is worth keeping as a clear fast generator or as the first stage of a two-stage process, but the full rerun shows it regresses held-out reviewer-facing support versus the previous targeted run.

Recommended next configuration direction: generate a random acceptance archive, then run a targeted held-out rescue/refinement stage for cells with holdout pass count < 3, keeping effective-diverse selection as a downstream selection step rather than as proof of generalization.

## Audit Files

- Cell deltas: `outputs/reviewer_synthesis/step04_random_acceptance160_full_rerun_cell_delta.csv`
- Stratum deltas: `outputs/reviewer_synthesis/step04_random_acceptance160_full_rerun_stratum_delta.csv`
- Effective diversity deltas: `outputs/reviewer_synthesis/step04_random_acceptance160_full_rerun_effective_delta.csv`
