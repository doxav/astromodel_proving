# Step 04 diverse candidate generation benchmark

## Scope
This is a method-development benchmark, not a reviewer-facing full-target Step 04 replacement. It used nine representative cells spanning DH/VH and CONTROL/MFA/MFA_BA, including hard or sparse cells from the current full run. Holdout was disabled to measure all-six-sweep candidate generation speed and diversity.

Target cells: `DH_2_CONTROL`, `3_DH_1_CONTROL`, `2_VH_1_CONTROL`, `1_VH_2_CONTROL`, `DH_1_MFA`, `DH_1_MFA_BA`, `VH_1_MFA`, `VH_6_MFA`, `VH_4_MFA_BA`.

## Main Result
The strongest fast generation strategy is `optuna_random_acceptance_archive`: it generated accepted candidates for 7/9 cells in 93 s, produced >=3 effective clusters for all 7 accepted cells, and retained enough candidates for effective-space maximin selection. `optuna_nsga2_multi_acceptance_archive` is the best complementary strategy: it is faster and gives larger spacing among its top diverse candidates, but it produces fewer accepted rows.

The many-start least-squares route is not competitive for this goal: the 32-start variant was interrupted as too slow; the bounded 12-start variant took 292 s and accepted only 3/9 cells.

## Ranked Summary
| selection_strategy                          | elapsed_s | n_candidate_rows | n_accepted_rows | n_accepted_cells | cells_with_3plus_effective_clusters_t0_5 | median_accepted_effective_clusters_t0_5 | effective_diverse_top5_rows | median_top5_min_pairwise_effective_log_distance | accepted_cells_per_minute |
| ------------------------------------------- | --------- | ---------------- | --------------- | ---------------- | ---------------------------------------- | --------------------------------------- | --------------------------- | ----------------------------------------------- | ------------------------- |
| canonical_current_full_step04_accepted_only |           | 346              | 346             | 8                | 5                                        | 3                                       | 27                          | 2.07                                            |                           |
| optuna_random_trace_shape_archive           | 78.899    | 1080             | 353             | 7                | 7                                        | 14                                      | 33                          | 2.867                                           | 5.323                     |
| optuna_random_acceptance_archive            | 93.117    | 1080             | 353             | 7                | 7                                        | 14                                      | 33                          | 2.867                                           | 4.51                      |
| append_random_plus_nsga2                    | 142.707   | 1164             | 386             | 7                | 7                                        | 17                                      | 35                          | 2.867                                           | 2.943                     |
| append_random_plus_hybrid_random            | 420.5     | 1926             | 633             | 7                | 7                                        | 14                                      | 35                          | 2.958                                           | 0.999                     |
| append_random_plus_nsga2_plus_hybrid_random | 470.09    | 2010             | 666             | 7                | 7                                        | 17                                      | 35                          | 2.867                                           | 0.893                     |
| append_all_completed_generation_strategies  | 1084.671  | 3657             | 1202            | 7                | 7                                        | 17                                      | 35                          | 2.867                                           | 0.387                     |
| optuna_nsga2_multi_acceptance_archive       | 49.59     | 84               | 33              | 7                | 5                                        | 5                                       | 25                          | 4.338                                           | 8.469                     |
| hybrid_random_acceptance_large_archive      | 327.383   | 846              | 280             | 7                | 5                                        | 9                                       | 29                          | 2.749                                           | 1.283                     |
| baseline_hybrid_tpe_acceptance              | 243.817   | 459              | 149             | 4                | 3                                        | 29                                      | 16                          | 2.494                                           | 0.984                     |
| least_squares_12starts_soft_l1_bounded      | 291.865   | 108              | 34              | 3                | 3                                        | 11                                      | 15                          | 3.255                                           | 0.617                     |

## Interpretation
- Running more random Optuna acceptance-margin trials and retaining the full trial archive is the best fast way to generate more accepted effective-mechanism diversity.
- NSGA-II multi-objective acceptance-margin is a useful complementary archive because it finds fewer but more widely separated accepted mechanisms quickly.
- Hybrid random improves accepted-cell coverage versus the baseline but costs more than pure Optuna random and does not improve diversity per second.
- Trace-shape random produced the same accepted set as random acceptance in this benchmark because all 120 random trials were retained and then rescored by the Step 04 contract; the objective ranking did not matter when `candidate_top_k == n_trials`.
- More SciPy starts are expensive and low-yield for generating broad accepted diversity; use them only as targeted refinement of selected candidates, not as the primary diversity generator.

## Recommended Step 04 Generation Design
1. Add a first generation stage using pure Optuna random acceptance-margin with a full retained archive, e.g. `backend=optuna_scalar`, `optuna_sampler=random`, `optuna_objective=acceptance_margin`, `optuna_n_trials >= 120`, `candidate_top_k >= optuna_n_trials`.
2. Add a complementary NSGA-II multi-objective archive, e.g. `backend=optuna_multi`, `optuna_sampler=nsga2`, `multi_objective_names=(trace, feature, binary, fail)`, with the same retained-archive principle.
3. Append generated candidate pools per cell, prefixing candidate IDs by generation strategy, then re-rank under the existing Step 04 contract.
4. Apply the effective-space maximin selector after appending, not instead of generation.
5. Keep hybrid/SciPy refinement only for top selected diverse candidates or for cells still missing accepted candidates.

## Output Files
- `outputs/reviewer_synthesis/step04_diverse_generation_strategy_summary.csv`
- `outputs/reviewer_synthesis/step04_diverse_generation_strategy_by_cell.csv`
- `outputs/reviewer_synthesis/step04_diverse_generation_effective_clusters_pivot.csv`
- `outputs/reviewer_synthesis/step04_diverse_generation_append_pool_selected_candidates.csv`
