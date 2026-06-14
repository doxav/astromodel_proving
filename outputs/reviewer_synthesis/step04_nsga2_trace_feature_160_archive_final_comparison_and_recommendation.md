# Step 04 12-cell 12-worker benchmark recommendation

## Setup
- Target set: 12 cells from `TARGET_FILE_IDS`
- Workers: 12
- Output directory: `outputs/step04_nsga2_trace_feature_160_archive_benchmark_focus`

## Strategy ranking (aggregated)
`outputs/reviewer_synthesis/step04_nsga2_trace_feature_160_archive_final_table.csv`

strategy,elapsed_s,n_candidate_rows,n_accepted_rows,n_accepted_cells,cells_with_3plus_effective_clusters_t0_5,median_top5_min_pairwise_effective_log_distance,median_top5_effective_cluster_count,mean_best_trace_rmse_mV,mean_best_weighted_pass_fraction,accepted_cells_per_minute,acceptance_ratio
random_acceptance_160_archive,125.628,1920,1105,10,10,4.361,5.0,4.023,0.572,4.776,0.576
scipy_seed1_tpe160_pre60_post1,157.205,1944,1210,10,10,2.688,5.0,4.19,0.556,3.817,0.622
nsga2_trace_feature_160_archive_adaptive_500,713.448,161,132,10,10,1.537,5.0,3.953,0.607,0.841,0.82
nsga2_trace_feature_metric_multi_160_archive,61.879,68,51,10,9,2.085,4.0,4.084,0.578,9.696,0.75
nsga2_multi_160_archive,63.104,113,76,10,9,2.139,5.0,4.035,0.586,9.508,0.673
nsga2_multi_160_archive_refine_all_nfev15,186.171,226,144,10,9,2.138,5.0,4.032,0.586,3.223,0.637
nsga2_multi_160_archive_refine_all_nfev30,219.219,226,144,10,9,2.138,5.0,4.032,0.586,2.737,0.637
nsga2_trace_feature_160_archive,59.494,76,54,10,8,1.814,3.0,4.081,0.603,10.085,0.711
nsga2_trace_feature_160_archive_feature_focused,62.67,76,54,10,8,1.814,3.0,4.081,0.603,9.574,0.711
nsga2_trace_feature_160_archive_preseed_tpe160,72.038,79,57,10,8,1.675,3.0,4.081,0.603,8.329,0.722
nsga2_trace_feature_160_archive_strict,729.594,161,0,0,0,,0.0,,,,0.0

## Findings
- `nsga2_trace_feature_160_archive`: fastest with good quality/diversity balance (10 accepted cells, 10.085 cells/min, mean RMSE=4.081, mean pass=0.603).
- `nsga2_trace_feature_160_archive_adaptive_500`: strongest fit proxy and best candidate diversity (best RMSE 3.953, pass 0.607, 10 cells with >=3 clusters) but 12x slower (0.841 cells/min).
- `nsga2_trace_feature_160_archive_strict`: unusable under this contract (`trace<=0.9` and `feature>=0.5`): 0 accepted candidates in 729.6 s.
- `nsga2_trace_feature_160_archive_preseed_tpe160`: near-identical to baseline, no clear gain from pre-seeding in this cohort.
- Random/TPE baselines produce far more candidates but lower runtime efficiency and weaker feature contract than nsga2 baseline.

## Recommendation
1. Set default Step 04 generation strategy to `nsga2_trace_feature_160_archive` (recommended baseline).
2. Keep a paired fallback in docs: `nsga2_trace_feature_160_archive_adaptive_500` for quality-focused runs when runtime budget allows.
3. Do not use `nsga2_trace_feature_160_archive_strict` as a production acceptance rule.
4. Pre-seeding this nsga2 variant from `scipy_seed1_tpe160_pre60_post1` is low impact and optional.

## Visual evidence
- Trace-vs-fitted overlays PDF: `outputs/reviewer_synthesis/step04_generation_trace_overlays_allcells_12workers.pdf`
