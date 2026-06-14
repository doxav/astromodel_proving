# Step 04 pre-seed/nsga2 benchmark final recommendation

## Scope
- Target cells: 12 (`DH_2_CONTROL`, `3_DH_1_CONTROL`, `2_VH_1_CONTROL`, `1_VH_2_CONTROL`, `1_DH_1_CONTROL`, `DH_1_MFA`, `DH_2_MFA`, `VH_1_MFA`, `VH_6_MFA`, `DH_1_MFA_BA`, `VH_4_MFA_BA`, `VH_7_OG_MFA_BA`)
- Command: `python tools/benchmark_step04_generation_variants.py --project-root . --output-dir outputs/step04_nsga2_preseed_objective_benchmark --workers 12`
- Notebook baseline used: `nsga2_trace_feature_160_archive`
- Strategies benchmarked (all on same scope and runtime budget):
  - `nsga2_trace_feature_160_archive`
  - `nsga2_trace_feature_160_archive_strict`
  - `nsga2_trace_feature_metric_multi_160_archive`
  - `nsga2_trace_feature_metric_multi_160_archive_strict`
  - `random_acceptance_160_archive`
  - `nsga2_trace_feature_160_archive_preseed_tpe160`
  - `scipy_seed1_tpe160_pre60_post1`
  - `nsga2_multi_160_archive`

## Core results

| selection_strategy | elapsed_s | n_accepted_cells | n_accepted_rows | median_accepted_effective_clusters_t0_5 | cells_with_3plus_effective_clusters_t0_5 | mean_best_trace_rmse_mV | mean_best_weighted_pass_fraction | median_top5_min_pairwise_effective_log_distance | accepted_cells_per_minute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nsga2_trace_feature_metric_multi_160_archive | 66.10 | 10 | 75 | 5.5 | 10 | 4.9746 | 0.4142 | 2.0258 | 9.0770 |
| random_acceptance_160_archive | 115.08 | 10 | 754 | 75.5 | 10 | 3.8566 | 0.3887 | 3.8788 | 5.2137 |
| nsga2_multi_160_archive | 70.69 | 10 | 74 | 5.0 | 9 | 3.4004 | 0.4207 | 1.6301 | 8.4880 |
| nsga2_trace_feature_160_archive | 71.19 | 10 | 60 | 5.5 | 9 | 3.6164 | 0.3991 | 1.4748 | 8.4284 |
| nsga2_trace_feature_160_archive_preseed_tpe160 | 63.88 | 10 | 65 | 6.0 | 9 | 3.6062 | 0.4009 | 1.1064 | 9.3932 |
| nsga2_multi_160_archive_refine_all_nfev15 | 0.00* | 10 | 148 | 7.0 | 9 | 3.3758 | 0.4270 | 0.8167 | n/a |
| nsga2_multi_160_archive_refine_all_nfev30 | 0.00* | 10 | 148 | 7.0 | 9 | 3.3758 | 0.4270 | 0.8167 | n/a |
| scipy_seed1_tpe160_pre60_post1 | 173.09 | 8 | 748 | 81.0 | 8 | 3.2469 | 0.4095 | 3.6843 | 2.7731 |
| nsga2_trace_feature_160_archive_strict | 729.66 | 0 | 0 | 0.0 | 0 | n/a | n/a | n/a | n/a |
| nsga2_trace_feature_metric_multi_160_archive_strict | 857.66 | 0 | 0 | 0.0 | 0 | n/a | n/a | n/a | n/a |

`*` elapsed_s is 0.00 because these directories are reuse-only refinements chained from an existing source run.

## Key conclusions

1. **Strict contracts are not viable as currently specified**
   - Both strict variants reached 0 accepted cells (0/12) despite running longer (`~12` and `~14` minutes).
   - They should not be used in production for Step 04 candidate generation; the thresholds are too restrictive for this objective configuration.

2. **Most robust non-strict options are two families**
   - `nsga2_multi_160_archive`: best trace quality and better mean feature pass, with very similar per-cell breadth to random.
   - `random_acceptance_160_archive`: highest effective-space min pairwise distance for top candidates and broad per-cell accepted pool, but slower.

3. **Cross-cell stability check (10 cells with accepted solutions)**  
   - Both `nsga2_multi_160_archive` and `random_acceptance_160_archive` keep all 10 target regions/conditions represented.
   - Most fragile cells: `1_DH_1_CONTROL` and `3_DH_1_CONTROL` have low accepted counts across all strategies (3–4 accepted rows), so any configuration still needs downstream rescue logic for these cells.
   - `nsga2_trace_feature_metric_multi_160_archive` is cleaner than `nsga2_trace_feature_160_archive` in acceptance breadth but has higher mean best RMSE (poorer trace fit).

4. **Pre-seeding is low-value in current setup**
   - `nsga2_trace_feature_160_archive_preseed_tpe160` is close to plain `nsga2_trace_feature_160_archive` and does not materially raise acceptance breadth/diversity.

## Recommendation

### Primary
- Set Step 04 generation default to `nsga2_multi_160_archive` for this phase of Step 04 benchmarking.
- Keep `random_acceptance_160_archive` as the diversity-through-quantity fallback.
- Do **not** deploy strict acceptance at 0.9/0.5 while using the current objective scaling.

### Alternative (if downstream quality is prioritized above throughput)
- Use `nsga2_trace_feature_160_archive` as secondary exploratory branch, and only for short validation passes.

### If we need more aggressive diversity growth
- The strongest safe next lever is not stricter acceptance alone. Next useful experiments are:
  - widen objective exploration budget first (`optuna_n_trials`, objective-specific pruning),
  - then add post-hoc effective-space maximin filtering at Step 05/06.
- Avoid raising strictness before confirming that acceptance can remain above a minimum per-cell floor.

## Artefacts

- Comparison summary (bench): `outputs/step04_nsga2_preseed_objective_benchmark/proposed_variant_summary.csv`
- Per-cell comparison: `outputs/step04_nsga2_preseed_objective_benchmark/proposed_variant_by_cell.csv`
- Bench report: `outputs/reviewer_synthesis/step04_proposed_generation_variant_benchmark.md`
- Trace overlay PDF: `outputs/reviewer_synthesis/step04_generation_trace_overlays.pdf`
- Benchmark strategy config: `outputs/step04_nsga2_preseed_objective_benchmark/benchmark_config.json`
