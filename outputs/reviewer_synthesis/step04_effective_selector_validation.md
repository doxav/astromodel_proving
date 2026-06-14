# Step 04 Effective-Space Selector Validation

## Objective

Add a post-hoc Step 04 selection layer so downstream Step 05/06 can use accepted candidates that are distinct in effective-parameter space, rather than arbitrary rank/top-k rows or raw-parameter duplicates.

Effective coordinates used:

- `P_gap_eff`
- `gamma_t_eff`
- `gamma_s_eff`
- `volume_ratio_wa_wo`

Raw/nuisance coordinates are not part of the distance metric.

## Strategies Compared

All strategies used current canonical Step 04 accepted candidates from `outputs/cell_fits/accepted_cell_ensembles.csv`, filtered to all-six accepted and effective-plausible rows. Target was up to 3 selected candidates per cell.

| Strategy | Rows | Cells | Cells with 2+ selected | Cells with 2+ effective clusters | Reviewer-facing cells with 2+ effective clusters | Median min log-effective distance | Mean trace RMSE | Mean weighted pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `effective_maximin_best_seed` | 97 | 35 | 33 | 33 | 31 | 4.898 | 6.271 | 0.401 |
| `weighted_effective_novelty_quality` | 97 | 35 | 33 | 33 | 31 | 4.887 | 6.334 | 0.404 |
| `quality_filtered_effective_maximin` | 97 | 35 | 33 | 33 | 31 | 3.963 | 6.354 | 0.405 |
| `effective_cluster_medoids_t0_5` | 97 | 35 | 33 | 33 | 31 | 2.122 | 6.311 | 0.409 |
| `quality_top_k` | 97 | 35 | 33 | 33 | 31 | 1.266 | 6.503 | 0.411 |

`quality_top_k` has a 10th-percentile minimum distance of 0.0, confirming that plain rank/top-k can keep effective duplicates. The maximin variants remove this duplicate failure mode.

## Winners

### Primary: `quality_filtered_effective_maximin`

This is the best default. It starts from the best quality-ranked effective-plausible candidate, restricts the search to the higher-quality half of each cell's accepted pool, and then greedily maximizes minimum log-effective distance. It is less extreme than pure maximin while still giving strong spacing.

Current canonical output:

- `outputs/cell_fits/effective_diverse_cell_ensembles.csv`
- `outputs/cell_fits/effective_diverse_selection_summary.csv`
- SQLite tables:
  - `effective_diverse_cell_ensembles`
  - `effective_diverse_selection_summary`

It selected 97 candidates across 35 accepted cells. Thirty-three cells have 2+ effective clusters. The two sparse cells with only one selected candidate are `3_DH_1_CONTROL` and `DH_2_CONTROL`; they have only one effective-plausible accepted candidate available, so this is a data limitation, not a selector failure.

### Secondary: `effective_maximin_best_seed`

This is the best sensitivity strategy when maximum effective-space separation is the priority. It had the highest median minimum distance. It should be kept as an audit/sensitivity option because it can select more extreme accepted candidates.

## Implementation

Added `src/effective_candidate_selection.py` with:

- `select_effective_diverse_candidates`
- `summarize_effective_diverse_selection`

Step 04 now writes the effective-diverse selected subset and summary alongside the full accepted ensemble. The full accepted ensemble remains unchanged and auditable.

Step 05 and Step 06 now accept an optional `step04_source_path`, so they can be pointed at `outputs/cell_fits/effective_diverse_cell_ensembles.csv` without changing their default full-scope behavior.

## Verification

Commands run:

```bash
python -m py_compile src/effective_candidate_selection.py src/step04_cell_fits.py src/step04_outputs.py src/step05_mechanistic_decomposition.py src/step06_predictive_validation.py
pytest tests/test_effective_candidate_selection.py tests/test_step04_cell_fits.py::test_single_control_cell_fit_writes_expected_outputs tests/test_step04_cell_fits.py::test_step04_outputs_are_downstream_reusable tests/acceptance/test_step06_acceptance.py::test_step06_can_read_effective_diverse_step04_source
```

Result: 6 passed.

## Recommendation

Use `quality_filtered_effective_maximin` as the Step 04 default effective-diverse selected subset. Keep `effective_maximin_best_seed` as the sensitivity runner-up. Do not replace the full accepted ensemble; add the effective-diverse subset as the downstream candidate source when Step 05/06 need balanced per-cell effective diversity.
