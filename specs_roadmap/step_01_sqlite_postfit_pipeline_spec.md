# Step 01 spec update

## What changed
- Step 01 remains a legacy single-current diagnostic step.
- It must not be presented as final mechanism evidence.
- Cache-backed notebook reproducibility is acceptable when raw DBs are absent, but raw direct SQLite reading remains the preferred path.
- The first-pass "to be generated" layer uses top legacy Optuna trials as a source-scoped legacy library, not Step 04 accepted cell-specific ensembles.
- The initial legacy library is `top_n_by_objective` with `legacy_top_n_requested = 300`; thresholded good-enough legacy acceptance is deferred.

## Required outputs
- `outputs/postfit_sqlite/top_trials_all_dbs.csv`
- `outputs/postfit_sqlite/effective_parameter_summary.csv`
- `outputs/postfit_sqlite/representative_mechanism_summary.csv`
- `outputs/postfit_sqlite/legacy_configuration_library.csv`
- `outputs/postfit_sqlite/legacy_configuration_status_by_db.csv`
- `outputs/postfit_sqlite/legacy_condition_parameter_ratios.csv`

## Legacy library contract
- Rows must be source-scoped with `source_scope = "legacy_single_current_optuna"`.
- Rows must use `legacy_configuration_status = "legacy_top300_optuna_trial"` and `legacy_acceptance_rule = "not_thresholded_top_n_first_pass"`.
- Top trials are complete, finite, non-penalty trials ordered by ascending objective and then trial number.
- Preserve `db_name`, canonical `condition`, legacy protocol condition, `current_na`, `trial_number`, `objective`, raw parameters, effective parameters, rank, provenance status, and top-N availability.
- Candidate perturbation factors are exploratory first-pass factors from the Filtered baseline fold grid and legacy ratios; they are not Naris-derived biological magnitudes.

## Acceptance criteria
- `d × pk` invariance is shown.
- Effective parameters are reported for all 18 DBs.
- Representative mechanism summaries cover CONTROL, MFA, and BARIUM.
- Notebook text states that unresolved historical control provenance limits reviewer-facing interpretation.
- The legacy configuration library and factor table are written without using a generic `accepted` column.
