# Step 01 spec update

## What changed
- Step 01 remains a legacy single-current diagnostic step.
- It must not be presented as final mechanism evidence.
- Cache-backed notebook reproducibility is acceptable when raw DBs are absent, but raw direct SQLite reading remains the preferred path.

## Required outputs
- `outputs/postfit_sqlite/top_trials_all_dbs.csv`
- `outputs/postfit_sqlite/effective_parameter_summary.csv`
- `outputs/postfit_sqlite/representative_mechanism_summary.csv`

## Acceptance criteria
- `d × pk` invariance is shown.
- Effective parameters are reported for all 18 DBs.
- Representative mechanism summaries cover CONTROL, MFA, and BARIUM.
- Notebook text states that unresolved historical control provenance limits reviewer-facing interpretation.
