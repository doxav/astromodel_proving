# agents.md

## Repository intent

This repository is being reshaped into a reviewer-response package with three immediate priorities:

1. separate **legacy historical Optuna assets** from the **new ATF experimental dataset**;
2. preserve **DH/VH region** as a first-class biological factor in every ATF-derived table and figure;
3. keep every step **spec-driven, test-backed, notebook-validated**, and explicit about what remains provisional.

## Hard distinctions that must stay explicit

### Legacy historical assets
- `data/1_Initial_xp_fit/*.db`
- `data/1_Initial_xp_fit/*_TRACES.csv`
- `data/threshold_for_good_enough_fits(TO BE RECOMPUTED BASED ON ATF 2_K+ Pumpts Data).csv`

These are historical fitting artifacts. They are useful for provenance, structural-confounding checks, and provisional mechanism summaries, but they are **not** the new reviewer-facing experimental dataset.

### New ATF dataset
- `data/2_K+ Pumps Data/*.atf`

These files are the reviewer-facing experimental basis for step 02 and for downstream multi-sweep fitting. They must always retain:
- `file_id`
- `region` in `{DH, VH}`
- `condition` in `{CONTROL, MFA, MFA_BA}`
- `sweep` in `{1, 2, 3, 4, 5, 6}`

## Development rules

### 1. Specs before code
When behavior changes, update the step spec and the reviewer-response implementation spec first.

### 2. Tests before claims
A notebook figure or table is not reviewer-ready until the matching tests pass.

### 3. No silent provenance assumptions
If a historical objective cannot be reproduced from the documented source, keep it labeled `unresolved`.

### 4. Reference notebook contract
`analysis/astro_atf_analysis_improved_sectioned.ipynb` is the reference working notebook for ATF parsing, preprocessing, and feature extraction.

Whenever repo code or a step notebook reuses or adapts that notebook, say so explicitly:
- in the step notebook markdown,
- in the module docstring or comments,
- and in the reference notebook integration notes.

### 5. Region-first analysis
DH/VH is not a cosmetic label. It must survive all ATF-derived outputs:
- feature tables,
- thresholds,
- reliability weights,
- region-effect summaries,
- reviewer-facing plots.

Region pooling is allowed only as an explicitly labeled sensitivity/fallback.

### 6. Do not double-count redundant features
If two features are near-duplicates (for example `peak_depolarization_mV` and `stim_end_depolarization_mV`), down-weight or flag them rather than letting both dominate the loss.

### 7. Step-appropriate benchmarks only
Do not benchmark Numba-vs-NumPy in step 02. Performance benchmarking belongs in optimization-heavy steps where repeated simulation cost matters.

## Current Pareto conclusions from steps 00-02

- Removing `CONTROL_TRACES_old.csv` is correct.
- That removal does **not** verify the historical control DB objectives.
- Therefore:
  - step 00 should emphasize **legacy-vs-ATF separation**, not a false control-source resolution;
  - step 01 remains a **legacy single-current diagnostic** step;
  - step 02 is the main reviewer-facing gain for now because it rebuilds thresholds from the 37 ATF files with DH/VH structure preserved.
