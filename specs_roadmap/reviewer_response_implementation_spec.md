# Reviewer-response implementation spec update (steps 00-02)

This update narrows the near-term implementation contract to the changes that matter most for the reviewer response.

## Pareto conclusions from the current outputs

1. The repo must make a hard distinction between:
   - **legacy historical Optuna DB/CSV assets**, and
   - **new ATF experimental traces**.
2. Removing `CONTROL_TRACES_old.csv` is the right cleanup, but it does **not** verify the historical control objectives.
3. Therefore, the efficient next move is **not** to spend more time trying to rescue legacy control validation inside steps 00-01.
4. The biggest reviewer-facing gain remains:
   - region-aware ATF thresholds,
   - explicit reliability weighting,
   - and downstream six-sweep fitting that uses those thresholds.

## Step 00 — Data provenance and objective reproducibility audit

### Updated scientific message
Step 00 is now primarily a **dataset contract and provenance separation** step.

It must show, side by side:
- the historical single-current DB/CSV fitting assets;
- the legacy threshold example file;
- and the new ATF dataset that will drive reviewer-facing threshold rebuilding.

### Updated technical contract
- Discover 18 historical SQLite DBs when raw historical assets are present.
- Read `CONTROL_TRACES.csv`, `MFA_TRACES.csv`, and `BARIUM_TRACES.csv`.
- Do **not** reference or use `CONTROL_TRACES_old.csv`.
- Parse 37 ATF files with explicit `region`, `condition`, and `file_id`.
- Write a `data_source_contract.csv` that distinguishes:
  - `legacy_optuna_db`
  - `legacy_optuna_trace_csv`
  - `legacy_threshold_example_csv`
  - `new_atf_trace`
- Keep historical control objectives labeled `unresolved` unless the documented `CONTROL_TRACES.csv` source verifies them.
- Do not imply that ATF traces and historical Optuna traces are interchangeable.

### Updated notebook obligations
The notebook must explicitly say:
- historical Optuna traces are legacy fitting artifacts;
- ATF files are the new reviewer-facing experimental dataset;
- `CONTROL_TRACES_old.csv` was removed and is not used;
- control provenance remains unresolved under the currently documented source.

## Step 01 — SQLite post-fit pipeline and hidden-mechanism simulation

### Updated scientific message
Step 01 remains useful, but only as a **legacy single-current diagnostic step**.

It supports:
- structural confounding demonstration (`d × pk`);
- effective-parameter reporting;
- provisional mechanism summaries.

It does **not** by itself justify final mechanism claims because:
- it is single-current;
- historical control provenance remains unresolved;
- and the final reviewer-facing accepted ensembles should come from six-sweep fitting.

### Updated technical contract
- Prefer raw direct SQLite reading when DBs are present.
- Allow a cache-backed summary path for notebook reproducibility when raw DBs are absent.
- Preserve `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`.
- Keep representative mechanism summaries explicit about being provisional legacy summaries.

### Updated notebook obligations
The notebook must say that:
- no ATF feature-extraction logic is reused here;
- this step interprets legacy DBs, not the new ATF dataset;
- unresolved historical control provenance limits reviewer-facing strength.

## Step 02 — Rebuild region-aware feature thresholds from the 37 ATF files

### Updated scientific message
Step 02 is now the strongest immediate reviewer-facing step among 00-02.

### Updated technical contract
- Reuse/adapt the working logic from `analysis/astro_atf_analysis_improved_sectioned.ipynb` for:
  - ATF discovery,
  - parsing,
  - preprocessing,
  - and sweep-level feature extraction.
- Preserve all 222 sweep rows.
- Build primary thresholds by `condition × region × sweep × feature`.
- Export:
  - `feature_table_by_sweep.csv`
  - `condition_region_sweep_thresholds.csv`
  - `feature_reliability_weights.csv`
  - `condition_feature_reliability.csv`
  - `region_condition_cell_counts.csv`
  - `region_effect_summary.csv`
  - `redundancy_diagnostics.csv`
- Flag the near-redundancy of `peak_depolarization_mV` and `stim_end_depolarization_mV`.
- Make condition-level reliability explicit.
- Omit the Numba-vs-NumPy benchmark from step 02.

### Updated notebook obligations
The notebook must:
- state exactly what is reused from the reference notebook and what is not;
- show the full canonical sweep-level feature table rather than a 12-row preview;
- show the full redundancy diagnostics table;
- show the full region-effect summary table;
- show the condition-level reliability table;
- avoid performance benchmarking that belongs to later optimization steps.

## Downstream consequence for later steps

Because the current outputs show strong redundancy between `peak_depolarization_mV` and `stim_end_depolarization_mV`, later fitting steps should either:
- keep only one of them in the primary weighted loss, or
- down-weight the pair so they do not count twice.

That is a better Pareto use of effort than trying to reinterpret unresolved historical control DB provenance as if it were fixed.
