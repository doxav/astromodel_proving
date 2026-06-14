To be fixed in notebook steps for reviewers acceptances for model proving:

1. locate the step which has effective parameter, they should be contrained by biological limits (a cell of an early matching step could allow to set them and convert to a csv or json), then updated effective parameters ?
        for current parameter_ranges.csv
            is it generated ?
            should we specify it ? merge it maybe if 1 generated and 1 is defined ?
 
2. fix: there are many interpretations in notebooks which are not conditioned by data but just defined text => we have to condition them if possible or write that this analysis must be updated if results changes (specify what)

3. Re-use only what is missing in a DRY code way from the characterization/classification notebook based on the analysis (mainly on Step 05):
        <<<
        **Short Answer**
        As executed, the attached notebook does **not strongly address** the weak reviewer critiques. It is a **legacy Optuna/SQLite smoke run**: 1 DB (`CONTROL_50nA.db`), 2 top-ranked trials, no threshold CSV, and `threshold_source = none_top_n_provisional`.
        Methodologically, though, it contains useful material that could strengthen **R5 mechanistic interpretation** and **R4 hidden ionic-dynamics interpretation** if rerun at full scope and connected to the accepted ATF-based Step 04-06 pipeline.
        | Critique | Current weak point in `outputs/executed_notebooks` | Attached smoke notebook contribution | Strength as executed | Could become strong if expanded? |
        |---|---|---|---|---|
        | `R1` degeneracy vs non-identifiability | Step 03 handles this best; Step 05 still insufficient for true mechanism regimes | Corrects interpretation of `d`, uses `Pkgap = pk*d`, separates mechanism labels from raw parameters | Weak/partial | Yes, but only as downstream mechanism evidence, not as identifiability proof |
        | `R2` experimental variability/noise/data constraints | Step 02 is ATF-based; Step 04-06 still small | Uses legacy DBs, optional old thresholds, no new ATF variability | No | Not unless rewritten to use ATF feature distributions |
        | `R3` model assumptions | Step 07 proxy/gating/split still limited | Does not test model families or proxy assumptions systematically | No | Not directly |
        | `R4` Vm-only weakly constrains ionic dynamics | Step 03 says raw parameters are weakly constrained; Step 05/06 limited | Reconstructs hidden variables, `K_o`, currents, flux budgets, Kir/GJ/local-spatial scores | Moderate method, weak evidence | Yes, if applied to accepted ATF ensembles and reported conservatively |
        | `R5` regimes/pathways/phenotypes | Step 05 says `insufficient_evidence` | Adds signed flux modes, phenotype tags, local load vs spatial export, pump/Kir/GJ proxies | Weak as executed | Most promising target for strong improvement |
        | `R6` predictive/perturbation robustness | Step 06 still `insufficient_evidence` | No held-out prediction or perturbation validation | No | Only if combined with Step 06-style validation |
        | `R7` clarity/organization | Existing outputs are improving but fragmented | Has measure registry, phenotype dictionaries, explicit deprecated/proxy labels | Partial | Yes, useful for supplement organization |
        **Neurobiology Assessment**
        The biological framing is useful: signed local K load, spatial export, Kir contribution, gap-junction conductance proxy, dynamic recruitment, and `K_o` recovery are more interpretable than raw fitted parameters. That is directly relevant to astrocyte K buffering.
        But the current executed file is **hypothesis-generating**, not reviewer-proof. The phenotypes are simulation-derived from legacy single-current fits, not validated against new ATF cell-specific accepted ensembles, region/condition effects, held-out currents, or perturbations.
        **Best Use**
        Use this notebook as a source of **mechanistic feature engineering** for Step 05:
        - signed flux budget: local load `L`, spatial export `S`, bath/source terms;
        - corrected `d` / `Pkgap` interpretation;
        - Kir/GJ/leak/pump proxy summaries;
        - phenotype dictionary and clear measure registry.
        Do **not** cite the attached smoke execution as strong evidence. To make it reviewer-facing, rerun or port it on full accepted Step 04 ensembles across all ATF cells/conditions/regions, then validate with Step 06 prediction and perturbation checks.
        Commands run: structured `python3` notebook extraction, `rg`-style content search via parsed notebook text, and `find`/`sed` checks for generated smoke outputs.
        I completed the systematic comparison and put the full cell-by-cell table into a separate report:
        [Download the detailed notebook-vs-repo analysis report](sandbox:/mnt/data/notebook_vs_astromodel_proving_analysis.md)
        I could inspect the notebook structure and its recorded smoke outputs, but I could not rerun the full notebook because the notebook’s referenced `/mnt/data/data(3).zip` is not present in this session. The analysis below is therefore based on the attached executed notebook, its saved outputs, the thread PDF, and the current GitHub repo state.
        ## Executive conclusion
        The attached notebook is **not yet equivalent to `astromodel_proving` Step 05**. It is best understood as an **exploratory legacy-Optuna phenotype-characterization notebook** that extends the current repo’s Step 05 mechanistic decomposition.
        The repo already addresses the formal reviewer-response mechanism screen: Step 05 converts accepted Step 04 ensembles into mechanism-level evidence, with conservative claim language, explicit outputs, clustering, representative selection, geometry diagnostics, and bootstrap stability. It also explicitly says candidate mechanisms should remain provisional until Step 06 validation.
        The notebook adds several important layers that are **not yet in the repo**: four-window M-vector analysis, signed local/spatial flux budgets, `dKs_activation_score`, `gs/alpha2` as available spatial-transfer surface capacity, dynamic conversion factors, `r_model`/isopotentiality proxy, phenotype labels, Fv/Fk dictionaries, MFA-Control phenotype contrasts, and legacy Optuna `.db` parsing. These directly follow the thread’s final refinement around `d`, `Pkgap`, `χK/A_dKs`, `gs/alpha2`, and dynamic surface recruitment.
        ## A/B status summary
        | Notebook element                                      |                              Repo status | Classification               | Main judgment                                                                                                                             |
        | ----------------------------------------------------- | ---------------------------------------: | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
        | Core ODE structure                                    |                           Mostly present | **A1 equivalent**            | Repo already has canonical astrocyte model logic; notebook duplicates it.                                                                 |
        | Current/K_o summary                                   |                      Present but simpler | **A3 refine**                | Repo computes Kir/gap/leak fractions and K_o summaries; notebook adds signed state-flux windows.                                          |
        | `d`, `pk`, `Pkgap` mapping                            |                       Partly contradicts | **A2 contradict somewhat**   | Repo Step 05 correctly collapses to `P_gap_eff=d×pk` because only the product is identifiable; notebook still analyzes raw `d` and `pk`.  |
        | `gs/alpha2` available surface                         |         Not addressed as classifier axis | **B missing**                | Repo stores/uses `gamma_s_eff`, but not the notebook’s `gs/alpha2 → available surface` interpretation.                                    |
        | `dKs_activation_score`                                |         Not found in inspected repo path | **B missing**                | Central notebook measure should become a tested Step 05 extension.                                                                        |
        | Four-window modes: `M0`, `M_rise`, `M_decay`, `M_tot` |                           Not equivalent | **B missing**                | Repo Step 05 is candidate × sweep; notebook is candidate × sweep × window.                                                                |
        | Fv/Fk dictionaries                                    |                            Not addressed | **B missing**                | Repo has K_o summaries but not Fv/Fk variability dictionaries.                                                                            |
        | Phenotype classifier                                  |                            Not addressed | **B missing**                | Repo has mechanism clusters, not named buffering phenotypes.                                                                              |
        | Top-N provisional acceptance                          | Conflicts with reviewer-facing repo path | **A2 contradict somewhat**   | Acceptable for smoke/exploration, not for manuscript claims.                                                                              |
        | Step06 validation pathway                             |                          Already present | **A1 equivalent downstream** | Repo already has the right place to validate phenotype predictions and perturbation robustness.                                           |
        ## Cell-by-cell analysis, condensed
        | Cell | What the notebook does                                                                       | Result observed                                                                                                                                                                                        | Repo comparison                                                                       | Action                                                                                                    |
        | ---: | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
        |    0 | Defines scope, inputs, corrections, literature anchors, outputs.                             | No runtime output.                                                                                                                                                                                     | Partly overlaps Step 05 spec, but adds phenotype-specific measures.                   | Add phenotype-extension section to Step 05 spec.                                                          |
        |    1 | Says pipeline is self-contained and does not import previous scripts.                        | No runtime output.                                                                                                                                                                                     | Contradicts repo modular architecture.                                                | Refactor into `src/phenotype_classifier.py` and notebook wrapper.                                         |
        |    2 | Defines full pipeline: DB loader, ODE, features, windows, modes, classifier, plots, outputs. | Definitions only.                                                                                                                                                                                      | Mixed: equivalent core model; missing phenotype extensions; raw `d/pk` contradiction. | Split into repo modules; avoid duplicated ODE.                                                            |
        |    3 | Explains run profile and selective extraction.                                               | No runtime output.                                                                                                                                                                                     | Useful but legacy-data-specific.                                                      | Put in `analysis/05b_legacy_optuna_buffering_characterization.ipynb`.                                     |
        |    4 | Configures smoke/full profiles, detects zip, extracts selected DB.                           | Smoke used CONTROL 50 nA, top 2, `sim_dt_ms=100`, no threshold CSV.                                                                                                                                    | Repo Step 05 expects Step04 accepted ensembles, not old Optuna DBs.                   | Keep as exploratory loader only.                                                                          |
        |    5 | Run section header.                                                                          | No output.                                                                                                                                                                                             | Equivalent notebook scaffolding.                                                      | No major action.                                                                                          |
        |    6 | Runs `run_pipeline(args)`.                                                                   | 1 DB, 2 successful trials, 2 accepted configs, 8 window rows, provisional top-N.                                                                                                                       | Repo has stricter acceptance contract.                                                | Do not use smoke results as biological conclusions.                                                       |
        |    7 | Inspect outputs section header.                                                              | No output.                                                                                                                                                                                             | Equivalent structure.                                                                 | No major action.                                                                                          |
        |    8 | Displays summary, measure registry, phenotyped rows, M vectors, Fv/Fk variability.           | 13 measure-registry rows, 2 M-vectors, 9 Fv and 9 Fk features.                                                                                                                                         | Most of this is missing from repo.                                                    | Add `measure_registry_status.csv`, `M_mode_vector_by_configuration.csv`, `feature_variability_Fv_Fk.csv`. |
        |    9 | Additional summaries section header.                                                         | No output.                                                                                                                                                                                             | Equivalent structure.                                                                 | No major action.                                                                                          |
        |   10 | Displays phenotype counts, signed flux mode counts, selected score rows.                     | Smoke phenotypes included `largeAvailableSurface_highGJ_but_unrecruited`, `smallN_lowS_lowGJ_GHK_like_local`, `bigN_lowGJ_large_range_weak_coupling`; M_rise was `MIXED_LOCAL` for both smoke configs. | Not present in current Step 05 outputs.                                               | Add optional phenotype-count and signed-mode outputs.                                                     |
        |   11 | Zip/download helper header.                                                                  | No output.                                                                                                                                                                                             | Notebook convenience only.                                                            | Keep outside core repo.                                                                                   |
        |   12 | Optional zip code, disabled.                                                                 | Not executed.                                                                                                                                                                                          | Low priority.                                                                         | Keep as notebook-only helper.                                                                             |
        ## Most important contradictions/refinements
        | Issue                                                                 | Why it matters                                                                                                                                               | Recommendation                                                                                                                                           |
        | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | **Raw `d` and `pk` are separated in the notebook.**                   | Repo Step 05 correctly treats only `P_gap_eff=d×pk` as identifiable and reconstructs with `d=1`, `pk=P_gap_eff`.                                             | In reviewer-facing outputs, keep `P_gap_eff`/`Pkgap` primary. Keep raw `d` only for old Optuna database interpretation, explicitly labelled exploratory. |
        | **Notebook duplicates the model.**                                    | Repo already has canonical model and hidden-output pipeline. Duplication risks silent divergence.                                                            | Notebook should import repo model functions instead of redefining equations.                                                                             |
        | **Top-N accepted = provisional.**                                     | The notebook smoke run accepted top objective-ranked trials because no threshold CSV was available. That is useful for smoke tests, not claims.              | Full biological analysis must use Step04 accepted ensembles or threshold-filtered accepted configs.                                                      |
        | **Phenotype labels sound stronger than current validation supports.** | Repo Step05 is conservative and Step06 is designed to validate mechanisms by held-out prediction, posterior predictive checks, and perturbation robustness.  | Call them “provisional phenotype tags” until Step06 supports them.                                                                                       |
        | **Fk feature names inherit voltage units.**                           | Notebook shows Fk features such as `baseline_mV`; for K_o these should be mM.                                                                                | Rename Fk fields to `baseline_mM`, `rise_slope_mM_per_s`, etc.                                                                                           |
        ## Recommended repo integration
        | Priority | Change                                                                                                                                                    | Where                                                                                           |
        | -------: | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
        |        1 | Add `src/phenotype_classifier.py` with `dKs_activation`, `gs/alpha2`, dynamic conversion, `r_model`, temporal recruitment, and phenotype label functions. | `src/phenotype_classifier.py`                                                                   |
        |        2 | Add `src/windowed_mechanisms.py` for `M0`, `M_rise`, `M_decay`, `M_tot` signed flux budgets.                                                              | `src/windowed_mechanisms.py` or `src/mechanisms.py`                                             |
        |        3 | Add optional Step05 output `accepted_fit_mechanisms_windowed.csv`.                                                                                        | `outputs/mechanisms/`                                                                           |
        |        4 | Add phenotype outputs: `buffering_phenotype_tags.csv`, `phenotype_counts_by_experiment_current_window.csv`, `M_mode_vector_by_configuration.csv`.         | `outputs/mechanisms/`                                                                           |
        |        5 | Preserve legacy Optuna `.db` analysis separately.                                                                                                         | `src/legacy_optuna_loader.py` and `analysis/05b_legacy_optuna_buffering_characterization.ipynb` |
        |        6 | Update Step05 spec with an “exploratory phenotype extension” section, not replacing current conservative mechanism decomposition.                         | `specs_roadmap/step_05_mechanistic_decomposition_spec.md`                                       |
        |        7 | Push phenotype validation into Step06 perturbation and posterior predictive checks.                                                                       | `src/step06_predictive_validation.py` and `outputs/predictive_validation/`                      |
        |        8 | Add tests for `dKs_activation_score ∈ [0,1]`, alpha conversion formula, four-window row count, hidden sanity flags, and no silent failed simulations.     | `tests/bootstrap/` and `tests/integration/`                                                     |
        ## Final recommendation
        Do **not** merge the notebook as a replacement for the current repo Step 05. Merge it as a **Step 05 phenotype-extension layer** plus a **legacy Optuna characterization notebook**.
        The current repo already has the formal Step 05 output structure: `accepted_fit_mechanisms.csv`, `mechanism_clusters.csv`, `representatives.csv`, enrichment, geometry diagnostics, bootstrap stability, claim-scope table, and summary.  It also already computes basic hidden-current/K_o summaries and proxy validity.  
        The attached notebook’s value is the **extra mechanistic phenotype vocabulary**: signed local/spatial fluxes, recruited vs available surface, delayed ionic catch-up, and provisional DH/VH-like buffering tags. Those should become optional, tested outputs under Step 05, then validated under Step 06 before being used as manuscript-level biological phenotypes.
        >>>

3. outputs/executed_notebooks/03_combined_identifiability_profiles_fim.ipynb
    Some structural identifiability can be brought from the new classification/characterization notebook ((unified_astrocyte_K_buffering_characterization_EXECUTED_SMOKE.ipynb))

4. outputs/executed_notebooks/04_cell_specific_six_sweep_fitting_model_aligned.executed.ipynb
    Explain What is the logic of filtering
        Also what are the roles in:
            <<<
            criterion 	scope 	operator 	value 	role
            0 	trace_rmse_mean_mV 	all6 	<= 	18.0 	accepted_by_trace
            1 	weighted_pass_fraction_mean 	all6 	>= 	0.3 	 accepted_by_feature_contract
            2 	heldout_trace_rmse_mV 	leave_one_out 	<= 	20.0 	heldout_screen
            3 	heldout_weighted_pass_fraction 	leave_one_out 	>= 	0.3 	heldout_screen
            4 	ensemble_rank 	all6 	<= 	3.0 	accepted_all6_topk
            5 	holdout_pass_count 	cell 	>= 	3.0 	reviewer_facing_cell
            >>>
    If not the case, We should store:
        Full candidate history (Not persisted; only accepted candidates or best rejected candidate are kept)
    Action: Compare updated filtering logic from unified_astrocyte_K_buffering_characterization_EXECUTED_SMOKE.ipynb directly against the filtering logic used in the Step 02 ATF Threshold script.
    look into confound features ? maybe it has a too high standard / threshold for acceptance ? or it is just optimization time is too short ?

5. outputs/executed_notebooks/05_mechanistic_decomposition.ipynb
    analysis only based on best matching ?
    Accepted ensemble inventory
        What is the logic of filtering ?
    CAN BE IMPROVED for the model proving/reviewer facing (R1 to R7) BY some the characterization/classification notebook (unified_astrocyte_K_buffering_characterization_EXECUTED_SMOKE.ipynb)
        SOLVE the PROBLEM: Kir/gap/leak/K_o decomposition and clustering, but current evidence is insufficient_evidence.
        we have to add its classifier script for finding biologicaly relevant buffering pathways
         Use the measures introduced in classifier script that generates deeper biological description layer for the raw and effective parameters, and mechanistic descriptions that allow for more biologically relevant classification tags for the accepted output. then using the existing perturbation logic to generate a numerical "Biological Description Score" that quantifies how specific design interventions impact the model's inference and predictive ability (bridging to Step 06).
    provide meaning/impact/classification pathway
        long range / ...
        meaningful counterpart in the litterature
        buffering style

6. outputs/executed_notebooks/06_predictive_validation_and_perturbation.ipynb
    Solve those current limitations:
        stimulus_duration_short/long	Intended 0.75x / 1.25x duration	No	Marked unsupported: simulator API lacks timing support
        "Pass/fail uses only K_o peak/recoveryIgnores Vm feature contracts and hidden-current plausibility" => look at the full k_o profile feature for perturbation step
        "Relative perturbations are mildK_o0 ±5% and current drive ±10% are useful but narrow" => just increase range of perturbation (show parameter at the begining of the notebook)
        Is it covered above ?
            For full biological relevance, Step 06 should perturb all six currents, all accepted cells, and all conditions/regions. It should implement real protocol-duration perturbation, add physiological pass/fail criteria such as absolute K_o_peak, K_o_final, Vm recovery, and feature-contract coverage under perturbation, and report robustness stratified by region × condition × mechanism_cluster.

7. outputs/executed_notebooks/07_assumption_sensitivity.ipynb
    needs ECS variant or additional data.for validty

8. 08_parameter_plausibility_and_constrained_reruns.ipynb
    Build a dual-layer parameter constraint system:
        *
        Create (editable if new) or re-use csv table of updated effective parameters, along with the plausible ranges (based on literature, e.g., Hopkins) for raw and effective parameters and their interpretations (I think it should be in a previous step)
    upper_bound is false
        why ?
        correct ?
    constrained vs unconsrained :
        there's no unconstrained, why ? what should we do ?
    add a biological interpretation model to effective parameters (examined here)
    "6. performance tuning" => no idea how to use this

9. create "Step 09 integrating provenance, assumptions, predictions, parameter plausibility, and manuscript-facing tables to provide all materials required to reply to reviewers"
