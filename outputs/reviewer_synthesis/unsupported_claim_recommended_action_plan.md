# Unsupported Claim Recommended Action Plan

Source detail table: `outputs/reviewer_synthesis/unsupported_claim_gap_action_table.csv`.

## Current unsupported claims

- `candidate_regime_screen` - candidate mechanism regimes are biologically interpretable: 6 gap/action rows, 3 priority-1 rows (`MECH-02; MECH-03; MECH-05`).
- `model_dependent_or_unresolved` - model assumptions do not drive the conclusion: 6 gap/action rows, 2 priority-1 rows (`ASSUMP-02; ASSUMP-06`).
- `partial` - accepted parameters are physiologically interpretable: 7 gap/action rows, 4 priority-1 rows (`PARAM-01; PARAM-05; PARAM-06; PARAM-07`).
- `not_allowed_yet` - final biological degeneracy wording is allowed: 7 gap/action rows, 5 priority-1 rows (`FINAL-01; FINAL-02; FINAL-03; FINAL-04; FINAL-05`).

## Quick-win focus

### candidate mechanism regimes are biologically interpretable
- First gap IDs: MECH-02; MECH-03; MECH-05
- Expected first artifacts: outputs/predictive_validation/phenotype_robustness_summary.csv; outputs/reviewer_synthesis/stratum_support_gate.csv; outputs/predictive_validation/prediction_limited_failure_modes.csv
- Gate logic: Phenotype claim only if phenotype groups have sufficient cells and pass held-out, PPC, perturbation, Vm-feature, and hidden-flux gates. | Region/condition mechanism claim requires predefined minimum reviewer-facing cells and predictive-supported groups. | A group can be upgraded only after failing features/perturbations are either fixed, justified, or excluded by predeclared criteria. | Main text may say 'model-derived buffering scenario' only; 'pathway/phenotype' requires external validation or perturbation-specific support.
- Fallback: Keep phenotype tags as descriptive/provisional supplements. | Report only cell-level associations and identify sparse strata as limitations. | Restrict claims to predictive_supported groups only. | Use technical model terminology only.

### model assumptions do not drive the conclusion
- First gap IDs: ASSUMP-02; ASSUMP-06
- Expected first artifacts: outputs/assumption_sensitivity/proxy_exclusion_claim_sensitivity.csv; later explicit_ecs_variant_screen.csv; outputs/reviewer_synthesis/assumption_gate_audit.csv
- Gate logic: Assumption support improves if conclusions hold after excluding proxy-limited candidates; full support needs explicit ECS variant or external data. | Assumption claim supported only if all configured axes meet gate thresholds.
- Fallback: State model conclusions require the intracellular-K proxy and cannot be generalized to ECS homeostasis. | Report exact failing axis and keep final claim blocked.

### accepted parameters are physiologically interpretable
- First gap IDs: PARAM-01; PARAM-05; PARAM-06; PARAM-07
- Expected first artifacts: outputs/parameter_plausibility/parameter_semantics_audit.csv; outputs/parameter_plausibility/full_accepted_parameter_audit.csv; outputs/parameter_plausibility/parameter_interpretation_class_audit.csv; outputs/parameter_plausibility/constrained_failure_modes.csv
- Gate logic: Only parameters with defensible units/ranges and identifiability can be biologically interpreted; phenomenological parameters are excluded from physiological claims. | Best-per-cell parameter conclusions should match full-ensemble distribution or be explicitly scoped. | Physiological parameter claim excludes coordinates declared phenomenological before analysis; guardrail violations remain limitations. | Candidate-level parameter claim allowed only if prediction and mechanism persist under constraints.
- Fallback: Keep all raw parameter claims downgraded and report effective coordinates only. | State Step08 is a best-per-cell screen only. | Keep zth/zs as blockers and avoid physiological parameter claim. | Exclude failing candidates from parameter interpretation claims.

### final biological degeneracy wording is allowed
- First gap IDs: FINAL-01; FINAL-02; FINAL-03; FINAL-04; FINAL-05
- Expected first artifacts: outputs/reviewer_synthesis/integrated_degeneracy_gate_matrix.csv; outputs/reviewer_synthesis/degeneracy_level_table.csv; outputs/reviewer_synthesis/restricted_validation_claims.csv; outputs/reviewer_synthesis/restricted_all_gate_join.csv; outputs/predictive_validation/K_o_homeostasis_endpoint_audit.csv
- Gate logic: Final or restricted degeneracy claim only where every required row-level gate is supported. | Degeneracy level cannot exceed the weakest required evidence layer. | Only predictive_supported groups can be candidates for restricted final wording after other gates pass. | Restricted final claim requires predictive_supported + assumption gate pass + parameter gate pass for each contributing candidate/group. | Homeostasis claim requires K_o peak/recovery/final robustness, not only Vm feature support. | Every claim in the response must cite a source artifact and maturity status.
- Fallback: Keep global final claim blocked and list exact blockers. | Use compensation/non-identifiability terminology only. | No group-level degeneracy wording; maintain mechanism-screen language. | Prioritize assumption and parameter fixes before more mechanism work. | Use Vm-predictive mechanism language, not K_o homeostasis language. | Remove or downgrade unsupported manuscript claims.

## Phased plan

### Phase 1: Immediate current-output gates
- Scope: all unsupported claims
- Gap IDs: MECH-02; MECH-03; MECH-05; ASSUMP-06; ASSUMP-02; PARAM-01; PARAM-05; PARAM-06; PARAM-07; FINAL-01; FINAL-02; FINAL-03; FINAL-04; FINAL-05; FINAL-06
- Action: Produce the no-refit audits and guardrail tables first: phenotype robustness, stratum support, prediction-limited failures, assumption gates, proxy-exclusion sensitivity, parameter semantic/full-ensemble/failure audits, and integrated degeneracy gates.
- Decision gate: If any restricted stratum passes predictive, assumption, parameter, and K_o/homeostasis gates, Step09 may upgrade only that restricted claim wording; otherwise global biological degeneracy remains blocked.
- Fallback: Keep candidate-screen, model-dependent, and partial-parameter wording; report exact failing gates by claim and stratum.

### Phase 2: Modest code sensitivity upgrades
- Scope: mechanism, assumptions, and parameters
- Gap IDs: MECH-01; MECH-06; ASSUMP-04; PARAM-02; PARAM-04
- Action: Add interpolation/threshold sensitivity, all-current assumption sensitivity, cell-specific identifiability diagnostics, and literature/basis audit for parameter ranges.
- Decision gate: Upgrade mechanism and parameter wording only if labels are stable, separated or explicitly continuous, assumptions remain stable across all currents, and parameters are both range-supported and identifiable.
- Fallback: Use compensation-manifold or continuous-score wording; keep physiological parameter claims restricted to cited and identifiable coordinates.

### Phase 3: Refit and structural assumption tests
- Scope: assumptions and parameter plausibility
- Gap IDs: ASSUMP-01; PARAM-03; ASSUMP-03; ASSUMP-05
- Action: Run constrained Step04 refits, alternative-gating refit screens, spatial surrogate screens, and omitted-process sensitivity where earlier gates remain promising.
- Decision gate: Stronger biological mechanism wording requires refit-level robustness to accepted alternative model forms and constrained physiological parameter ranges.
- Fallback: Declare the current model a minimal projection and avoid claims that assumptions do not drive the conclusion.

### Phase 4: Prospective external validation
- Scope: final biological degeneracy and pathway claims
- Gap IDs: FINAL-07
- Action: Add a prospective validation table with discriminating observables for each model-derived phenotype/regime and classify them as future tests unless current data already contain the observable.
- Decision gate: Broad biological/pathway degeneracy requires independent measured observables or explicitly prospective wording.
- Fallback: Frame regimes as model-generated hypotheses and keep the manuscript response focused on validated restricted claims.

## Recommended stopping rules

- Do not upgrade global biological degeneracy unless the integrated gate matrix passes across mechanism distinction, predictive perturbation support, assumption robustness, parameter plausibility, and homeostasis endpoint evidence.
- If only a subset of strata pass, use restricted stratum-level wording and list blocked strata explicitly.
- If mechanism labels are continuous or unstable, use compensation-manifold or model-derived scenario wording rather than biological regime wording.
- If parameters remain weakly identified or phenomenological, report effective-coordinate behavior and keep physiological parameter claims partial or blocked.
- If the ECS/proxy and omitted-process assumptions remain untested, keep final homeostasis and broad biological degeneracy claims prospective.
