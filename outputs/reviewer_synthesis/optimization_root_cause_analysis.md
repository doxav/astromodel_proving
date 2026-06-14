# Optimization Root-Cause Analysis for Unsupported Claims

## Bottom Line

More Step 04 optimization is a real root cause for sparse VH MFA coverage, but it is not a strong global root cause for the unsupported claims. After adding strict ODE solver-warning handling, targeted high-budget Step 04 reruns upgraded all 7 VH MFA cells to reviewer-facing status. The same strict high-budget screen did not rescue VH CONTROL: it remains 0/4 reviewer-facing and feature-level failures concentrate in stim-end/rise/decay timing rather than trace RMSE. Targeted Step 04 reruns are therefore justified for VH MFA coverage, but they do not solve biological mechanism interpretation, model-assumption robustness, parameter physiological interpretability, or final biological degeneracy wording.

For Step 06, the analogous question is not Optuna/SciPy length because Step 06 does not optimize fits. It re-simulates selected Step 04 candidates and scores held-out, PPC, and perturbation gates. Higher time-resolution reruns did not improve support in the earlier sensitivity. The relevant Step 06 scope sensitivity is candidate selection: best-per-cell gives 3 predictive-supported and 5 prediction-limited rows, top-3-per-cell gives 8 supported and 5 limited rows, and mechanism-diverse top-3 gives 10 supported and 5 limited rows. This improves screening support with candidate diversity, but final biological degeneracy wording remains blocked.

## Step 04 Evidence

- Canonical Step 04 after targeted merge: 37 cells, 6230 candidates, 2110 accepted candidates, 33 reviewer-facing cells.
- Not search-limited strata at acceptance level: DH MFA, DH MFA_BA, VH MFA_BA have mean accepted fractions near 0.95-0.99.
- Sparse/problem stratum after targeted rerun: VH CONTROL has 0/4 reviewer-facing cells; VH MFA is now 7/7 reviewer-facing.
- Non-reviewer-facing failures are mostly feature-contract/held-out feature failures, often with low trace RMSE; this points to objective/contract/generalization limits, not only raw runtime.

### High-Budget Probe

Separate output directory: `outputs/reviewer_synthesis/step04_high_budget_probe_vh_failures/`.

| Cell | Accepted Canonical -> Probe | Heldout Passes Canonical -> Probe | Reviewer-Facing Result |
| --- | ---: | ---: | --- |
| 3_VH_1_CONTROL | 0 -> 0 | 1 -> 0 | still_not_reviewer_facing |
| VH_3_MFA | 6 -> 26 | 2 -> 5 | upgraded_to_reviewer_facing |
| VH_4_MFA | 0 -> 3 | 0 -> 4 | upgraded_to_reviewer_facing |
| VH_6_MFA | 2 -> 1 | 0 -> 0 | still_not_reviewer_facing |

### Strict Numerical-Health Targeted Reruns

Separate output directories:
- `outputs/reviewer_synthesis/step04_targeted_vh_high_budget_numerical_health_v2/`
- `outputs/reviewer_synthesis/step04_vh6_mfa_intensified_numerical_health/`

Final targeted comparison: `outputs/reviewer_synthesis/step04_targeted_best_comparison.csv`.

| Stratum | Canonical reviewer-facing | Targeted reviewer-facing | Interpretation |
| --- | ---: | ---: | --- |
| VH MFA | 0/7 | 7/7 | Strong local evidence that the original Step 04 run was under-searching this stratum. |
| VH CONTROL | 0/4 | 0/4 | Not rescued by strict high-budget search; feature-level bottleneck persists. |

Interpretation: the strict reruns falsify the claim that Step 04 budget never matters. They also falsify the stronger claim that more Step 04 budget is the global fix. VH MFA was fit/search limited; VH CONTROL and downstream biological claims are not resolved by more unconstrained Step 04 runtime.

## Step 06 Evidence

- Current Step 06 uses `candidate_policy=best_per_cell`, 35 candidates, all six currents, 280 heldout rows, and 1890 perturbation rows.
- Every current robustness group has holdout pass fraction 1.0; prediction-limited labels are caused by perturbation functional robustness/PPC limits, not held-out fit failure.
- Time-resolution sensitivity:
  - 50 time points: 5 supported, 5 limited, mean biological score 0.657.
  - 80 time points: 5 supported, 5 limited, mean biological score 0.653.
  - 120 time points: 4 supported, 6 limited, mean biological score 0.650.
- Candidate-scope sensitivity:
  - best per cell: 35 candidates, 3 predictive-supported rows, 5 prediction-limited rows.
  - top-3 per cell: 99 candidates, 8 predictive-supported rows, 5 prediction-limited rows.
  - mechanism-diverse top-3 per cell: 84 candidates, 10 predictive-supported rows, 5 prediction-limited rows.

Interpretation: Step 06 should not be treated as under-run in the optimizer sense. More time points do not rescue the unsupported groups. Candidate diversity improves screen support, so the reviewer-facing interpretation should report both the conservative best-per-cell result and the expanded-scope sensitivity; it still must not claim final biological degeneracy.

## Claim-Level Conclusions

### candidate mechanism regimes are biologically interpretable
- Conclusion: yes_for_local_VH_MFA_coverage; unlikely_as_primary_for_biological_interpretability
- Confidence: high_for_local_coverage_medium_high_for_not_global
- Evidence: Step04 has 2110/6230 accepted candidates and 33/37 reviewer-facing cells after targeted VH MFA replacement. Step06 unsupported groups all have holdout_pass_fraction=1.0; prediction-limited labels are perturbation/PPC limited, not fit-heldout limited. Candidate-scope sensitivity improves predictive-supported rows from 3/8 to 10/15 but does not remove all limitations.
- Recommended next test: Report VH MFA as upgraded by targeted high-budget strict numerical-health refits; treat VH CONTROL as a remaining model/feature mismatch; do not expect this alone to authorize biological pathway wording.

### model assumptions do not drive the conclusion
- Conclusion: no
- Confidence: high
- Evidence: Step07 blockers are assumption axes (gating_form; intracellular_K_as_ECS_proxy); these require alternative model forms/ECS/proxy tests under same data, not more iterations of the same Step04 objective.
- Recommended next test: Implement assumption-specific refit/sensitivity analyses rather than increasing the current Step04 search budget globally.

### accepted parameters are physiologically interpretable
- Conclusion: unlikely; constrained/refit-prior formulation is the relevant test
- Confidence: high
- Evidence: Step08 blocks best-per-cell candidates from parameter claims; parameter rows include weak identifiability/out-of-range flags. More unconstrained search optimizes Vm fit, not physiological identifiability.
- Recommended next test: Run constrained or penalized Step04 refits for selected cells; do not spend on unconstrained global trial count as the first parameter-plausibility fix.

### final biological degeneracy wording is allowed
- Conclusion: no_as_global_claim; at_most_a_minor_upstream_coverage_factor
- Confidence: high
- Evidence: Final claim is blocked by the conjunction of mechanism distinction, Step06 perturbation/PPC support, Step07 assumptions, Step08 parameters, and homeostasis endpoint gates. Step04 fit coverage is only one upstream prerequisite. Targeted high-budget strict refits upgraded VH MFA but not VH CONTROL; Step06 candidate-scope sensitivity improves support but still leaves prediction-limited groups.
- Recommended next test: Use integrated gate matrix to identify whether any restricted stratum is fit-limited before spending on full Step04 reruns.

## Practical Recommendation

1. Do not run a blind full Step 04 massive optimization as the first global fix.
2. Keep the targeted high-budget Step 04 VH MFA replacement in canonical evidence; do not promote VH CONTROL without model/feature mismatch work.
3. Report Step 06 candidate-scope sensitivity after targeted Step 04 reruns: best-per-cell versus top-k-per-cell/mechanism-diverse representatives.
4. Keep assumption and parameter claims blocked until Step 07/08-specific fixes are done; more unconstrained Step 04 runtime is not an adequate substitute.
5. Treat ODE warning frequency in high-budget probes as a signal to add numerical-health diagnostics before scaling global optimization.
