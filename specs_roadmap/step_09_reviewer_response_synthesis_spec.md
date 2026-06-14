# Step 09 - Reviewer-response synthesis and manuscript-facing traceability

## Purpose

Step 09 integrates outputs from Steps 00-08 into reviewer-facing traceability
tables. It does not create new biological evidence. Its role is to prevent
over-claiming by making each R1-R7 answer traceable to source artifacts,
claim-maturity status, and remaining requirements.

## Inputs

- `outputs/provenance/`
- `outputs/features/`
- `outputs/identifiability/`
- `outputs/cell_fits/`
- `outputs/mechanisms/`
- `outputs/predictive_validation/`
- `outputs/assumption_sensitivity/`
- `outputs/parameter_plausibility/`

## Outputs

All outputs are written under `outputs/reviewer_synthesis/`.

| File | Purpose |
|---|---|
| `reviewer_traceability_table.csv` | One row per R1-R7 critique with evidence status, source outputs, and claim boundary. |
| `claim_maturity_table.csv` | Claim-level maturity table separating supported, partial, unresolved, and not-allowed claims. |
| `manuscript_asset_manifest.csv` | Source artifact manifest for reviewer response tables and figures. |
| `analysis_summary.json` | Counts, missing artifacts, and final degeneracy claim status. |

## Scientific contract

1. Step 09 may only upgrade a claim when the corresponding upstream outputs
   support it.
2. Final biological degeneracy wording remains disabled unless mechanism
   distinction, predictive/perturbation support, assumption robustness, and
   parameter plausibility are all supported.
3. Provisional phenotype tags from Step 05 remain provisional unless Step 06
   validates them.
4. Missing artifacts are reported in the manifest and summary, not ignored.

## Notebook contract

`analysis/09_reviewer_response_synthesis.ipynb` must run from repository root,
write the three CSV tables and summary JSON, display R1-R7 status, and state the
current objective claim boundary from the actual outputs.
