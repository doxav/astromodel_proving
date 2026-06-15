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
- `outputs/legacy_mechanisms/`
- `outputs/predictive_validation/`
- `outputs/legacy_perturbation/`
- `outputs/assumption_sensitivity/`
- `outputs/parameter_plausibility/`

## Outputs

All outputs are written under `outputs/reviewer_synthesis/`.

| File | Purpose |
|---|---|
| `reviewer_traceability_table.csv` | One row per R1-R7 critique with evidence status, source outputs, and claim boundary. |
| `claim_maturity_table.csv` | Claim-level maturity table separating supported, partial, unresolved, and not-allowed claims. |
| `reviewer_remark_artifact_links.csv` | Ordered R1-R7 links to generated upstream artifacts, notebooks, and stable cell references. |
| `mechanistic_pathway_perturbation_gate.csv` | Gate table for FV-to-FK and perturbation-direction pathway claims. |
| `legacy_perturbation_claim_gate.csv` | Gate table for source-scoped legacy perturbation claims and remaining blockers. |
| `degeneracy_scientific_value_statement.csv` | Objective statement of what the current evidence supports and what remains required before a biological degeneracy claim. |
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
5. Legacy mechanism and perturbation outputs may support source-scoped
   mechanistic-pathway hypotheses, but they do not by themselves prove
   cell-specific biological degeneracy.
6. R1-R7 artifact links must point to upstream notebook sections/cell references
   and machine-readable CSV/JSON/PDF artifacts in order of reviewer usefulness.

## Notebook contract

`analysis/09_reviewer_response_synthesis.ipynb` must run from repository root,
write the CSV tables and summary JSON, display R1-R7 status, display ordered
artifact links, and state the current objective claim boundary from the actual
outputs.
