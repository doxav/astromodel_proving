# Step 04 Phenotype-Guided Alternative Fit Analysis

## Question

Can the phenotype tags from the characterization/classification work help Step 04 find new alternative fits or mutually exclusive pathways?

## Tag Sources

The current phenotype vocabulary is implemented in `src/phenotype_classifier.py` and written by Step 05, not by Step 04 itself. The current outputs contain these candidate-level tags:

- `available_surface_voltage_coupled_but_ionic_recruitment_low`
- `kir_dominant_local_buffering`
- `low_recruitment_local_storage`
- `mixed_local_spatial_buffering`
- `recruited_surface_gap_assisted_buffering`

The code also defines `long_range_recruited_spatial_buffering`, but that label is not present in the current Step 05 accepted-candidate outputs.

The tags are derived from continuous mechanism scores:

- `dKs_activation_score`
- `long_range_distribution_fraction`
- `voltage_coupling_score`
- `kir_current_score`
- `alpha2_available_surface_proxy`
- `recruited_surface_alpha2_x_A_dKs`

This is important: the continuous scores are the better search/selection features. The labels are useful summaries, but several are threshold-sensitive.

## Evidence Files

- Tag counts by stratum: `outputs/reviewer_synthesis/step04_phenotype_tag_counts_by_stratum.csv`
- Per-cell tag diversity: `outputs/reviewer_synthesis/step04_phenotype_diversity_by_cell.csv`
- Per-stratum tag diversity: `outputs/reviewer_synthesis/step04_phenotype_diversity_by_stratum.csv`
- Tag co-occurrence: `outputs/reviewer_synthesis/step04_phenotype_pair_cooccurrence_by_cell.csv`
- Threshold stability: `outputs/reviewer_synthesis/step04_phenotype_threshold_stability_by_label.csv`
- Step 06 policy support by tag: `outputs/reviewer_synthesis/step04_phenotype_step06_support_by_policy.csv`

## Main Findings

Phenotype diversity is strong in the high-support treatment strata but weak in controls:

| Region | Condition | Cells | Cells with 2+ phenotypes | Median phenotypes | Cells with 2+ mechanism clusters |
| --- | --- | ---: | ---: | ---: | ---: |
| DH | CONTROL | 7 | 2 | 1.0 | 6 |
| DH | MFA | 6 | 6 | 5.0 | 6 |
| DH | MFA_BA | 6 | 6 | 5.0 | 6 |
| VH | CONTROL | 2 accepted cells | 1 | 1.5 | 0 |
| VH | MFA | 7 | 4 | 2.0 | 5 |
| VH | MFA_BA | 7 | 7 | 5.0 | 7 |

Tag stability is uneven under threshold perturbation:

| Baseline tag | Stable fraction | Interpretation |
| --- | ---: | --- |
| `recruited_surface_gap_assisted_buffering` | 1.000 | Stable enough for reporting and soft selection. |
| `available_surface_voltage_coupled_but_ionic_recruitment_low` | 0.945 | Stable enough for reporting and soft selection. |
| `mixed_local_spatial_buffering` | 0.859 | Mostly stable, but still a broad residual class. |
| `kir_dominant_local_buffering` | 0.430 | Too threshold-sensitive for hard optimization constraints. |
| `low_recruitment_local_storage` | 0.296 | Too threshold-sensitive for hard optimization constraints. |

The tag pairs are not mutually exclusive in the practical sense needed for sequential optimization. All major tag pairs co-occur within accepted alternatives in many cells. For example, `available_surface_voltage_coupled_but_ionic_recruitment_low` co-occurs with `kir_dominant_local_buffering` in 22 cells, and `low_recruitment_local_storage` co-occurs with `mixed_local_spatial_buffering` in 20 cells.

## Step 06 Relevance

The strongest existing test is not a new optimizer; it is candidate-scope sensitivity in Step 06. Mechanism-diverse candidate selection improves support relative to best-per-cell:

| Step 06 candidate policy | Candidates | Supported rows | Prediction-limited rows | Mean biological score | Mean perturbation robust fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best_per_cell` | 35 | 3 | 5 | 0.617 | 0.413 |
| `top_k_per_cell` | 99 | 8 | 5 | 0.648 | 0.522 |
| `mechanism_diverse_per_cell` | 84 | 10 | 5 | 0.664 | 0.557 |

This supports using mechanism/phenotype diversity in selection. It does not support hard phenotype claims yet, because final degeneracy wording remains blocked by Step 06 limitations plus Step 07/08/09 gates.

## Tested Approaches and Recommendation

### Best current approach: post-hoc continuous novelty selection

Use Step 04 to generate a large accepted pool, then select alternatives per cell using maximin distance in effective and mechanism-score space. This is already closest to the existing `mechanism_diverse_per_cell` policy and has the best Step 06 sensitivity result.

Recommended feature axes:

- log effective coordinates: `P_gap_eff`, `gamma_t_eff`, `gamma_s_eff`, `volume_ratio_wa_wo`
- Step 05 continuous scores: activation, long-range fraction, voltage coupling, Kir score, recruited surface
- optional stable tags as reporting labels, not primary distance variables

### Secondary approach: soft phenotype novelty after acceptance

After all-six acceptance, prefer candidates whose stable phenotype label or continuous-score region is underrepresented for that cell. This can help reviewers see alternative pathways without letting weak labels drive the fit objective.

Use stable labels only as soft tie-breakers:

- `recruited_surface_gap_assisted_buffering`
- `available_surface_voltage_coupled_but_ionic_recruitment_low`
- `mixed_local_spatial_buffering`, with caution

Avoid hard penalties on:

- `kir_dominant_local_buffering`
- `low_recruitment_local_storage`

Those labels are too threshold-sensitive in the current audit.

### Not recommended now: phenotype penalty inside every Optuna trial

Putting phenotype scoring directly inside Step 04 Optuna would require hidden-state simulation and windowed phenotype extraction for every trial. That is expensive and would couple Step 04 fitting to provisional Step 05 labels. It also risks optimizing for a threshold artifact rather than a robust pathway.

## Conclusion

Phenotype tags should help Step 04 downstream selection, not the primary Step 04 fitting objective. The best practical design is:

1. Keep Step 04 focused on full-scope all-six acceptance and heldout support.
2. Select 2+ alternatives per cell by effective-space and continuous mechanism-score novelty.
3. Use stable phenotype labels to annotate and sanity-check selected alternatives.
4. Use Step 06 to decide whether any selected mechanism/phenotype regime survives prediction and perturbation.

The idea of mutually exclusive phenotype tags is not supported by the current evidence. The data support continuous novelty and stable-label coverage, not hard exclusion penalties.
