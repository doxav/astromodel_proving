# Updated-spec difference analysis for steps 00 and 01

This note compares the updated `reviewer_response_implementation_spec.md` against the previous working draft used to scaffold the first prototype package. It is intentionally limited to the changes that materially affect **step 00** and **step 01**, plus the downstream design contracts those steps must now support.

## Executive summary

The updated specification does not merely reword the original plan. It tightens the scientific contract in four important ways:

1. **Structural non-separability is now explicit before practical sloppiness.**
   The earlier wording moved directly from “Vm-compatible parameter sets” to practical sloppiness/identifiability. The updated wording requires the pipeline to first remove obvious structural non-separabilities before talking about degeneracy.
2. **Brain region is now a hard design factor, not an optional stratification.**
   The updated specification adds a formal `DH`/`VH` experimental-design contract, expected region-by-condition counts, region-aware thresholds, region-aware predictive checks, and explicit warnings against silent pooling.
3. **Step 06 is upgraded from a condition/current validation idea to a true cell-specific six-sweep fit.**
   This changes the role of steps 00 and 01: they now need to produce inputs that preserve `file_id`, `region`, `condition`, and `sweep`, and they must treat the historical single-current DBs only as debug/triage material.
4. **The plan now includes an internal coverage audit and stronger traceability requirements.**
   The updated spec audits its own coverage, adds profile-interpretation classes, cluster-stability/interpolation logic, more gating variants, posterior predictive feature checks, and figure traceability.

These changes are not cosmetic. They materially affect what step 00 and step 01 must output and how their tests should be written.

---

## Difference inventory

## 1. Reframing of the core scientific claim

### Previous draft
The previous draft said the pipeline must first quantify structural non-identifiability and practical sloppiness before reserving the term degeneracy.

### Updated draft
The updated draft says the pipeline must first **remove obvious structural non-separabilities**, then quantify practical identifiability and sloppiness, and only then reserve the term degeneracy.

### Impact on step 00
Step 00 is no longer only a provenance audit. It must preserve enough provenance metadata to prevent accidental interpretation of raw fitted parameters before structural confoundings are handled later.

### Impact on step 01
Step 01 must surface exact effective-parameter identities such as `P_gap_eff = d × pk` and avoid presenting raw historical best fits as interpretable molecular estimates.

---

## 2. Addition of an explicit coverage audit section

The updated specification adds a self-audit table titled **Coverage audit against the current recommendation**.

### Why this matters
This table upgrades several earlier “nice-to-have” items into concrete design constraints. For step 00 and step 01, the most important are:

- region must be treated as a first-class factor;
- single-current historical DBs are provisional/debug only;
- effective parameters remain the primary reporting coordinates;
- step 03 must be framed as a soft structural-inspection plus practical-identifiability screen rather than a formal symbolic proof unless such a proof is truly implemented.

### Implementation consequence
The new step 00 and step 01 specifications must include explicit sections explaining what they do **not** prove yet, to avoid reviewer-facing overclaims.

---

## 3. New region-aware experimental-design contract

This is the largest structural change relevant to step 00/01.

The updated spec adds:

- the meanings of `DH` and `VH`;
- the exact expected `region × condition` cell counts;
- a rule that every downstream table must preserve `file_id`, `region`, `condition`, and `sweep`;
- a rule that primary thresholds and posterior predictive checks must be region-aware;
- a rule that pooled results are secondary and must be labeled as such;
- a rule that the design is unpaired at the cell level and must not be described as paired pharmacology.

### Impact on step 00
Step 00 must now audit ATF filenames and write a machine-readable region inventory. This was absent from the earlier prototype.

### Impact on step 01
Step 01 still works on historical single-current DBs, but its outputs must be clearly labeled as provisional and must not erase the region-aware downstream design contract. This means step 01 cannot become the primary reviewer-facing evidence for regional claims.

---

## 4. Step 00 is expanded beyond historical DB inventory

### Previous draft
Step 00 focused on DB discovery, trace-source discovery, and Control provenance ambiguity.

### Updated draft
Step 00 now also requires:

- ATF filename parsing;
- explicit region inference;
- an ATF region-condition inventory output;
- verification that the 37 ATFs include both regions and all three conditions;
- no silent acceptance of unknown or ambiguous region labels.

### Consequence for code and tests
The step 00 code must include ATF parsing utilities and expected-count validation. The tests must include both historical-fit provenance and ATF-region provenance.

---

## 5. Step 02 becomes explicitly region-aware

This is downstream of step 00, but it changes what step 00 must prepare.

### Previous draft
Step 02 rebuilt thresholds by `condition × region × sweep × feature`, but region was not described as a hard design contract and pooled thresholds were described only as fallback.

### Updated draft
Step 02 now requires:

- primary thresholds indexed by `region × condition × sweep × feature`;
- region-pooled/global-pooled outputs to be explicitly labeled as sensitivity/fallback only;
- explicit region-effect summaries;
- small-stratum warnings, especially VH-control.

### Consequence for step 00
Step 00 must write region inventories in a format that step 02 can consume directly.

---

## 6. Step 03 is reframed as a combined structural-inspection + practical-identifiability workflow

### Previous draft
Step 03 was titled more narrowly around effective-parameter identifiability, FIM, and profile likelihood.

### Updated draft
Step 03 now explicitly combines:

- equation-level structural inspection,
- exact invariance demonstrations,
- an `effective_parameter_map.csv`,
- profile-interpretation classes,
- FIM after reparameterization.

### Consequence for step 01
Step 01 must produce cleaner effective-parameter outputs because step 03 will consume them. The step 01 tests and notebook should therefore verify that effective combinations are exported in normalized form.

---

## 7. Step 04 adds geometry/stability logic

The updated spec strengthens step 04 with:

- bootstrap cluster stability,
- interpolation tests between candidate modes,
- a rule that continuous accepted sets must be called compensation manifolds rather than degeneracy.

### Consequence for step 01
Step 01 mechanism summaries must be framed as provisional diagnostics for later ensemble analyses, not as final evidence of distinct mechanisms.

---

## 8. Step 05 expands gating-family sensitivity

The updated spec adds:

- hard-threshold gating,
- double-sigmoid gating,
- a requirement that gating families be separate runs rather than a mixed categorical search inside one inference job.

### Consequence for step 01
Step 01 should decode and report the historical `switching_function`, but it must not treat mixed historical DB categories as a fair model-comparison result.

---

## 9. Step 06 is materially upgraded

### Previous draft
The earlier working draft still allowed the main validation narrative to lean on condition/current-level multi-current validation and left full cell-specific six-sweep fitting partly in the backlog.

### Updated draft
Step 06 is now explicit:

- one shared biological/effective parameter set per **cell**,
- six sweeps per cell,
- region preserved throughout,
- leave-one-sweep-out rotation,
- population-level posterior predictive feature checks by `region × condition × sweep`.

### Consequence for step 00
Step 00 must preserve cell-level provenance and region metadata from the ATF source.

### Consequence for step 01
Step 01 historical DB analysis is now explicitly secondary. It remains useful for debugging, invariance proofs, and hidden-current readouts, but not as the final reviewer-facing fitting result.

---

## 10. Step 07 is refined with interpretability classes

The updated spec distinguishes:

- `within_range`,
- `identifiable`,
- `physiologically_interpretable`.

This matters because a parameter can lie within bounds and still be weakly identified.

### Consequence for step 01
Step 01 must avoid presenting raw best-fit values as automatically interpretable. Its outputs should prefer effective-parameter summaries and leave physiological interpretation to later constrained analyses.

---

## 11. Step 08 and Step 09 become more explicitly region-aware and traceable

The updated spec now requires:

- perturbation summaries to report DH and VH separately before pooled results;
- figure traceability tables mapping each reviewer-facing item back to critique IDs and source outputs.

### Consequence for step 00/01
Their outputs must be machine-readable and stable enough to support later figure traceability.

---

## What changed concretely for implementation of step 00 and step 01

## Step 00 — required upgrades

Compared with the earlier prototype, the new step 00 must now:

1. audit ATF files, not just DBs and trace CSVs;
2. parse `DH` and `VH` explicitly;
3. write region-condition inventories and counts;
4. keep Control provenance ambiguity explicit;
5. write outputs that later steps can join with region-aware feature and fit tables.

## Step 01 — required upgrades

Compared with the earlier prototype, the new step 01 must now:

1. export normalized effective parameters as primary outputs;
2. be explicit that the historical DBs are provisional and single-current only;
3. provide hidden-current/flux summaries in a form reusable by step 04;
4. avoid presenting mixed historical gating results as a valid model-comparison result;
5. support later step 03 structural-inspection/profile/FIM workflows.

---

## Bottom line

The updated specification tightens the scientific discipline of the entire implementation plan. For the first two steps, the main changes are:

- **step 00 becomes both a historical-fit provenance audit and a region-aware ATF-design audit**;
- **step 01 becomes a post-fit mechanism/effective-parameter bridge that must remain explicitly provisional**.

This is the correct direction. It reduces the risk of reviewer-facing overclaiming while preserving the immediate practical value of the historical DBs for debugging, structural-invariance demonstrations, and hidden-current readouts.
