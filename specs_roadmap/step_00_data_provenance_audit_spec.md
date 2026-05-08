# Step 00 spec update

## What changed
- `CONTROL_TRACES_old.csv` is removed and must not be used.
- Step 00 now emphasizes the distinction between legacy historical DB/CSV assets and the new ATF dataset.
- Historical control objectives remain `unresolved` under the documented `CONTROL_TRACES.csv` source.

## Required outputs
- `outputs/provenance/db_study_summary.csv`
- `outputs/provenance/trace_source_summary.csv`
- `outputs/provenance/control_trace_verification.csv`
- `outputs/provenance/atf_region_condition_inventory.csv`
- `outputs/provenance/atf_region_condition_counts.csv`
- `outputs/provenance/data_source_contract.csv`

## Acceptance criteria
- 18 DB rows are reported.
- Only `CONTROL_TRACES.csv`, `MFA_TRACES.csv`, and `BARIUM_TRACES.csv` appear as legacy trace sources.
- 37 ATF files are inventoried.
- DH/VH × condition counts match the expected contract.
- Control DB rows remain explicit and unresolved rather than silently promoted.
