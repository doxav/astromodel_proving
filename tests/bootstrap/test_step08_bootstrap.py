from __future__ import annotations

import numpy as np

from src.step08_parameter_plausibility import (
    AUDITED_PARAMETERS,
    EFFECTIVE_PARAMETERS,
    Step08Config,
    build_parameter_range_audit,
    default_parameter_ranges,
    load_identifiability_status,
    load_step08_inputs,
)


def test_step08_range_definitions_cover_required_parameters():
    ranges = default_parameter_ranges()
    assert set(AUDITED_PARAMETERS).issubset(set(ranges["parameter"]))
    assert ranges["coordinate_type"].isin({"raw", "effective"}).all()
    assert np.isfinite(ranges[["lower_bound", "upper_bound"]].to_numpy()).all()
    assert (ranges["lower_bound"] < ranges["upper_bound"]).all()


def test_step08_inputs_preserve_identity_prediction_and_mechanism_labels(project_root):
    candidates = load_step08_inputs(project_root, Step08Config(max_candidates=2, write_outputs=False))
    required = {
        "file_id",
        "region",
        "condition",
        "candidate_id",
        "holdout_mean_rmse_mV",
        "holdout_mean_pass_fraction",
        "mechanism_cluster",
        "dominant_mechanism",
    }
    assert required.issubset(candidates.columns)
    assert candidates["region"].notna().all()


def test_step08_parameter_audit_assigns_complete_statuses(project_root):
    cfg = Step08Config(max_candidates=1, write_outputs=False)
    candidates = load_step08_inputs(project_root, cfg)
    audit = build_parameter_range_audit(
        candidates,
        default_parameter_ranges(),
        load_identifiability_status(project_root),
    )
    assert set(AUDITED_PARAMETERS).issubset(set(audit["parameter"]))
    assert audit["plausibility_status"].isin({"within_range", "out_of_range", "missing_value"}).all()
    assert audit["identifiability_status"].isin({"identifiable", "weakly_identified", "effective_only", "not_profiled"}).all()
    assert audit["physiologically_interpretable"].isin({True, False}).all()
    weak_inside = audit[
        audit["plausibility_status"].eq("within_range")
        & audit["identifiability_status"].eq("weakly_identified")
    ]
    if not weak_inside.empty:
        assert not weak_inside["physiologically_interpretable"].astype(bool).any()


def test_step08_effective_parameters_are_configured_as_effective():
    ranges = default_parameter_ranges().set_index("parameter")
    for parameter in EFFECTIVE_PARAMETERS:
        assert ranges.loc[parameter, "coordinate_type"] == "effective"
