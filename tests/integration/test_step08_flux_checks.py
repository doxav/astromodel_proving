from __future__ import annotations

from src.step08_parameter_plausibility import (
    Step08Config,
    run_step08_parameter_plausibility,
)


def test_step08_constrained_rerun_computes_flux_and_mechanism_change(project_root):
    """Ensure that the constrained rescore tests actual mechanistic persistence via flux fractions."""
    result = run_step08_parameter_plausibility(
        project_root,
        Step08Config(
            max_candidates=1,
            constrained_max_candidates=1,
            time_points=15,
            write_outputs=False,
        ),
    )
    constrained = result["constrained_rerun_comparison"]

    required_cols = {
        "dominant_mechanism_unconstrained",
        "dominant_mechanism_constrained",
        "mechanism_flux_fraction_delta_max",
        "mechanism_changed_under_constraints",
    }
    assert required_cols.issubset(constrained.columns)

    ok_runs = constrained[constrained["simulation_status"] == "ok"]
    if not ok_runs.empty:
        assert ok_runs["mechanism_flux_fraction_delta_max"].notna().all()
        assert ok_runs["dominant_mechanism_constrained"].isin({"gap", "kir", "leak", "Mixed"}).any()
