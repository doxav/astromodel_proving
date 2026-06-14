from __future__ import annotations

from src.reviewer_gate_audits import (
    ReviewerGateAuditConfig,
    run_reviewer_gate_audits,
)


def test_reviewer_gate_audits_write_selected_action_outputs(project_root):
    config = ReviewerGateAuditConfig(
        max_candidates=3,
        write_outputs=False,
        all_current_time_points=20,
        interpolation_time_points=20,
        interpolation_points=3,
    )
    result = run_reviewer_gate_audits(project_root, config)

    required = {
        "phenotype_robustness_summary",
        "stratum_support_gate",
        "prediction_limited_failure_modes",
        "assumption_gate_audit",
        "proxy_exclusion_claim_sensitivity",
        "parameter_semantics_audit",
        "full_accepted_parameter_audit",
        "parameter_interpretation_class_audit",
        "constrained_failure_modes",
        "integrated_degeneracy_gate_matrix",
        "degeneracy_level_table",
        "restricted_validation_claims",
        "restricted_all_gate_join",
        "K_o_homeostasis_endpoint_audit",
        "phenotype_threshold_sensitivity",
        "all_current_assumption_sensitivity",
        "cell_specific_identifiability_audit",
        "parameter_ranges_citation_audit",
        "selected_action_strategy_comparison",
        "selected_action_scientific_value_assessment",
        "selected_action_results_summary",
    }
    assert required.issubset(result)
    assert not result["assumption_gate_audit"].empty
    assert {"assumption_axis", "gate_pass", "gate_status"}.issubset(
        result["assumption_gate_audit"].columns
    )
    assert not result["parameter_interpretation_class_audit"].empty
    assert "claim_class_after_semantic_filter" in result["parameter_interpretation_class_audit"]
    assert not result["restricted_all_gate_join"].empty
    assert "blocking_axes" in result["restricted_all_gate_join"]


def test_reviewer_gate_audits_keep_final_degeneracy_blocked_when_any_gate_fails(project_root):
    result = run_reviewer_gate_audits(
        project_root,
        ReviewerGateAuditConfig(
            max_candidates=3,
            write_outputs=False,
            all_current_time_points=20,
            interpolation_time_points=20,
            interpolation_points=3,
        ),
    )
    join = result["restricted_all_gate_join"]
    assert "restricted_degeneracy_claim_allowed" in join.columns
    if not result["assumption_gate_audit"]["gate_pass"].astype(bool).all():
        assert not join["restricted_degeneracy_claim_allowed"].astype(bool).any()
