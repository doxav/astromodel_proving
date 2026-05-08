from __future__ import annotations



def test_step02_pipeline_builds_feature_table_and_outputs(step02_results) -> None:
    results = step02_results
    output_dir = results["paths"].output_dir
    feature_df = results["feature_table_by_sweep"]

    assert len(feature_df) == 222
    assert feature_df["file_id"].nunique() == 37
    assert feature_df.groupby("file_id")["sweep"].nunique().eq(6).all()
    assert set(feature_df["region"]) == {"DH", "VH"}
    assert set(feature_df["condition"]) == {"CONTROL", "MFA", "MFA_BA"}

    for file_name in [
        "feature_table_by_sweep.csv",
        "preprocess_qc_by_sweep.csv",
        "region_condition_cell_counts.csv",
        "condition_region_sweep_thresholds.csv",
        "feature_reliability_weights.csv",
        "region_effect_summary.csv",
        "analysis_summary.json",
    ]:
        assert (output_dir / file_name).exists(), file_name


def test_step02_threshold_scopes_and_index_contract(step02_results) -> None:
    results = step02_results
    thresholds = results["condition_region_sweep_thresholds"]

    scope_counts = thresholds["threshold_scope"].value_counts().to_dict()
    assert scope_counts == {"region_specific": 432, "region_pooled": 216, "global_pooled": 72}

    region_specific = thresholds[thresholds["threshold_scope"] == "region_specific"]
    assert set(region_specific["region"]) == {"DH", "VH"}
    assert set(region_specific["condition"]) == {"CONTROL", "MFA", "MFA_BA"}
    assert set(region_specific["sweep"]) == {1, 2, 3, 4, 5, 6}
    assert set(region_specific["feature"]) >= {
        "peak_depolarization_mV",
        "stim_end_depolarization_mV",
        "return_slope_mV_per_s",
        "plateau_reached",
    }


def test_step02_reliability_weights_capture_redundancy_and_missingness(step02_results) -> None:
    results = step02_results
    reliability = results["feature_reliability_weights"]

    region_specific = reliability[reliability["threshold_scope"] == "region_specific"]
    mean_weights = region_specific.groupby("feature")["reliability_weight"].mean()
    assert mean_weights["stim_end_depolarization_mV"] < mean_weights["peak_depolarization_mV"]
    assert region_specific[region_specific["feature"] == "stim_end_depolarization_mV"]["is_redundant_feature"].all()

    return_by_condition = (
        region_specific[region_specific["feature"] == "return_slope_mV_per_s"]
        .groupby("condition")["reliability_weight"]
        .mean()
    )
    assert return_by_condition["MFA_BA"] < return_by_condition["MFA"] < return_by_condition["CONTROL"]


def test_step02_region_effect_summary_flags_vh_control_small_stratum(step02_results) -> None:
    results = step02_results
    region_effects = results["region_effect_summary"]

    small = region_effects[region_effects["small_stratum"]]
    assert len(small) > 0
    assert set(small["condition"]) == {"CONTROL"}
    assert set(small["sweep"]) == {1, 2, 3, 4, 5, 6}
    assert set(small["feature"]) >= {"peak_depolarization_mV", "plateau_reached", "return_slope_mV_per_s"}
