from __future__ import annotations

import numpy as np

from src.step05_mechanistic_decomposition import (
    Step05Config,
    kbath_middle_for_current,
    load_step04_accepted_ensemble,
    reconstruct_flat_params,
    run_step05_mechanistic_decomposition,
)
from src.astro_model import CURRENT_DICT_K_BATH_VALUES, build_paramdict


def test_step05_loads_step04_ensemble_with_region_condition_contract(project_root):
    ensemble, source = load_step04_accepted_ensemble(project_root, max_candidates=2)
    assert source.exists()
    assert {
        "file_id",
        "region",
        "condition",
        "candidate_id",
        "P_gap_eff",
        "gamma_t_eff",
        "gamma_s_eff",
        "volume_ratio_wa_wo",
    }.issubset(ensemble.columns)
    assert ensemble["accepted"].all()
    assert set(ensemble["region"]).issubset({"DH", "VH"})


def test_effective_parameter_reconstruction_preserves_model_coordinates(project_root):
    ensemble, _ = load_step04_accepted_ensemble(project_root, max_candidates=1)
    row = ensemble.iloc[0]
    params = reconstruct_flat_params(row)
    paramdict = build_paramdict(str(row["condition"]), 100, params)
    astro = paramdict["Astrocyte"]
    external = paramdict["external"]
    assert np.isclose(astro["d_gap"] * astro["P_k"], float(row["P_gap_eff"]))
    assert np.isclose(
        astro["gama_t"] * astro["Sig_a"] / (astro["w_a"] * astro["F"]),
        float(row["gamma_t_eff"]),
    )
    assert np.isclose(
        astro["gama_s"] * astro["Sig_a"] / (astro["w_a"] * astro["F"]),
        float(row["gamma_s_eff"]),
    )
    assert np.isclose(astro["w_a"] / external["w_o"], float(row["volume_ratio_wa_wo"]))


def test_step05_flux_rows_have_hidden_current_and_proxy_metrics(project_root):
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=1,
            time_points=60,
            bootstrap_iterations=0,
            write_outputs=False,
        ),
    )
    flux = result["accepted_fit_mechanisms"]
    required = {
        "I_Kir_integral",
        "I_kgap_integral",
        "I_leak_integral",
        "K_o_peak",
        "gap_to_kir_integral_ratio",
        "proxy_validity_class",
    }
    assert required.issubset(flux.columns)
    assert len(flux) == 6
    assert (flux["simulation_status"] == "ok").all()


def test_kbath_middle_is_current_specific_and_gain_scaled(project_root):
    ensemble, _ = load_step04_accepted_ensemble(project_root, max_candidates=1)
    row = ensemble.iloc[0].to_dict()
    row["k_bath_gain"] = 1.25

    p50 = reconstruct_flat_params(row, current_na=50, sweep=1)
    p175 = reconstruct_flat_params(row, current_na=175, sweep=6)

    assert np.isclose(
        p50["K_bath_value_middle"], CURRENT_DICT_K_BATH_VALUES["50"][1] * 1.25
    )
    assert np.isclose(
        p175["K_bath_value_middle"], CURRENT_DICT_K_BATH_VALUES["175"][1] * 1.25
    )
    assert p50["K_bath_value_middle"] != p175["K_bath_value_middle"]
    assert (
        kbath_middle_for_current(row, 100, sweep=3)[1]
        == "historical_current_kbath_scaled_by_gain"
    )


def test_step05_flux_rows_record_current_specific_protocol_and_stimulus_window(
    project_root,
):
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=1,
            time_points=60,
            bootstrap_iterations=0,
            write_outputs=False,
        ),
    )
    flux = result["accepted_fit_mechanisms"]
    assert {
        "K_bath_middle_used",
        "K_bath_override_mode",
        "stim_window_start_s",
        "stim_window_end_s",
    }.issubset(flux.columns)
    assert flux["K_bath_middle_used"].nunique() == 6
    for _, row in flux.iterrows():
        expected = CURRENT_DICT_K_BATH_VALUES[str(int(row["current_na"]))][1] * float(
            row.get("k_bath_gain", 1.0)
        )
        assert np.isclose(float(row["K_bath_middle_used"]), expected)
