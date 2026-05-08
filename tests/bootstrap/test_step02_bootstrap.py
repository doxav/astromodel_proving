from __future__ import annotations

from pathlib import Path

from src.atf_features import (
    build_atf_inventory,
    count_region_condition_cells,
    extract_features,
    get_sweep_map,
    parse_atf,
    parse_experimental_factors,
    preprocess_parsed,
)


def test_step02_required_inputs_exist(atf_dir: Path, threshold_csv: Path) -> None:
    assert atf_dir.exists()
    assert threshold_csv.exists()
    assert len(list(atf_dir.glob("*.atf"))) == 37


def test_step02_parser_resolves_region_and_condition_labels() -> None:
    dh_control = parse_experimental_factors("DH_old.atf")
    assert dh_control.region == "DH"
    assert dh_control.condition == "CONTROL"

    vh_mfa = parse_experimental_factors("VH_6_MFA.atf")
    assert vh_mfa.region == "VH"
    assert vh_mfa.condition == "MFA"

    vh_ba = parse_experimental_factors("VH_OG_MFA_Ba.atf")
    assert vh_ba.region == "VH"
    assert vh_ba.condition == "MFA_BA"


def test_step02_inventory_and_region_contract(atf_dir: Path) -> None:
    inventory = build_atf_inventory(atf_dir)
    counts = count_region_condition_cells(inventory)
    assert len(inventory) == 37
    assert set(inventory["region"]) == {"DH", "VH"}
    assert set(inventory["condition"]) == {"CONTROL", "MFA", "MFA_BA"}
    expected = {
        ("DH", "CONTROL"): 7,
        ("VH", "CONTROL"): 4,
        ("DH", "MFA"): 6,
        ("VH", "MFA"): 7,
        ("DH", "MFA_BA"): 6,
        ("VH", "MFA_BA"): 7,
    }
    observed = {(row.region, row.condition): int(row.n_cells) for row in counts.itertuples(index=False)}
    assert observed == expected
    assert counts["matches_expected"].all()


def test_step02_representative_atf_file_can_be_parsed_and_featured(atf_dir: Path) -> None:
    path = atf_dir / "1_DH_1.atf"
    parsed = parse_atf(path)
    sweep_map = get_sweep_map(parsed)
    cleaned = preprocess_parsed(parsed)
    feature_df = extract_features(cleaned)

    assert parsed["data"].shape[0] > 1000
    assert len(sweep_map) == 6
    assert len(feature_df) == 6
    assert feature_df["file_id"].nunique() == 1
    assert set(feature_df["sweep"]) == {1, 2, 3, 4, 5, 6}
    for column in [
        "region",
        "condition",
        "peak_depolarization_mV",
        "stim_end_depolarization_mV",
        "plateau_reached",
        "has_undershoot",
        "return_slope_mV_per_s",
    ]:
        assert column in feature_df.columns
