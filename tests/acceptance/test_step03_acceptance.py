from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.identifiability import PRIMARY_EFFECTIVE_PARAMETERS, Step03Config, run_step03_identifiability_screen


@pytest.fixture(scope="module")
def step03_results(project_root: Path, tmp_path_factory: pytest.TempPathFactory):
    outputs_dir = project_root / "outputs" / "identifiability"
    expected = {
        "effective_parameter_map.csv",
        "structural_invariance_diagnostics.csv",
        "invariance_diagnostics.csv",
        "profile_likelihoods.csv",
        "profile_summary.csv",
        "fim_spectrum.csv",
        "fim_mode_loadings.csv",
        "fim_diagnostics.csv",
        "interpretation_notes.csv",
        "analysis_summary.csv",
    }
    if expected.issubset({p.name for p in outputs_dir.glob("*.csv")}):
        return {path.stem: pd.read_csv(path) for path in outputs_dir.glob("*.csv") if path.name in expected}
    output_dir = tmp_path_factory.mktemp("step03_identifiability")
    return run_step03_identifiability_screen(project_root, output_dir=output_dir, config=Step03Config(n_timepoints=80, observable_stride=2))


def test_step03_pipeline_writes_required_outputs(step03_results) -> None:
    assert {
        "effective_parameter_map",
        "structural_invariance_diagnostics",
        "invariance_diagnostics",
        "profile_likelihoods",
        "profile_summary",
        "fim_spectrum",
        "fim_mode_loadings",
        "fim_diagnostics",
        "interpretation_notes",
        "analysis_summary",
    }.issubset(step03_results)


def test_step03_profiles_cover_primary_effective_parameters(step03_results) -> None:
    profiles = step03_results["profile_likelihoods"]
    summary = step03_results["profile_summary"]
    allowed = {"clear_valley", "broad_valley", "flat_unbounded", "boundary_hit"}

    assert set(PRIMARY_EFFECTIVE_PARAMETERS).issubset(set(profiles["parameter"]))
    assert set(summary["profile_class"]).issubset(allowed)
    assert profiles.groupby("parameter")["multiplier"].nunique().ge(5).all()
    assert np.isfinite(profiles["loss"]).all()
    assert profiles["nuisance_refit_method"].str.contains("least_squares_affine_vm_scale_offset").all()


def test_step03_fim_outputs_are_finite_and_annotated(step03_results) -> None:
    spectrum = step03_results["fim_spectrum"]
    loadings = step03_results["fim_mode_loadings"]

    assert {"eigenvalue", "log10_eigenvalue", "relative_eigenvalue", "mode_class", "dominant_parameter"}.issubset(spectrum.columns)
    assert np.isfinite(spectrum["eigenvalue"]).all()
    assert (spectrum["eigenvalue"] >= 0).all()
    assert set(spectrum["mode_class"]).issubset({"stiff", "sloppy"})
    assert {"raw", "effective"}.issubset(set(loadings["coordinate_type"]))
    assert loadings.groupby(["db_name", "mode_index"])["parameter"].nunique().ge(5).all()
