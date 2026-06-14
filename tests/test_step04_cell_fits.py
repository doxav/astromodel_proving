from __future__ import annotations

import json
import sqlite3
import pandas as pd

import pytest

import numpy as np

from src.step04_loss import Step04LossConfig, Step04OptimizerConfig
from src.step04_cell_fits import (
    TRACE_SHAPE_OBJECTIVE_SPECS,
    Step04Config,
    _named_to_x,
    _seed_optuna_study_from_candidate_csv,
    _trace_objective_loss,
    _transform_trace_for_objective,
    build_cell_trace_inventory,
    run_step04_cell_specific_six_sweep_fitting,
)
from src.step04_outputs import (
    STEP04_DOWNSTREAM_ARTIFACTS,
    STEP04_OUTPUT_SCHEMA_VERSION,
    missing_step04_downstream_artifacts,
)

def test_single_control_cell_fit_writes_expected_outputs(project_root):
    inv = build_cell_trace_inventory(project_root / 'data' / '2_K+ Pumps Data', n_fit_points=10, file_ids=['1_DH_1_CONTROL'])
    assert list(inv) == ['1_DH_1_CONTROL']
    assert len(inv['1_DH_1_CONTROL']) == 6
    assert inv['1_DH_1_CONTROL'][1].stim_onset_s == pytest.approx(11.166, abs=0.01)
    assert inv['1_DH_1_CONTROL'][1].step_source == 'IP_curr'
    out = project_root / 'outputs' / 'step04_test_single'
    res = run_step04_cell_specific_six_sweep_fitting(project_root, output_dir=out, selected_file_ids=['1_DH_1_CONTROL'], max_cells=1, n_fit_points=10, n_starts=1, max_nfev_all6=1, max_nfev_holdout=1)
    assert (out / 'cell_fit_candidates.csv').exists()
    assert (out / 'accepted_cell_ensembles.csv').exists()
    assert (out / 'effective_diverse_cell_ensembles.csv').exists()
    assert (out / 'effective_diverse_selection_summary.csv').exists()
    sqlite_path = out / 'step04_cell_fits.sqlite'
    assert sqlite_path.exists()
    with sqlite3.connect(sqlite_path) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "cell_fit_candidates",
            "candidate_sweep_metrics",
            "effective_diverse_cell_ensembles",
            "effective_diverse_selection_summary",
            "run_metadata",
        }.issubset(tables)
        n_candidate_rows = con.execute("SELECT COUNT(*) FROM cell_fit_candidates").fetchone()[0]
        n_sweep_rows = con.execute("SELECT COUNT(*) FROM candidate_sweep_metrics").fetchone()[0]
    assert n_candidate_rows == len(res['cell_fit_candidates'])
    assert n_sweep_rows == len(res['candidate_sweep_metrics'])
    config_path = out / 'optimization_config.json'
    assert config_path.exists()
    optimization_config = json.loads(config_path.read_text())
    assert optimization_config['optimizer_config']['backend'] == 'least_squares'
    assert optimization_config['optimization_config_hash']
    assert 'scalar_objective' in res['cell_fit_candidates'].columns
    assert 'objective_trace' in res['cell_fit_candidates'].columns
    assert 'effective_selection_strategy' in res['effective_diverse_cell_ensembles'].columns
    assert set(res['cell_fit_candidates']['optimization_config_hash']) == {optimization_config['optimization_config_hash']}
    assert len(res['heldout_current_screen']) == 6


def test_cell_trace_inventory_detects_per_file_control_timing(project_root):
    inv = build_cell_trace_inventory(
        project_root / 'data' / '2_K+ Pumps Data',
        n_fit_points=10,
        file_ids=['1_DH_1_CONTROL', '1_DH_2_CONTROL'],
    )

    assert inv['1_DH_1_CONTROL'][1].stim_onset_s == pytest.approx(11.166, abs=0.01)
    assert inv['1_DH_2_CONTROL'][1].stim_onset_s == pytest.approx(21.153, abs=0.01)
    assert inv['1_DH_2_CONTROL'][1].stim_offset_s == pytest.approx(41.129, abs=0.01)


def test_step04_outputs_are_downstream_reusable(project_root):
    out = project_root / "outputs" / "step04_test_downstream_reusable"
    run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=["1_DH_1_CONTROL"],
        max_cells=1,
        n_fit_points=8,
        n_starts=1,
        max_nfev_all6=1,
        max_nfev_holdout=1,
    )

    assert missing_step04_downstream_artifacts(out) == []

    summary = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["output_schema_version"] == STEP04_OUTPUT_SCHEMA_VERSION
    assert summary["downstream_artifacts"] == STEP04_DOWNSTREAM_ARTIFACTS

    manifest_path = out / "step04_artifact_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["step"] == "step04"
    assert manifest["output_schema_version"] == STEP04_OUTPUT_SCHEMA_VERSION
    assert manifest["missing_artifacts"] == []
    assert manifest["downstream_artifacts"] == STEP04_DOWNSTREAM_ARTIFACTS

    # Step 05 should be able to start from accepted ensembles, while Step 06
    # should be able to validate candidates/sweep metrics without rerunning optimization.
    for key in ("accepted_ensembles", "candidates", "sweep_metrics", "optimization_config"):
        assert (out / STEP04_DOWNSTREAM_ARTIFACTS[key]).exists()


def test_trace_shape_objective_variants_cover_legacy_loss_and_target_modes():
    expected = {
        ("L2", "default"),
        ("L1", "default"),
        ("HUBER", "centered"),
        ("LOG_COSH", "centered_scaled"),
        ("COMBINED", "centered_scaled"),
    }

    assert expected.issubset(set(TRACE_SHAPE_OBJECTIVE_SPECS.values()))


def test_trace_shape_target_transform_and_loss_are_finite():
    trace = np.array([1.0, 3.0, 5.0])
    centered = _transform_trace_for_objective(trace, "centered")
    centered_scaled = _transform_trace_for_objective(trace, "centered_scaled")

    assert centered.mean() == pytest.approx(0.0)
    assert centered_scaled.mean() == pytest.approx(0.0)
    assert centered_scaled.std() == pytest.approx(1.0)
    assert _trace_objective_loss(np.array([1000.0, 1.0]), np.zeros(2), "L2", Step04LossConfig()) < np.inf


def test_optuna_scalar_smoke_writes_config(project_root):
    pytest.importorskip("optuna")
    out = project_root / 'outputs' / 'step04_test_optuna_scalar'
    optimizer_config = Step04OptimizerConfig(
        backend='optuna_scalar',
        optuna_n_trials=2,
        run_holdout=False,
    )
    res = run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=['1_DH_1_CONTROL'],
        max_cells=1,
        n_fit_points=10,
        n_starts=1,
        max_nfev_all6=1,
        max_nfev_holdout=1,
        optimizer_config=optimizer_config,
    )
    config_path = out / 'optimization_config.json'
    assert config_path.exists()
    optimization_config = json.loads(config_path.read_text())
    assert optimization_config['optimizer_config']['backend'] == 'optuna_scalar'
    assert not res['cell_fit_candidates'].empty
    assert res['heldout_current_screen'].empty


def test_optuna_multi_smoke_returns_pareto_candidates(project_root):
    pytest.importorskip("optuna")
    out = project_root / 'outputs' / 'step04_test_optuna_multi'
    optimizer_config = Step04OptimizerConfig(
        backend='optuna_multi',
        optuna_n_trials=2,
        run_holdout=False,
    )
    res = run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=['1_DH_1_CONTROL'],
        max_cells=1,
        n_fit_points=10,
        n_starts=1,
        max_nfev_all6=1,
        max_nfev_holdout=1,
        optimizer_config=optimizer_config,
    )
    assert not res['cell_fit_candidates'].empty
    assert 'pareto_front' in res['cell_fit_candidates'].columns


def test_hybrid_optimizer_seeds_optuna_and_refines_top_candidate(project_root):
    pytest.importorskip("optuna")
    out = project_root / 'outputs' / 'step04_test_hybrid'
    optimizer_config = Step04OptimizerConfig(
        backend='hybrid',
        optuna_n_trials=2,
        hybrid_scipy_pre_nfev=1,
        hybrid_scipy_post_nfev=1,
        hybrid_refine_top_k=1,
        run_holdout=False,
    )
    res = run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=['1_DH_1_CONTROL'],
        max_cells=1,
        n_fit_points=8,
        n_starts=1,
        max_nfev_all6=1,
        max_nfev_holdout=1,
        optimizer_config=optimizer_config,
    )
    candidates = res['cell_fit_candidates']
    optimization_config = json.loads((out / 'optimization_config.json').read_text())

    assert optimization_config['optimizer_config']['backend'] == 'hybrid'
    assert {'scipy_pre', 'optuna_seeded', 'scipy_post'}.issubset(set(candidates['hybrid_stage']))
    assert 'hybrid_parent_trial_number' in candidates.columns


def test_cell_fit_workers_parallel_cell_level_execution(project_root):
    out = project_root / 'outputs' / 'step04_test_parallel'
    res = run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=['1_DH_1_CONTROL', '1_DH_2_CONTROL'],
        max_cells=None,
        n_fit_points=8,
        n_starts=1,
        max_nfev_all6=1,
        max_nfev_holdout=1,
        cell_fit_workers=2,
    )
    summary = json.loads((out / 'analysis_summary.json').read_text())

    assert summary['cell_fit_workers'] == 2
    assert set(res['cell_fit_quality_summary']['file_id']) == {'1_DH_1_CONTROL', '1_DH_2_CONTROL'}
    assert set(res['cell_fit_candidates']['file_id']) == {'1_DH_1_CONTROL', '1_DH_2_CONTROL'}


def test_optuna_fallback_is_explicit_opt_in(project_root, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ModuleNotFoundError("simulated missing optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = project_root / 'outputs' / 'step04_test_optuna_fallback'
    res = run_step04_cell_specific_six_sweep_fitting(
        project_root,
        output_dir=out,
        selected_file_ids=['1_DH_1_CONTROL'],
        max_cells=1,
        n_fit_points=8,
        optimizer_config=Step04OptimizerConfig(
            backend='optuna_scalar',
            optuna_n_trials=2,
            run_holdout=False,
            allow_optuna_fallback=True,
        ),
    )
    assert not res['cell_fit_candidates'].empty


def test_optuna_backend_raises_without_fallback_when_missing(project_root, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ModuleNotFoundError("simulated missing optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Optuna backend requested"):
        run_step04_cell_specific_six_sweep_fitting(
            project_root,
            output_dir=project_root / 'outputs' / 'step04_test_optuna_missing_error',
            selected_file_ids=['1_DH_1_CONTROL'],
            max_cells=1,
            n_fit_points=8,
            optimizer_config=Step04OptimizerConfig(
                backend='optuna_scalar',
                optuna_n_trials=2,
                run_holdout=False,
                allow_optuna_fallback=False,
            ),
        )


def test_optuna_preseed_csv_adds_only_accepted_candidates(tmp_path):
    pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution
    import optuna

    candidate_named = {
        "P_gap_eff": 0.8,
        "gamma_t_eff": 0.5,
        "gamma_s_eff": 0.5,
        "volume_ratio_wa_wo": 1.0,
        "gki": 0.7,
        "eps": 0.9,
        "gl_a": 0.6,
        "zth": 0.4,
        "zs": 0.3,
    }
    candidate_vector = _named_to_x(candidate_named)
    seed_csv = tmp_path / "seed_candidates.csv"
    pd.DataFrame(
        [
            {
                "file_id": "1_DH_1_CONTROL",
                "accepted_all6": True,
                "scalar_objective": 0.25,
                "objective_trace": 0.15,
                "objective_feature": 0.10,
                **candidate_named,
            },
            {
                "file_id": "1_DH_1_CONTROL",
                "accepted_all6": False,
                "scalar_objective": 0.10,
                "objective_trace": 0.10,
                "objective_feature": 0.05,
                **candidate_named,
            },
            {
                "file_id": "2_DH_1_CONTROL",
                "accepted_all6": True,
                "scalar_objective": 0.05,
                "objective_trace": 0.02,
                "objective_feature": 0.01,
                **candidate_named,
            },
        ]
    ).to_csv(seed_csv, index=False)

    cfg = Step04Config(
        project_root=tmp_path,
        output_dir=tmp_path / "seeded_run",
        optimizer_config=Step04OptimizerConfig(
            optuna_preseed_candidate_csv=str(seed_csv),
            optuna_preseed_candidate_limit=1,
            optuna_preseed_only_accepted=True,
        ),
    ).resolve()

    study = optuna.create_study(direction="minimize")
    distributions = {
        name: FloatDistribution(float(center - 1.0), float(center + 1.0))
        for name, center in zip(
            ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo", "gki", "eps", "gl_a", "zth", "zs"],
            candidate_vector,
        )
    }

    n_added = _seed_optuna_study_from_candidate_csv(
        study=study,
        file_id="1_DH_1_CONTROL",
        cfg=cfg,
        objective_names=(),
        multi_objective=False,
        distributions=distributions,
        evaluate_x=None,
    )

    assert n_added == 1
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 1
    assert complete[0].value is not None
    assert complete[0].value == pytest.approx(0.25)
