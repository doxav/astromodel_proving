from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.step04_cell_specific_multisweep import (
    Step04Config,
    build_threshold_table,
    compute_reliability_weights,
    generate_candidates_for_cell,
    score_candidate_for_cell,
    select_cells_for_run,
)


def _fake_cell(condition: str = 'CONTROL', region: str = 'DH', file_id: str = 'cell_a'):
    sweeps = {}
    for sweep_idx, current_na in enumerate([50, 75, 100, 125, 150, 175], start=1):
        t = np.linspace(0.0, 40.0, 50)
        v = np.linspace(-70.0, -60.0 + sweep_idx, 50)
        sweeps[sweep_idx] = {
            'time_s': t,
            'vm_mV': v,
            'features': {
                'stim_onset_s': 10.0,
                'stim_offset_s': 30.0,
                'peak_depolarization_mV': float(sweep_idx),
            },
        }
    return {
        'meta': {
            'file_id': file_id,
            'file': f'{file_id}.atf',
            'region': region,
            'condition': condition,
            'group_label': f'{region}_{condition}',
            'cell_label': file_id,
        },
        'sweeps': sweeps,
    }


def test_select_cells_for_run_group_balanced_prefers_group_coverage() -> None:
    cells = [
        _fake_cell('CONTROL', 'DH', 'c1'),
        _fake_cell('CONTROL', 'DH', 'c2'),
        _fake_cell('CONTROL', 'VH', 'c3'),
        _fake_cell('MFA', 'DH', 'c4'),
        _fake_cell('MFA', 'VH', 'c5'),
    ]
    selected = select_cells_for_run(cells, max_cells=4, mode='group_balanced')
    groups = {(c['meta']['condition'], c['meta']['region']) for c in selected}
    assert ('CONTROL', 'DH') in groups
    assert ('CONTROL', 'VH') in groups
    assert ('MFA', 'DH') in groups
    assert ('MFA', 'VH') in groups


def test_select_cells_for_run_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        select_cells_for_run([_fake_cell()], max_cells=1, mode='bad_mode')


def test_build_threshold_table_leave_one_out_scope_excludes_target_cell() -> None:
    df = pd.DataFrame(
        {
            'file_id': ['a'] * 6 + ['b'] * 6,
            'region': ['DH'] * 12,
            'condition': ['CONTROL'] * 12,
            'sweep': [1, 2, 3, 4, 5, 6] * 2,
            'peak_depolarization_mV': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
            'stim_end_depolarization_mV': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
            'rise_slope_mV_per_s': [1] * 12,
            'rise_tau_s': [1] * 12,
            'plateau_slope_mV_per_s': [0] * 12,
            'undershoot_magnitude_mV': [0.5] * 12,
            'decay_slope_mV_per_s': [1] * 12,
            'decay_tau_s': [1] * 12,
            'return_slope_mV_per_s': [1] * 12,
        }
    )
    rel = compute_reliability_weights(df)
    thr = build_threshold_table(df, rel, exclude_file_id='a')
    assert set(thr['threshold_scope']) == {'leave_one_cell_out_region_specific'}
    row = thr[(thr['feature'] == 'peak_depolarization_mV') & (thr['sweep'] == 1)].iloc[0]
    assert row['median'] == 10
    assert row['n_total_rows'] == 1


def test_generate_candidates_for_cell_returns_requested_count() -> None:
    seed_map = {
        'CONTROL': {
            'condition': 'CONTROL',
            'P_gap_eff': 1e-5,
            'gamma_t_eff': 1e-5,
            'gamma_s_eff': 2e-5,
            'volume_ratio_wa_wo': 1.2,
            'g_kir': 1.0,
            'gl_a': 0.01,
            'zth': 0.2,
            'zs': 0.05,
            'eps': 1e-3,
            'switching_function': 'sigmoid',
            'Va_l': -70.0,
            'Va_s': -90.0,
            'ca': 400.0,
        }
    }
    rng = np.random.default_rng(1)
    candidates = generate_candidates_for_cell(_fake_cell()['meta'], seed_map, 1.0, Step04Config(n_candidates=5), rng)
    assert len(candidates) == 5
    assert all(c['P_gap_eff'] > 0 for c in candidates)
    assert all(c['candidate_id'] == idx for idx, c in enumerate(candidates))


def test_score_candidate_for_cell_rejects_non_monotonic_peaks(monkeypatch) -> None:
    cell = _fake_cell()
    threshold_rows = []
    for sweep in range(1, 7):
        for feature in [
            'peak_depolarization_mV',
            'stim_end_depolarization_mV',
            'rise_slope_mV_per_s',
            'rise_tau_s',
            'plateau_slope_mV_per_s',
            'undershoot_magnitude_mV',
            'decay_slope_mV_per_s',
            'decay_tau_s',
            'return_slope_mV_per_s',
        ]:
            threshold_rows.append(
                {
                    'condition': 'CONTROL',
                    'region': 'DH',
                    'sweep': sweep,
                    'feature': feature,
                    'acceptable_lower': -1e9,
                    'acceptable_upper': 1e9,
                    'reliability_weight': 1.0,
                }
            )
    threshold_df = pd.DataFrame(threshold_rows)

    simulated_peaks = iter([1.0, 2.0, 1.5, 4.0, 5.0, 6.0])

    def fake_simulate(*args, **kwargs):
        time_ms = kwargs['time_ms']
        return np.linspace(-70.0, -65.0, len(time_ms))

    def fake_extract_features(*args, **kwargs):
        peak = next(simulated_peaks)
        return {
            'peak_depolarization_mV': peak,
            'stim_end_depolarization_mV': peak,
            'rise_slope_mV_per_s': 1.0,
            'rise_tau_s': 1.0,
            'plateau_slope_mV_per_s': 0.0,
            'undershoot_magnitude_mV': 0.0,
            'decay_slope_mV_per_s': 1.0,
            'decay_tau_s': 1.0,
            'return_slope_mV_per_s': 1.0,
        }

    monkeypatch.setattr('src.step04_cell_specific_multisweep.simulate_voltage_trace', fake_simulate)
    monkeypatch.setattr('src.step04_cell_specific_multisweep.extract_features_from_trace', fake_extract_features)

    summary, sweep_rows, _ = score_candidate_for_cell(
        cell,
        {'candidate_id': 0, 'P_gap_eff': 1e-5, 'gamma_t_eff': 1e-5, 'gamma_s_eff': 2e-5, 'volume_ratio_wa_wo': 1.2, 'g_kir': 1.0, 'gl_a': 0.01, 'zth': 0.2, 'zs': 0.05, 'eps': 1e-3, 'k_bath_gain': 1.0, 'switching_function': 'sigmoid', 'Va_l': -70.0, 'Va_s': -90.0, 'ca': 400.0},
        threshold_df,
        Step04Config(n_candidates=1, comparison_points=10),
    )
    assert len(sweep_rows) == 6
    assert summary['monotonic_predicted_peak'] is False
    assert summary['accepted'] is False
