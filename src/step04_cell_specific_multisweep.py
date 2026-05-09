from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .astro_model import VALID_CURRENTS, simulate_voltage_trace
from .atf_features import FEATURE_COLUMNS, build_feature_table, extract_features_from_trace
from .contracts import protocol_condition
from .feature_contracts import ThresholdScope, build_threshold_table as _shared_build_threshold_table, compute_reliability_weights as _shared_compute_reliability_weights, score_feature_contract
from .trace_utils import baseline_center as _shared_baseline_center, downsample_trace, nrmse

OUTPUT_SUBDIR = 'step04_cell_specific_multisweep'
_CACHE: dict[tuple[str, str], object] = {}


@dataclass(slots=True)
class Step04Config:
    random_seed: int = 7
    n_candidates: int = 16
    max_accepted_per_cell: int = 3
    max_cells: Optional[int] = None
    cell_selection_mode: str = 'ordered'
    comparison_points: int = 180
    feature_mean_pass_threshold: float = 0.72
    feature_min_sweep_pass_threshold: float = 0.55
    max_mean_trace_nrmse: float = 0.55
    candidate_log10_width: float = 0.45
    k_bath_gain_bounds: tuple[float, float] = (0.6, 1.5)
    benchmark_max_cells: int = 1
    write_outputs: bool = True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _project_root_from(project_root: Path | str) -> Path:
    return Path(project_root).resolve()


def compute_effective_seed_summary(legacy_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(legacy_dir.glob('*_BEST_FIT_PARAM.csv')):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            experiment = str(row['Experiment'])
            condition = experiment.split('_')[0].upper()
            w_a = float(row.get('w_a', 2000.0))
            P_gap_eff = float(row['d']) * float(row['pk'])
            gamma_t_eff = float(row['gt']) * 1600.0 / (w_a * 96485.0)
            gamma_s_eff = float(row['gs']) * 1600.0 / (w_a * 96485.0)
            volume_ratio = w_a / float(row['wo']) if float(row['wo']) != 0 else np.nan
            rows.append(
                {
                    'condition': condition,
                    'experiment': experiment,
                    'P_gap_eff': P_gap_eff,
                    'gamma_t_eff': gamma_t_eff,
                    'gamma_s_eff': gamma_s_eff,
                    'volume_ratio_wa_wo': volume_ratio,
                    'g_kir': float(row['gki']),
                    'gl_a': float(row['gl_a']),
                    'zth': float(row['zth']),
                    'zs': float(row['zs']),
                    'eps': float(row['eps']),
                    'switching_function': str(row.get('switching_function', 'sigmoid')),
                    'Va_l': float(row.get('Va_l', -70.0)),
                    'Va_s': float(row.get('Va_s', -90.0)),
                    'ca': float(row.get('ca', 400.0)),
                }
            )
    if not rows:
        raise FileNotFoundError('No *_BEST_FIT_PARAM.csv files found for seeding.')
    seed_df = pd.DataFrame(rows)
    medians = (
        seed_df.groupby('condition', as_index=False)
        .agg(
            P_gap_eff=('P_gap_eff', 'median'),
            gamma_t_eff=('gamma_t_eff', 'median'),
            gamma_s_eff=('gamma_s_eff', 'median'),
            volume_ratio_wa_wo=('volume_ratio_wa_wo', 'median'),
            g_kir=('g_kir', 'median'),
            gl_a=('gl_a', 'median'),
            zth=('zth', 'median'),
            zs=('zs', 'median'),
            eps=('eps', 'median'),
            Va_l=('Va_l', 'median'),
            Va_s=('Va_s', 'median'),
            ca=('ca', 'median'),
        )
    )
    mode_switch = seed_df.groupby('condition')['switching_function'].agg(lambda s: s.mode().iat[0]).rename('switching_function')
    medians = medians.merge(mode_switch.reset_index(), on='condition', how='left')
    return medians


def compute_reliability_weights(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper around shared feature reliability scoring."""

    return _shared_compute_reliability_weights(feature_df, FEATURE_COLUMNS)


def build_threshold_table(feature_df: pd.DataFrame, reliability_df: pd.DataFrame, exclude_file_id: str | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper around shared threshold construction."""

    scope = ThresholdScope("leave_one_cell_out", exclude_file_id=exclude_file_id) if exclude_file_id is not None else ThresholdScope("region_specific")
    return _shared_build_threshold_table(feature_df, reliability_df, scope, feature_columns=FEATURE_COLUMNS)

def _baseline_center(t_s: np.ndarray, v_mV: np.ndarray, onset_s: float) -> np.ndarray:
    """Backward-compatible wrapper around :func:`trace_utils.baseline_center`."""

    return _shared_baseline_center(t_s, v_mV, onset_s)


def _downsample_to_common_grid(t_s: np.ndarray, v_mV: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper around :func:`trace_utils.downsample_trace`."""

    return downsample_trace(t_s, v_mV, n_points, preserve_short=False)


def _distance_to_interval(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return 1.0
    if low <= value <= high:
        return 0.0
    width = max(high - low, 1e-6)
    if value < low:
        return float((low - value) / width)
    return float((value - high) / width)


def _make_condition_seed_map(seed_summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {str(row['condition']).upper(): row.to_dict() for _, row in seed_summary.iterrows()}


def _canonical_condition(condition: str) -> str:
    """Backward-compatible wrapper returning the simulation protocol condition."""

    return protocol_condition(condition)

def select_cells_for_run(cells: Iterable[Mapping[str, object]], max_cells: int | None, mode: str = 'ordered') -> list[Mapping[str, object]]:
    ordered = sorted(
        list(cells),
        key=lambda c: (str(c['meta']['condition']), str(c['meta']['region']), str(c['meta']['file_id'])),
    )
    mode = str(mode).lower()
    if mode not in {'ordered', 'group_balanced'}:
        raise ValueError(f'Unknown cell_selection_mode: {mode}')
    if max_cells is None or max_cells >= len(ordered):
        return ordered
    if mode == 'ordered':
        return ordered[:max_cells]
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for cell in ordered:
        key = (str(cell['meta']['condition']), str(cell['meta']['region']))
        groups.setdefault(key, []).append(cell)
    group_keys = sorted(groups)
    selected: list[Mapping[str, object]] = []
    round_idx = 0
    while len(selected) < max_cells:
        progressed = False
        for key in group_keys:
            bucket = groups[key]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
                if len(selected) >= max_cells:
                    break
        if not progressed:
            break
        round_idx += 1
    return selected


def generate_candidates_for_cell(
    cell_meta: Mapping[str, object],
    seed_map: Mapping[str, Mapping[str, object]],
    cell_peak_scale: float,
    config: Step04Config,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    condition = _canonical_condition(str(cell_meta['condition']))
    seed = dict(seed_map[condition])
    base = {
        'P_gap_eff': float(seed['P_gap_eff']),
        'gamma_t_eff': float(seed['gamma_t_eff']),
        'gamma_s_eff': float(seed['gamma_s_eff']),
        'volume_ratio_wa_wo': float(seed['volume_ratio_wa_wo']),
        'g_kir': float(seed['g_kir']),
        'gl_a': float(seed['gl_a']),
        'zth': float(seed['zth']),
        'zs': float(seed['zs']),
        'eps': float(seed['eps']),
        'switching_function': str(seed['switching_function']),
        'Va_l': float(seed['Va_l']),
        'Va_s': float(seed['Va_s']),
        'ca': float(seed['ca']),
        'k_bath_gain': float(np.clip(cell_peak_scale, *config.k_bath_gain_bounds)),
    }
    candidates = [dict(candidate_id=0, **base)]
    positive_keys = ['P_gap_eff', 'gamma_t_eff', 'gamma_s_eff', 'volume_ratio_wa_wo', 'g_kir', 'gl_a', 'zs', 'eps']
    for cand_id in range(1, config.n_candidates):
        cand = dict(base)
        for key in positive_keys:
            factor = 10 ** rng.uniform(-config.candidate_log10_width, config.candidate_log10_width)
            cand[key] = max(1e-12, float(cand[key]) * factor)
        cand['zth'] = max(0.01, float(cand['zth']) * 10 ** rng.uniform(-0.30, 0.30))
        cand['k_bath_gain'] = float(np.clip(cand['k_bath_gain'] * 10 ** rng.uniform(-0.15, 0.15), *config.k_bath_gain_bounds))
        # controlled family variation
        if condition != 'CONTROL' and rng.random() < 0.25:
            cand['switching_function'] = 'tanh'
        elif rng.random() < 0.10:
            cand['switching_function'] = 'hill'
        cand['candidate_id'] = cand_id
        candidates.append(cand)
    return candidates


def score_candidate_for_cell(
    cell: Mapping[str, object],
    candidate: Mapping[str, object],
    threshold_df: pd.DataFrame,
    config: Step04Config,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    meta = cell['meta']
    sweep_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    failed = 0
    predicted_peaks: list[float] = []
    for sweep_idx, current_na in enumerate(VALID_CURRENTS, start=1):
        if sweep_idx not in cell['sweeps']:
            continue
        sweep = cell['sweeps'][sweep_idx]
        obs_t = np.asarray(sweep['time_s'], dtype=float)
        obs_v = np.asarray(sweep['vm_mV'], dtype=float)
        onset_s = float(sweep['features']['stim_onset_s'])
        offset_s = float(sweep['features']['stim_offset_s'])
        obs_grid, obs_centered = _downsample_to_common_grid(obs_t, _baseline_center(obs_t, obs_v, onset_s), config.comparison_points)
        try:
            sim_v = simulate_voltage_trace(
                str(meta['condition']),
                current_na,
                candidate,
                time_ms=obs_grid * 1000.0,
                onset_ms=onset_s * 1000.0,
                offset_ms=offset_s * 1000.0,
            )
            sim_centered = _baseline_center(obs_grid, sim_v, onset_s)
            pred_feat = extract_features_from_trace(obs_grid, sim_v, onset_s=onset_s, offset_s=offset_s)
            predicted_peaks.append(float(pred_feat['peak_depolarization_mV']))
            trace_nrmse = nrmse(sim_centered, obs_centered, denominator=max(float(np.nanmax(np.abs(obs_centered))), 1.0))
            feature_score = score_feature_contract(
                pred_feat,
                threshold_df,
                condition=str(meta['condition']),
                region=str(meta['region']),
                sweep=sweep_idx,
                feature_columns=FEATURE_COLUMNS,
            )
            weighted_pass_fraction = float(feature_score['weighted_pass_fraction'])
            weighted_feature_penalty = float(feature_score['weighted_feature_penalty'])
            pass_weights = []
            for feature in FEATURE_COLUMNS:
                thr = threshold_df[
                    (threshold_df['condition'] == meta['condition'])
                    & (threshold_df['region'] == meta['region'])
                    & (threshold_df['sweep'] == sweep_idx)
                    & (threshold_df['feature'] == feature)
                ]
                if thr.empty:
                    continue
                thr_row = thr.iloc[0]
                value = float(pred_feat.get(feature, np.nan))
                within = bool(feature_score.get(f'pass_{feature}', False))
                dist = _distance_to_interval(value, float(thr_row['acceptable_lower']), float(thr_row['acceptable_upper']))
                pass_weights.append((feature, within, float(thr_row['reliability_weight']), dist, value))
            sweep_rows.append(
                {
                    'file_id': meta['file_id'],
                    'region': meta['region'],
                    'condition': meta['condition'],
                    'candidate_id': int(candidate['candidate_id']),
                    'sweep': sweep_idx,
                    'current_na': current_na,
                    'trace_nrmse': trace_nrmse,
                    'weighted_pass_fraction': weighted_pass_fraction,
                    'weighted_feature_penalty': weighted_feature_penalty,
                    'pred_peak_depolarization_mV': float(pred_feat['peak_depolarization_mV']),
                    'pred_stim_end_depolarization_mV': float(pred_feat['stim_end_depolarization_mV']),
                    'pred_rise_tau_s': float(pred_feat['rise_tau_s']) if np.isfinite(pred_feat['rise_tau_s']) else np.nan,
                    'pred_decay_tau_s': float(pred_feat['decay_tau_s']) if np.isfinite(pred_feat['decay_tau_s']) else np.nan,
                    'simulation_failed': False,
                }
            )
            for feature, within, weight, dist, value in pass_weights:
                trace_rows.append(
                    {
                        'file_id': meta['file_id'],
                        'region': meta['region'],
                        'condition': meta['condition'],
                        'candidate_id': int(candidate['candidate_id']),
                        'sweep': sweep_idx,
                        'current_na': current_na,
                        'feature': feature,
                        'predicted_value': value,
                        'within_threshold': within,
                        'distance_to_interval': dist,
                        'reliability_weight': weight,
                    }
                )
        except Exception:
            failed += 1
            sweep_rows.append(
                {
                    'file_id': meta['file_id'],
                    'region': meta['region'],
                    'condition': meta['condition'],
                    'candidate_id': int(candidate['candidate_id']),
                    'sweep': sweep_idx,
                    'current_na': current_na,
                    'trace_nrmse': np.inf,
                    'weighted_pass_fraction': 0.0,
                    'weighted_feature_penalty': np.inf,
                    'pred_peak_depolarization_mV': np.nan,
                    'pred_stim_end_depolarization_mV': np.nan,
                    'pred_rise_tau_s': np.nan,
                    'pred_decay_tau_s': np.nan,
                    'simulation_failed': True,
                }
            )
    sweep_df = pd.DataFrame(sweep_rows)
    mean_trace_nrmse = float(sweep_df['trace_nrmse'].replace(np.inf, np.nan).mean()) if not sweep_df.empty else np.inf
    mean_pass_fraction = float(sweep_df['weighted_pass_fraction'].mean()) if not sweep_df.empty else 0.0
    min_pass_fraction = float(sweep_df['weighted_pass_fraction'].min()) if not sweep_df.empty else 0.0
    monotonic_predicted_peak = bool(np.all(np.diff(predicted_peaks) >= -1e-6)) if len(predicted_peaks) > 1 else False
    accepted_sweep_count = int((sweep_df['weighted_pass_fraction'] >= config.feature_min_sweep_pass_threshold).sum()) if not sweep_df.empty else 0
    objective = mean_trace_nrmse + float(sweep_df['weighted_feature_penalty'].replace(np.inf, np.nan).mean()) + (0.5 if not monotonic_predicted_peak else 0.0)
    accepted = bool(
        mean_pass_fraction >= config.feature_mean_pass_threshold
        and min_pass_fraction >= config.feature_min_sweep_pass_threshold
        and mean_trace_nrmse <= config.max_mean_trace_nrmse
        and monotonic_predicted_peak
        and accepted_sweep_count >= 5
        and failed == 0
    )
    summary = {
        'file_id': meta['file_id'],
        'file': meta['file'],
        'region': meta['region'],
        'condition': meta['condition'],
        'candidate_id': int(candidate['candidate_id']),
        'objective': objective,
        'mean_trace_nrmse': mean_trace_nrmse,
        'mean_weighted_pass_fraction': mean_pass_fraction,
        'min_weighted_pass_fraction': min_pass_fraction,
        'accepted_sweep_count': accepted_sweep_count,
        'simulation_failed_count': failed,
        'monotonic_predicted_peak': monotonic_predicted_peak,
        'accepted': accepted,
        **{k: candidate[k] for k in ['P_gap_eff', 'gamma_t_eff', 'gamma_s_eff', 'volume_ratio_wa_wo', 'g_kir', 'gl_a', 'zth', 'zs', 'eps', 'k_bath_gain', 'switching_function', 'Va_l', 'Va_s', 'ca']},
    }
    return summary, sweep_rows, trace_rows


def fit_cell(
    cell: Mapping[str, object],
    seed_map: Mapping[str, Mapping[str, object]],
    feature_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    config: Step04Config,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = cell['meta']
    # Use the strongest sweep to calibrate k_bath seed multiplicatively
    peak_by_sweep = [float(cell['sweeps'][s]['features']['peak_depolarization_mV']) for s in sorted(cell['sweeps'])]
    cell_peak_scale = float(np.median(np.array(peak_by_sweep)[-2:]) / max(np.median(peak_by_sweep[:2]), 1.0)) ** 0.25
    threshold_df = build_threshold_table(feature_df, reliability_df, exclude_file_id=str(meta['file_id']))
    candidates = generate_candidates_for_cell(meta, seed_map, cell_peak_scale, config, rng)
    summary_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for candidate in candidates:
        summ, sweeps, traces = score_candidate_for_cell(cell, candidate, threshold_df, config)
        summary_rows.append(summ)
        sweep_rows.extend(sweeps)
        trace_rows.extend(traces)
    summary_df = pd.DataFrame(summary_rows).sort_values(['accepted', 'objective', 'candidate_id'], ascending=[False, True, True]).reset_index(drop=True)
    accepted_df = summary_df[summary_df['accepted']].head(config.max_accepted_per_cell).copy()
    if accepted_df.empty:
        accepted_df = summary_df.head(1).copy()
        accepted_df['accepted'] = False
    accepted_ids = set(int(x) for x in accepted_df['candidate_id'])
    sweep_df = pd.DataFrame(sweep_rows)
    trace_df = pd.DataFrame(trace_rows)
    if 'candidate_id' in sweep_df.columns:
        sweep_df = sweep_df[sweep_df['candidate_id'].isin(accepted_ids)].copy()
    if 'candidate_id' in trace_df.columns:
        trace_df = trace_df[trace_df['candidate_id'].isin(accepted_ids)].copy()
    return accepted_df, sweep_df, trace_df


def benchmark_step04_configs(project_root: Path | str, base_config: Step04Config | None = None) -> pd.DataFrame:
    project_root = _project_root_from(project_root)
    base = base_config or Step04Config(write_outputs=False)
    # Warm shared caches (ATF parsing, feature table, reliability) outside the timed region
    warm_cfg = Step04Config(
        random_seed=base.random_seed,
        n_candidates=1,
        max_accepted_per_cell=1,
        max_cells=1,
        cell_selection_mode=base.cell_selection_mode,
        comparison_points=max(60, min(base.comparison_points, 90)),
        feature_mean_pass_threshold=base.feature_mean_pass_threshold,
        feature_min_sweep_pass_threshold=base.feature_min_sweep_pass_threshold,
        max_mean_trace_nrmse=base.max_mean_trace_nrmse,
        candidate_log10_width=base.candidate_log10_width,
        k_bath_gain_bounds=base.k_bath_gain_bounds,
        benchmark_max_cells=base.benchmark_max_cells,
        write_outputs=False,
    )
    run_step04_cell_specific_multisweep(project_root, warm_cfg)
    presets = [
        {
            'preset': 'coarse',
            'config': Step04Config(
                random_seed=base.random_seed,
                n_candidates=max(3, min(base.n_candidates, 4)),
                max_accepted_per_cell=base.max_accepted_per_cell,
                max_cells=min(base.benchmark_max_cells, base.max_cells) if base.max_cells is not None else base.benchmark_max_cells,
                cell_selection_mode=base.cell_selection_mode,
                comparison_points=max(90, min(base.comparison_points, 120)),
                feature_mean_pass_threshold=base.feature_mean_pass_threshold,
                feature_min_sweep_pass_threshold=base.feature_min_sweep_pass_threshold,
                max_mean_trace_nrmse=base.max_mean_trace_nrmse,
                candidate_log10_width=base.candidate_log10_width,
                k_bath_gain_bounds=base.k_bath_gain_bounds,
                benchmark_max_cells=base.benchmark_max_cells,
                write_outputs=False,
            )
        },
        {
            'preset': 'default',
            'config': Step04Config(
                random_seed=base.random_seed,
                n_candidates=base.n_candidates,
                max_accepted_per_cell=base.max_accepted_per_cell,
                max_cells=min(base.benchmark_max_cells, base.max_cells) if base.max_cells is not None else base.benchmark_max_cells,
                cell_selection_mode=base.cell_selection_mode,
                comparison_points=base.comparison_points,
                feature_mean_pass_threshold=base.feature_mean_pass_threshold,
                feature_min_sweep_pass_threshold=base.feature_min_sweep_pass_threshold,
                max_mean_trace_nrmse=base.max_mean_trace_nrmse,
                candidate_log10_width=base.candidate_log10_width,
                k_bath_gain_bounds=base.k_bath_gain_bounds,
                benchmark_max_cells=base.benchmark_max_cells,
                write_outputs=False,
            )
        },
    ]
    rows = []
    for preset in presets:
        cfg = preset['config']
        t0 = time.perf_counter()
        res = run_step04_cell_specific_multisweep(project_root, cfg)
        elapsed = time.perf_counter() - t0
        fit_status = res['fit_status_by_cell']
        n_cells_run = int(fit_status['file_id'].nunique()) if not fit_status.empty else 0
        n_accepted = int((fit_status['status'] == 'accepted').sum()) if not fit_status.empty else 0
        rows.append({
            'preset': preset['preset'],
            'elapsed_s': elapsed,
            'n_cells_run': n_cells_run,
            'n_candidates': cfg.n_candidates,
            'comparison_points': cfg.comparison_points,
            'n_accepted_cells': n_accepted,
            'accepted_fraction': (n_accepted / n_cells_run) if n_cells_run else 0.0,
        })
    bench = pd.DataFrame(rows).sort_values('preset').reset_index(drop=True)
    coarse = bench[bench['preset'] == 'coarse'].iloc[0]
    default = bench[bench['preset'] == 'default'].iloc[0]
    if coarse['accepted_fraction'] >= default['accepted_fraction'] - 0.10 and coarse['elapsed_s'] < default['elapsed_s']:
        recommended = 'coarse'
    else:
        recommended = 'default'
    bench['recommended_default'] = recommended
    return bench


def run_step04_cell_specific_multisweep(project_root: Path | str, config: Step04Config | None = None) -> dict[str, pd.DataFrame | dict[str, object]]:
    project_root = _project_root_from(project_root)
    config = config or Step04Config()
    output_dir = _ensure_dir(project_root / 'outputs' / OUTPUT_SUBDIR)
    atf_dir = project_root / 'data' / '2_K+ Pumps Data'
    legacy_dir = project_root / 'data' / '1_Initial_xp_fit'
    cache_key = (str(project_root), 'feature_table')
    if cache_key in _CACHE:
        feature_df, cell_dict = _CACHE[cache_key]
    else:
        feature_df, cell_dict = build_feature_table(atf_dir)
        _CACHE[cache_key] = (feature_df, cell_dict)
    rel_key = (str(project_root), 'reliability')
    if rel_key in _CACHE:
        reliability_df = _CACHE[rel_key]
    else:
        reliability_df = compute_reliability_weights(feature_df)
        _CACHE[rel_key] = reliability_df
    thr_key = (str(project_root), 'thresholds_full')
    if thr_key in _CACHE:
        threshold_df = _CACHE[thr_key]
    else:
        threshold_df = build_threshold_table(feature_df, reliability_df, exclude_file_id=None)
        _CACHE[thr_key] = threshold_df
    seed_key = (str(project_root), 'seed_summary')
    if seed_key in _CACHE:
        seed_summary = _CACHE[seed_key]
    else:
        seed_summary = compute_effective_seed_summary(legacy_dir)
        _CACHE[seed_key] = seed_summary
    seed_map = _make_condition_seed_map(seed_summary)

    ordered_cells = select_cells_for_run(cell_dict.values(), config.max_cells, config.cell_selection_mode)

    rng = np.random.default_rng(config.random_seed)
    accepted_rows: list[pd.DataFrame] = []
    accepted_sweep_rows: list[pd.DataFrame] = []
    accepted_trace_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for cell in ordered_cells:
        accepted_df, sweep_df, trace_df = fit_cell(cell, seed_map, feature_df, reliability_df, config, rng)
        accepted_rows.append(accepted_df)
        accepted_sweep_rows.append(sweep_df)
        accepted_trace_rows.append(trace_df)
        best = accepted_df.iloc[0]
        status_rows.append(
            {
                'file_id': best['file_id'],
                'file': best['file'],
                'region': best['region'],
                'condition': best['condition'],
                'accepted_candidate_count': int(accepted_df['accepted'].sum()),
                'best_candidate_id': int(best['candidate_id']),
                'best_objective': float(best['objective']),
                'best_mean_trace_nrmse': float(best['mean_trace_nrmse']),
                'best_mean_weighted_pass_fraction': float(best['mean_weighted_pass_fraction']),
                'status': 'accepted' if bool(best['accepted']) else 'rejected_but_ranked',
            }
        )

    accepted_candidates = pd.concat(accepted_rows, ignore_index=True) if accepted_rows else pd.DataFrame()
    accepted_sweeps = pd.concat(accepted_sweep_rows, ignore_index=True) if accepted_sweep_rows else pd.DataFrame()
    accepted_feature_contracts = pd.concat(accepted_trace_rows, ignore_index=True) if accepted_trace_rows else pd.DataFrame()
    fit_status = pd.DataFrame(status_rows).sort_values(['condition', 'region', 'file_id']).reset_index(drop=True)
    accepted_summary = (
        fit_status.groupby(['condition', 'region'], as_index=False)
        .agg(
            n_cells=('file_id', 'nunique'),
            n_accepted_cells=('status', lambda s: int((s == 'accepted').sum())),
            median_best_pass_fraction=('best_mean_weighted_pass_fraction', 'median'),
            median_best_trace_nrmse=('best_mean_trace_nrmse', 'median'),
        )
    )
    analysis_summary = {
        'step_name': 'Step 04 cell-specific six-sweep fitting and accepted ensemble construction',
        'n_cells_total': int(feature_df['file_id'].nunique()),
        'n_cells_run': int(fit_status['file_id'].nunique()),
        'n_accepted_cells': int((fit_status['status'] == 'accepted').sum()),
        'n_candidates_per_cell': int(config.n_candidates),
        'max_accepted_per_cell': int(config.max_accepted_per_cell),
        'cell_selection_mode': str(config.cell_selection_mode),
        'feature_mean_pass_threshold': float(config.feature_mean_pass_threshold),
        'max_mean_trace_nrmse': float(config.max_mean_trace_nrmse),
        'conditions_run': sorted(fit_status['condition'].dropna().unique().tolist()),
        'regions_run': sorted(fit_status['region'].dropna().unique().tolist()),
        'current_contract': VALID_CURRENTS,
        'outputs_subdir': OUTPUT_SUBDIR,
        'performance_benchmark_written': bool(config.write_outputs),
    }

    if config.write_outputs:
        feature_out = _ensure_dir(project_root / 'outputs' / 'features')
        feature_df.to_csv(feature_out / 'feature_table_by_sweep.csv', index=False)
        reliability_df.to_csv(feature_out / 'feature_reliability_weights.csv', index=False)
        threshold_df.to_csv(feature_out / 'condition_region_sweep_thresholds.csv', index=False)
        (
            feature_df.groupby(['region', 'condition'], as_index=False)
            .agg(n_cells=('file_id', 'nunique'))
            .to_csv(feature_out / 'region_condition_cell_counts.csv', index=False)
        )
        accepted_candidates.to_csv(output_dir / 'accepted_candidates.csv', index=False)
        accepted_sweeps.to_csv(output_dir / 'accepted_sweep_scores.csv', index=False)
        accepted_feature_contracts.to_csv(output_dir / 'accepted_feature_contracts.csv', index=False)
        fit_status.to_csv(output_dir / 'fit_status_by_cell.csv', index=False)
        accepted_summary.to_csv(output_dir / 'accepted_ensemble_summary.csv', index=False)
        seed_summary.to_csv(output_dir / 'seed_summary_by_condition.csv', index=False)
        bench = benchmark_step04_configs(project_root, config)
        bench.to_csv(output_dir / 'performance_benchmark.csv', index=False)
        analysis_summary['benchmark_recommended_default'] = str(bench['recommended_default'].iloc[0])
        with open(output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2)

    return {
        'feature_table_by_sweep': feature_df,
        'feature_reliability_weights': reliability_df,
        'condition_region_sweep_thresholds': threshold_df,
        'accepted_candidates': accepted_candidates,
        'accepted_sweep_scores': accepted_sweeps,
        'accepted_feature_contracts': accepted_feature_contracts,
        'fit_status_by_cell': fit_status,
        'accepted_ensemble_summary': accepted_summary,
        'analysis_summary': analysis_summary,
        'seed_summary_by_condition': seed_summary,
        'cell_dict': cell_dict,
    }
