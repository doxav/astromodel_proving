from __future__ import annotations

from pathlib import Path

from src.step04_cell_specific_multisweep import Step04Config, compute_effective_seed_summary, run_step04_cell_specific_multisweep

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_step04_seed_summary_has_expected_conditions() -> None:
    df = compute_effective_seed_summary(PROJECT_ROOT / 'data' / '1_Initial_xp_fit')
    assert set(df['condition']) == {'CONTROL', 'MFA', 'BARIUM'}
    assert {'P_gap_eff', 'gamma_t_eff', 'gamma_s_eff', 'volume_ratio_wa_wo'}.issubset(df.columns)


def test_step04_more_candidates_does_not_worsen_best_objective() -> None:
    small = run_step04_cell_specific_multisweep(PROJECT_ROOT, Step04Config(random_seed=11, n_candidates=3, max_cells=1, write_outputs=False))
    larger = run_step04_cell_specific_multisweep(PROJECT_ROOT, Step04Config(random_seed=11, n_candidates=5, max_cells=1, write_outputs=False))
    best_small = float(small['accepted_candidates']['objective'].min())
    best_large = float(larger['accepted_candidates']['objective'].min())
    assert best_large <= best_small + 1e-9
