"""Reusable reviewer-response modules for the astrocytic K-buffering project.

This package is intentionally initialized as a small `src/` layer that merges:

- existing fitting/protocol code from `Filtered_basline_sweep (1).ipynb` and
  `AFT_uncut_end_BARIUM_Opt_v2 (1).ipynb`; and
- reusable architectural patterns from `astrosim-master`.

The modules are not meant to replace the ATF/Optuna notebooks immediately.  They
provide stable target functions for incremental migration and reviewer-facing
analyses: hidden-current extraction, mechanistic summaries, perturbation checks,
and compensation figures.
"""

from .astro_model import (
    build_paramdict,
    compute_rhs_and_currents,
    simulate_odeint,
    simulate_rk4_numba,
    simulate_with_hidden_outputs,
)
from .mechanisms import (
    compute_flux_summary,
    compute_gap_kir_ratio,
    compute_proxy_validity,
    select_mechanistically_diverse_representatives,
)

__all__ = [
    "build_paramdict",
    "compute_rhs_and_currents",
    "simulate_odeint",
    "simulate_rk4_numba",
    "simulate_with_hidden_outputs",
    "compute_flux_summary",
    "compute_gap_kir_ratio",
    "compute_proxy_validity",
    "select_mechanistically_diverse_representatives",
]
