from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DB_NAME_RE = re.compile(r"(?P<condition>CONTROL|MFA|BARIUM)_(?P<current_na>50|75|100|125|150|175)nA\.db$")


def _decode_distribution_value(raw_value: float, distribution_json: str) -> Any:
    spec = json.loads(distribution_json)
    if spec.get("name") == "CategoricalDistribution":
        return spec.get("attributes", {}).get("choices", [])[int(raw_value)]
    return float(raw_value)


def _normalize(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    params.setdefault("wo_middle", 1.0)
    params.setdefault("eps_middle", 1.0)
    params.setdefault("w_a", 2000.0)
    params.setdefault("switching_function", "sigmoid")
    return params


def _best_trial(db_path: Path) -> dict[str, Any]:
    m = DB_NAME_RE.fullmatch(db_path.name)
    if not m:
        raise ValueError(f"Unexpected DB name {db_path.name}")
    with sqlite3.connect(str(db_path)) as conn:
        study_name = str(conn.execute("SELECT study_name FROM studies LIMIT 1").fetchone()[0])
        trial_id, trial_number, objective = conn.execute(
            "SELECT t.trial_id, t.number, tv.value FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE' AND tv.value_type='FINITE' ORDER BY tv.value ASC, t.number ASC LIMIT 1"
        ).fetchone()
        params_rows = conn.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=? ORDER BY param_name ASC", (trial_id,)).fetchall()
        params = {str(name): _decode_distribution_value(value, dj) for name, value, dj in params_rows}
        fixed = conn.execute("SELECT value_json FROM trial_system_attributes WHERE trial_id=? AND key='fixed_params' LIMIT 1", (trial_id,)).fetchone()
        if fixed:
            params.update(json.loads(fixed[0]))
    return {
        "db_name": db_path.name,
        "study_name": study_name,
        "condition": m.group("condition"),
        "current_na": int(m.group("current_na")),
        "trial_id": int(trial_id),
        "trial_number": int(trial_number),
        "objective": float(objective),
        "params": _normalize(params),
    }


def effective_parameter_summary(initial_fit_dir: str | Path) -> pd.DataFrame:
    rows = []
    for db_path in sorted(Path(initial_fit_dir).glob("*.db")):
        best = _best_trial(db_path)
        p = best["params"]
        w_a = float(p.get("w_a", 2000.0))
        sig_a = 1600.0
        F = 96485.0
        rows.append({
            "db_name": best["db_name"],
            "study_name": best["study_name"],
            "condition": best["condition"],
            "current_na": best["current_na"],
            "trial_id": best["trial_id"],
            "trial_number": best["trial_number"],
            "objective": best["objective"],
            "P_gap_eff": float(p.get("d", 1.0)) * float(p.get("pk", 0.0)),
            "gamma_t_eff": float(p.get("gt", 0.0)) * sig_a / (w_a * F),
            "gamma_s_eff": float(p.get("gs", 0.0)) * sig_a / (w_a * F),
            "volume_ratio_wa_wo": w_a / max(float(p.get("wo", 1500.0)), 1e-12),
        })
    return pd.DataFrame(rows).sort_values(["condition", "current_na"]).reset_index(drop=True)
