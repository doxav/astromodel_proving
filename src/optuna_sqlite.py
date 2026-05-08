"""Direct Optuna SQLite readers used by the reviewer-response pipeline."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from .astro_model import normalize_flat_params

DB_NAME_RE = re.compile(r"(?P<condition>CONTROL|MFA|BARIUM)_(?P<current_na>50|75|100|125|150|175)nA\.db$")
STUDY_NAME_RE = re.compile(
    r"(?P<condition>CONTROL|MFA|BARIUM)_(?P<current_na>50|75|100|125|150|175)nA_"
    r"(?P<target_mean_mode>centered|default|centered_scaled|centered_l2|centered_combined)_"
    r"(?P<objective_loss_type>L2|COMBINED|L1|HUBER|LOG_COSH)_"
    r"(?P<n_target_points>\d+)t"
)


@dataclass(frozen=True)
class StudySpec:
    condition: str
    current_na: int
    target_mean_mode: str
    objective_loss_type: str
    n_target_points: int
    study_name: str


@dataclass(frozen=True)
class TrialRecord:
    db_name: str
    study_name: str
    condition: str
    current_na: int
    trial_id: int
    trial_number: int
    objective: float
    params: Dict[str, Any]


def parse_db_name(name_or_path: str | Path) -> tuple[str, int]:
    name = Path(name_or_path).name
    match = DB_NAME_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected DB name: {name}")
    return match.group("condition"), int(match.group("current_na"))


def parse_study_name(study_name: str) -> StudySpec:
    match = STUDY_NAME_RE.search(study_name)
    if not match:
        raise ValueError(f"Could not parse study name: {study_name}")
    return StudySpec(
        condition=match.group("condition"),
        current_na=int(match.group("current_na")),
        target_mean_mode=match.group("target_mean_mode"),
        objective_loss_type=match.group("objective_loss_type"),
        n_target_points=int(match.group("n_target_points")),
        study_name=study_name,
    )


def sqlite_tables(db_path: str | Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _decode_distribution_value(raw_value: float, distribution_json: str) -> Any:
    spec = json.loads(distribution_json)
    if spec.get("name") == "CategoricalDistribution":
        choices = spec.get("attributes", {}).get("choices", [])
        index = int(raw_value)
        if index < 0 or index >= len(choices):
            raise IndexError(f"Categorical index {index} out of range for {choices}")
        return choices[index]
    return float(raw_value)


def _read_trial_params(conn: sqlite3.Connection, trial_id: int) -> Dict[str, Any]:
    rows = conn.execute(
        """
        SELECT param_name, param_value, distribution_json
        FROM trial_params
        WHERE trial_id = ?
        ORDER BY param_name ASC
        """,
        (trial_id,),
    ).fetchall()
    params_from_rows = {str(name): _decode_distribution_value(value, distribution_json) for name, value, distribution_json in rows}
    fixed = conn.execute(
        """
        SELECT value_json
        FROM trial_system_attributes
        WHERE trial_id = ? AND key = 'fixed_params'
        LIMIT 1
        """,
        (trial_id,),
    ).fetchone()
    if fixed:
        params_from_fixed = json.loads(fixed[0])
        params_from_rows.update(params_from_fixed)
    return normalize_flat_params(params_from_rows)


def read_db_study_summary(db_path: str | Path) -> Dict[str, Any]:
    db_path = Path(db_path)
    condition, current_na = parse_db_name(db_path)
    required_tables = {"studies", "trials", "trial_values", "trial_params"}
    tables = sqlite_tables(db_path)
    missing = sorted(required_tables - tables)
    if missing:
        raise ValueError(f"{db_path.name} is missing required tables: {missing}")
    with sqlite3.connect(str(db_path)) as conn:
        study_name = str(conn.execute("SELECT study_name FROM studies LIMIT 1").fetchone()[0])
        spec = parse_study_name(study_name)
        n_trials = int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
        n_complete = int(conn.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'").fetchone()[0])
        best = conn.execute(
            """
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value_type = 'FINITE'
            ORDER BY tv.value ASC, t.number ASC
            LIMIT 1
            """
        ).fetchone()
    if best is None:
        raise ValueError(f"No complete finite trial found in {db_path.name}")
    return {
        "db_name": db_path.name,
        "study_name": study_name,
        "condition": condition,
        "current_na": current_na,
        "target_mean_mode": spec.target_mean_mode,
        "objective_loss_type": spec.objective_loss_type,
        "n_target_points": spec.n_target_points,
        "n_trials": n_trials,
        "n_complete": n_complete,
        "best_trial_number": int(best[0]),
        "best_objective": float(best[1]),
    }


def read_best_trial(db_path: str | Path) -> TrialRecord:
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        study_name = str(conn.execute("SELECT study_name FROM studies LIMIT 1").fetchone()[0])
        spec = parse_study_name(study_name)
        row = conn.execute(
            """
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value_type = 'FINITE'
            ORDER BY tv.value ASC, t.number ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise ValueError(f"No complete finite trial found in {db_path.name}")
        trial_id, trial_number, objective = int(row[0]), int(row[1]), float(row[2])
        params = _read_trial_params(conn, trial_id)
    return TrialRecord(
        db_name=db_path.name,
        study_name=study_name,
        condition=spec.condition,
        current_na=spec.current_na,
        trial_id=trial_id,
        trial_number=trial_number,
        objective=objective,
        params=params,
    )


def read_top_trials(db_path: str | Path, top_n: int = 10) -> list[TrialRecord]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        study_name = str(conn.execute("SELECT study_name FROM studies LIMIT 1").fetchone()[0])
        spec = parse_study_name(study_name)
        rows = conn.execute(
            """
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value_type = 'FINITE'
            ORDER BY tv.value ASC, t.number ASC
            LIMIT ?
            """,
            (int(top_n),),
        ).fetchall()
        records: list[TrialRecord] = []
        for trial_id, trial_number, objective in rows:
            params = _read_trial_params(conn, int(trial_id))
            records.append(
                TrialRecord(
                    db_name=db_path.name,
                    study_name=study_name,
                    condition=spec.condition,
                    current_na=spec.current_na,
                    trial_id=int(trial_id),
                    trial_number=int(trial_number),
                    objective=float(objective),
                    params=params,
                )
            )
    return records


def top_trials_dataframe(db_paths: Sequence[str | Path], top_n: int = 10) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for db_path in db_paths:
        for record in read_top_trials(db_path, top_n=top_n):
            rows.append(
                {
                    "db_name": record.db_name,
                    "study_name": record.study_name,
                    "condition": record.condition,
                    "current_na": record.current_na,
                    "trial_id": record.trial_id,
                    "trial_number": record.trial_number,
                    "objective": record.objective,
                    **record.params,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["condition", "current_na", "objective", "trial_number"], ascending=[True, True, True, True]).reset_index(drop=True)
    return df
