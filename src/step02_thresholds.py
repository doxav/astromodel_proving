from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .atf_step02 import (
    build_condition_region_sweep_thresholds,
    build_feature_reliability_weights,
    build_feature_table,
    build_region_condition_cell_counts,
)


@dataclass
class Step02Config:
    project_root: Path
    atf_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    def resolve(self) -> "Step02Config":
        if self.atf_dir is None:
            self.atf_dir = self.project_root / "data" / "2_K+ Pumps Data"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs" / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


def run_step02_thresholds(project_root: Path, config: Optional[Step02Config] = None) -> Dict[str, pd.DataFrame]:
    cfg = (config or Step02Config(project_root=project_root)).resolve()
    feature_table = build_feature_table(cfg.atf_dir)
    counts = build_region_condition_cell_counts(feature_table)
    reliability = build_feature_reliability_weights(feature_table)
    thresholds = build_condition_region_sweep_thresholds(feature_table, reliability)

    feature_table.to_csv(cfg.output_dir / "feature_table_by_sweep.csv", index=False)
    counts.to_csv(cfg.output_dir / "region_condition_cell_counts.csv", index=False)
    reliability.to_csv(cfg.output_dir / "feature_reliability_weights.csv", index=False)
    thresholds.to_csv(cfg.output_dir / "condition_region_sweep_thresholds.csv", index=False)

    summary = {
        "n_feature_rows": int(len(feature_table)),
        "n_cells": int(feature_table["file_id"].nunique()),
        "n_threshold_rows": int(len(thresholds)),
        "regions": sorted(feature_table["region"].dropna().unique().tolist()),
        "conditions": sorted(feature_table["condition"].dropna().unique().tolist()),
    }
    (cfg.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "feature_table_by_sweep": feature_table,
        "region_condition_cell_counts": counts,
        "feature_reliability_weights": reliability,
        "condition_region_sweep_thresholds": thresholds,
    }
