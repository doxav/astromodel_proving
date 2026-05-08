from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any


@lru_cache(maxsize=1)
def load_filtered_runtime() -> Dict[str, Any]:
    vendor_path = Path(__file__).resolve().parents[1] / "vendor" / "Filtered_basline_sweep_1_.py"
    source = vendor_path.read_text(encoding="utf-8")
    cutoff_marker = "# %% cell 13 code"
    if cutoff_marker in source:
        source = source[: source.index(cutoff_marker)]
    namespace: Dict[str, Any] = {}
    exec(source, namespace)
    return namespace
