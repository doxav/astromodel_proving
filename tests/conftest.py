from __future__ import annotations
import os, sys
from pathlib import Path
import pytest

def _project_root() -> Path:
    env = os.environ.get('ASTROMODEL_PROJECT_ROOT')
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]

PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(scope='session')
def project_root() -> Path:
    return PROJECT_ROOT
