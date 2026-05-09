"""Shared condition/protocol naming contracts.

The ATF dataset uses ``MFA_BA`` for MFA plus barium, while legacy SQLite
protocol assets use ``BARIUM`` for the same model protocol.  Keep that mapping
in one small module so scientific tables and simulation calls do not drift.
"""

from __future__ import annotations

CONDITION_ALIASES: dict[str, str] = {
    "CONTROL": "CONTROL",
    "CTRL": "CONTROL",
    "MFA": "MFA",
    "MFA_BA": "MFA_BA",
    "MFA-BA": "MFA_BA",
    "MFA+BA": "MFA_BA",
    "BARIUM": "MFA_BA",
    "BA": "MFA_BA",
}


def canonical_condition(value: str) -> str:
    """Return the dataset-facing condition name: CONTROL, MFA, or MFA_BA."""

    key = str(value).strip().upper().replace(" ", "_")
    if key in CONDITION_ALIASES:
        return CONDITION_ALIASES[key]
    if "MFA" in key and "BA" in key:
        return "MFA_BA"
    raise ValueError(f"Unknown condition={value!r}")


def protocol_condition(value: str) -> str:
    """Return the model/legacy protocol name for a condition."""

    condition = canonical_condition(value)
    return "BARIUM" if condition == "MFA_BA" else condition
