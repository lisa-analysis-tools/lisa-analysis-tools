"""
Annotated types for the interface
"""
from __future__ import annotations

from typing import Annotated, Any
import os
from pathlib import Path
from pydantic import BeforeValidator, AfterValidator, ValidationInfo, PositiveFloat

from lisatools.utils.constants import YRSID_SI

_TIME_UNITS = {
    "s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0, "week": 7 * 86400.0,
    "month": YRSID_SI / 12.0, "yr": YRSID_SI,
}

def _to_seconds(value: Any) -> Any:
    """``"9 months"`` / ``"200 h"`` -> seconds; plain numbers are already seconds."""
    if not isinstance(value, str):
        return value
    number, _, unit = value.strip().partition(" ")
    unit = unit.strip()
    if unit not in _TIME_UNITS and unit.endswith("s") and unit[:-1] in _TIME_UNITS:
        unit = unit[:-1]  # plural: "months", "days"
    if unit not in _TIME_UNITS:
        raise ValueError(f"unknown time unit in {value!r}; use one of {sorted(_TIME_UNITS)}.")
    return float(number) * _TIME_UNITS[unit]

Duration = Annotated[PositiveFloat, BeforeValidator(_to_seconds)]

def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    """Relative paths are relative to the run folder (passed as validation context)."""
    value = value.expanduser()
    run_dir = (info.context or {}).get("run_dir")
    if run_dir is None or value.is_absolute():
        return value
    return Path(os.path.normpath(run_dir / value))


RunPath = Annotated[Path, AfterValidator(_resolve_path)]
