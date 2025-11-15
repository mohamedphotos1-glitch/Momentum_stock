"""Profile configuration for the momentum scanner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ProfileParams:
    """Container for the momentum filters used by the scanner."""

    name: str
    price_min: float
    price_max: float
    avg_vol_20_min: float
    r_3m_min: float
    r_6m_min: float
    r_12m_min: float
    drawdown_1y_max: float
    close_to_high_1y_min: float
    score_min: int


PROFILE_B = ProfileParams(
    name="B",
    price_min=0.5,
    price_max=80.0,
    avg_vol_20_min=100_000,
    r_3m_min=0.20,
    r_6m_min=0.40,
    r_12m_min=0.80,
    drawdown_1y_max=0.40,
    close_to_high_1y_min=0.80,
    score_min=6,
)

PROFILE_C = ProfileParams(
    name="C",
    price_min=1.0,
    price_max=80.0,
    avg_vol_20_min=200_000,
    r_3m_min=0.35,
    r_6m_min=0.70,
    r_12m_min=1.20,
    drawdown_1y_max=0.30,
    close_to_high_1y_min=0.90,
    score_min=7,
)


PROFILES: Dict[str, ProfileParams] = {
    PROFILE_B.name: PROFILE_B,
    PROFILE_C.name: PROFILE_C,
}
