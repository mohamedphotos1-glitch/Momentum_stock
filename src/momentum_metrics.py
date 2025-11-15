"""Computation of metrics and scoring for the momentum scanner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config_profiles import ProfileParams
from .utils_logging import get_logger

LOGGER = get_logger(__name__)

LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_12M = 252


def _pct_return(series: pd.Series, periods: int) -> Optional[float]:
    if len(series) < periods + 1:
        return None
    old_price = series.iloc[-periods - 1]
    if old_price == 0:
        return None
    return series.iloc[-1] / old_price - 1


def compute_all_metrics(price_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Calculate raw metrics used across profiles."""

    if price_df is None or price_df.empty:
        return None

    close = price_df["Close"].dropna()
    volume = price_df["Volume"].dropna()
    if close.empty or volume.empty:
        return None

    metrics: Dict[str, float] = {}
    metrics["last_close"] = float(close.iloc[-1])
    metrics["avg_vol_20"] = float(volume.tail(20).mean()) if len(volume) >= 1 else np.nan

    metrics["r_3m"] = _pct_return(close, LOOKBACK_3M)
    metrics["r_6m"] = _pct_return(close, LOOKBACK_6M)
    metrics["r_12m"] = _pct_return(close, LOOKBACK_12M)

    window = min(len(close), LOOKBACK_12M)
    last_year = close.tail(window)
    metrics["high_1y"] = float(last_year.max())
    metrics["low_1y"] = float(last_year.min())
    if metrics["high_1y"] == 0:
        metrics["drawdown_1y"] = None
        metrics["close_to_high_1y"] = None
    else:
        metrics["drawdown_1y"] = 1 - metrics["low_1y"] / metrics["high_1y"]
        metrics["close_to_high_1y"] = metrics["last_close"] / metrics["high_1y"]

    metrics["ma50"] = float(close.tail(50).mean()) if len(close) >= 50 else None
    metrics["ma150"] = float(close.tail(150).mean()) if len(close) >= 150 else None

    return metrics


def _flag(value: bool) -> str:
    return "OK" if value else "KO"


def score_with_profile(metrics: Dict[str, float], profile: ProfileParams) -> Dict[str, float]:
    """Evaluate metrics using the thresholds defined in a profile."""

    required_fields = [
        "last_close",
        "avg_vol_20",
        "r_3m",
        "r_6m",
        "r_12m",
        "drawdown_1y",
        "close_to_high_1y",
        "ma50",
        "ma150",
    ]
    for field in required_fields:
        if metrics.get(field) is None or pd.isna(metrics.get(field)):
            LOGGER.debug("Missing %s for scoring, skipping ticker", field)
            return {}

    results: Dict[str, float] = {
        "last_close": metrics["last_close"],
        "avg_vol_20": metrics["avg_vol_20"],
        "r_3m": metrics["r_3m"] * 100,
        "r_6m": metrics["r_6m"] * 100,
        "r_12m": metrics["r_12m"] * 100,
        "drawdown_1y": metrics["drawdown_1y"] * 100,
        "close_to_high_1y": metrics["close_to_high_1y"] * 100,
        "ma50": metrics["ma50"],
        "ma150": metrics["ma150"],
    }

    checks = {
        "price_range_ok": profile.price_min <= metrics["last_close"] <= profile.price_max,
        "volume_ok": metrics["avg_vol_20"] >= profile.avg_vol_20_min,
        "r_3m_ok": metrics["r_3m"] >= profile.r_3m_min,
        "r_6m_ok": metrics["r_6m"] >= profile.r_6m_min,
        "r_12m_ok": metrics["r_12m"] >= profile.r_12m_min,
        "drawdown_1y_ok": metrics["drawdown_1y"] <= profile.drawdown_1y_max,
        "close_to_high_1y_ok": metrics["close_to_high_1y"] >= profile.close_to_high_1y_min,
        "ma_trend_ok": metrics["last_close"] >= metrics["ma50"] >= metrics["ma150"],
    }

    score = sum(1 for value in checks.values() if value)
    for key, value in checks.items():
        results[key] = _flag(value)

    results["score"] = score
    return results
