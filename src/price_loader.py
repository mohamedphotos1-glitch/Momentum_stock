"""Download and normalize price data via yfinance."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .utils_logging import get_logger

LOGGER = get_logger(__name__)


def _extract_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)
    return data


def fetch_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Return a DataFrame with OHLCV columns for *ticker*."""

    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("yfinance download failed for %s: %s", ticker, exc)
        return None

    if data is None or data.empty:
        LOGGER.warning("No market data returned for %s", ticker)
        return None

    data = _extract_columns(data, ticker)

    close_col = None
    for column in ("Adj Close", "Close"):
        if column in data.columns:
            close_col = column
            break

    if close_col is None or "Volume" not in data.columns:
        LOGGER.warning("Missing Close/Volume for %s", ticker)
        return None

    selected_cols = [col for col in ("Open", "High", "Low", close_col, "Volume") if col in data.columns]
    if {"Open", "High", "Low", "Volume"}.issubset(set(selected_cols)) is False:
        LOGGER.warning("Missing OHLCV data for %s", ticker)
        return None

    cleaned = data[selected_cols].rename(columns={close_col: "Close"}).copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.dropna()

    if cleaned.empty:
        LOGGER.warning("Price data empty after cleaning for %s", ticker)
        return None

    for column in cleaned.columns:
        cleaned[column] = cleaned[column].astype(float)
    return cleaned


if __name__ == "__main__":  # pragma: no cover - manual usage helper
    print(fetch_price_history("AAPL").tail())
