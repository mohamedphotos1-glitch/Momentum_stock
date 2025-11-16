"""Functions that load and clean the universe of tickers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .utils_logging import get_logger

LOGGER = get_logger(__name__)
SYMBOL_COLUMNS = [
    "Symbol",
    "NASDAQ Symbol",
    "ACT Symbol",
    "symbol",
    "Ticker",
    "ticker",
]


def _detect_separator(sample: str) -> str:
    if sample.count(";") > sample.count(","):
        return ";"
    return ","


def _clean_symbols(symbols: Iterable[str]) -> List[str]:
    cleaned = []
    for symbol in symbols:
        if symbol is None:
            continue
        sym = str(symbol).strip().upper()
        if not sym or sym in SYMBOL_COLUMNS:
            continue
        if any(ch in sym for ch in (" ", "^", "/", "\\", "$")):
            continue
        cleaned.append(sym)
    return sorted(set(cleaned))


def load_universe_from_csv(filename: str = "univers_us.csv", data_dir: Optional[Path] = None) -> List[str]:
    """Load and clean the list of tickers from a CSV file."""

    if data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
    csv_path = data_dir / filename
    if not csv_path.exists():
        LOGGER.error("Universe file not found: %s", csv_path)
        return []

    try:
        if csv_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(csv_path, dtype=str)
        else:
            with csv_path.open("r", encoding="utf-8") as handle:
                sample = handle.readline()
            separator = _detect_separator(sample)
            df = pd.read_csv(csv_path, sep=separator, engine="python", dtype=str)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to read universe file %s: %s", csv_path, exc)
        return []

    column = next((col for col in SYMBOL_COLUMNS if col in df.columns), None)
    if column is None:
        LOGGER.error("No symbol column found in %s. Columns=%s", csv_path, list(df.columns))
        return []

    tickers = _clean_symbols(df[column].tolist())
    LOGGER.info("Loaded %d tickers from %s", len(tickers), csv_path)
    return tickers


if __name__ == "__main__":  # pragma: no cover - manual usage helper
    symbols = load_universe_from_csv()
    print(f"Found {len(symbols)} symbols")
