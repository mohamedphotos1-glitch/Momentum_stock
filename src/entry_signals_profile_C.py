"""Post-scan analysis that generates short-term entry signals for profile C."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .price_loader import fetch_price_history

LOG_FILE = Path("logs/entry_signals_profile_C.log")


def _setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("entry_signals_profile_C")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


LOGGER = _setup_logger()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse court terme des résultats du scanner C")
    parser.add_argument(
        "--input",
        default="momentum_results/momentum_weekly_profile_C.csv",
        help="Fichier CSV produit par le scanner hebdo (profil C)",
    )
    parser.add_argument(
        "--output",
        default="momentum_results/entry_signals_profile_C.csv",
        help="Chemin du CSV de signaux court terme",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=6,
        help="Score minimum du scanner C pour intégrer la watchlist",
    )
    return parser.parse_args()


def _load_watchlist(input_path: Path, min_score: int) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {"profile", "symbol", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV d'entrée: {sorted(missing)}")

    filtered = df[(df["profile"] == "C") & (df["score"] >= min_score)].copy()
    return filtered


def _latest_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _compute_atr(price_df: pd.DataFrame, period: int = 14) -> Optional[float]:
    high = price_df["High"]
    low = price_df["Low"]
    close = price_df["Close"]
    prev_close = close.shift(1)
    tr_components = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(period).mean()
    return _latest_value(atr)


def _compute_bb_width_pct(close: pd.Series) -> Optional[float]:
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ma_val = _latest_value(ma20)
    std_val = _latest_value(std20)
    close_val = _latest_value(close)
    if None in (ma_val, std_val, close_val) or close_val == 0:
        return None
    width = (ma_val + 2 * std_val) - (ma_val - 2 * std_val)
    return float(width / close_val)


def _compute_dist_high_20(high: pd.Series, close: pd.Series) -> Optional[float]:
    high_20 = high.rolling(20).max()
    high_val = _latest_value(high_20)
    close_val = _latest_value(close)
    if None in (high_val, close_val) or high_val == 0:
        return None
    return float((high_val - close_val) / high_val)


def _compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return _latest_value(rsi)


def _compute_moving_average(close: pd.Series, window: int) -> Optional[float]:
    if len(close) < window:
        return None
    return _latest_value(close.rolling(window).mean())


def _compute_indicators(price_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if price_df is None or price_df.empty or len(price_df) < 60:
        return None

    price_df = price_df.sort_index()
    close = price_df["Close"]
    volume = price_df["Volume"]
    high = price_df["High"]
    low = price_df["Low"]

    vol_ma20 = _latest_value(volume.rolling(20).mean())
    vol_ratio = None
    volume_today = _latest_value(volume)
    if vol_ma20 is not None and vol_ma20 > 0 and volume_today is not None:
        vol_ratio = float(volume_today / vol_ma20)

    atr14 = _compute_atr(price_df)
    close_today = _latest_value(close)
    atr_pct = None
    if atr14 is not None and close_today is not None and close_today != 0:
        atr_pct = float(atr14 / close_today)

    bb_width_pct = _compute_bb_width_pct(close)
    dist_high_20 = _compute_dist_high_20(high, close)
    rsi_14 = _compute_rsi(close)
    ma10 = _compute_moving_average(close, 10)
    ma20 = _compute_moving_average(close, 20)
    ma50 = _compute_moving_average(close, 50)

    indicators: Dict[str, float] = {
        "last_close_price": close_today if close_today is not None else np.nan,
        "volume_today": volume_today if volume_today is not None else np.nan,
        "vol_ratio": vol_ratio if vol_ratio is not None else np.nan,
        "atr14": atr14 if atr14 is not None else np.nan,
        "atr_pct": atr_pct if atr_pct is not None else np.nan,
        "bb_width_pct": bb_width_pct if bb_width_pct is not None else np.nan,
        "dist_high_20": dist_high_20 if dist_high_20 is not None else np.nan,
        "rsi_14": rsi_14 if rsi_14 is not None else np.nan,
        "ma10": ma10 if ma10 is not None else np.nan,
        "ma20": ma20 if ma20 is not None else np.nan,
        "ma50": ma50 if ma50 is not None else np.nan,
    }
    return indicators


def _points_volume(vol_ratio: float) -> int:
    if np.isnan(vol_ratio):
        return 0
    if vol_ratio >= 5:
        return 3
    if vol_ratio >= 3:
        return 2
    if vol_ratio >= 2:
        return 1
    return 0


def _points_atr(atr_pct: float) -> int:
    if np.isnan(atr_pct):
        return 0
    if atr_pct <= 0.02:
        return 2
    if atr_pct <= 0.04:
        return 1
    return 0


def _points_bb(bb_width_pct: float) -> int:
    if np.isnan(bb_width_pct):
        return 0
    if bb_width_pct <= 0.05:
        return 2
    if bb_width_pct <= 0.10:
        return 1
    return 0


def _points_dist(dist_high_20: float) -> int:
    if np.isnan(dist_high_20):
        return 0
    if dist_high_20 <= 0.01:
        return 2
    if dist_high_20 <= 0.03:
        return 1
    return 0


def _points_rsi(rsi_14: float) -> int:
    if np.isnan(rsi_14):
        return 0
    if 60 <= rsi_14 <= 75:
        return 1
    return 0


def _points_ma_alignment(close_price: float, ma10: float, ma20: float, ma50: float) -> int:
    values = (close_price, ma10, ma20, ma50)
    if any(pd.isna(v) for v in values):
        return 0
    if close_price > ma10 > ma20 > ma50:
        return 2
    if close_price > ma20 > ma50:
        return 1
    return 0


def _score_indicators(indicators: Dict[str, float]) -> int:
    close_price = indicators.get("last_close_price", np.nan)
    score = 0
    score += _points_volume(indicators.get("vol_ratio", np.nan))
    score += _points_atr(indicators.get("atr_pct", np.nan))
    score += _points_bb(indicators.get("bb_width_pct", np.nan))
    score += _points_dist(indicators.get("dist_high_20", np.nan))
    score += _points_rsi(indicators.get("rsi_14", np.nan))
    score += _points_ma_alignment(
        close_price,
        indicators.get("ma10", np.nan),
        indicators.get("ma20", np.nan),
        indicators.get("ma50", np.nan),
    )
    return score


def _score_to_probability(score: int) -> int:
    if score >= 9:
        return 80
    if score >= 7:
        return 60
    if score >= 5:
        return 40
    return 20


def _expected_window(indicators: Dict[str, float]) -> str:
    vol_ratio = indicators.get("vol_ratio", np.nan)
    dist_high_20 = indicators.get("dist_high_20", np.nan)
    bb_width_pct = indicators.get("bb_width_pct", np.nan)
    atr_pct = indicators.get("atr_pct", np.nan)

    if (not np.isnan(vol_ratio) and vol_ratio >= 3) and (not np.isnan(dist_high_20) and dist_high_20 <= 0.01):
        return "1-5j"
    if (not np.isnan(bb_width_pct) and bb_width_pct <= 0.05) or (not np.isnan(atr_pct) and atr_pct <= 0.02):
        return "5-20j"
    return "20-60j"


def _signal(probability: int, window: str) -> str:
    if probability >= 70 and window == "1-5j":
        return "BUY"
    if probability >= 50:
        return "WATCH"
    return "WAIT"


def _analyze_symbol(symbol: str, base_row: Dict[str, float]) -> Optional[Dict[str, float]]:
    LOGGER.info("Analyse du ticker %s", symbol)
    price_df = fetch_price_history(symbol, period="1y", interval="1d")
    if price_df is None:
        LOGGER.warning("Pas de données marché pour %s", symbol)
        return None

    indicators = _compute_indicators(price_df)
    if not indicators:
        LOGGER.warning("Données insuffisantes pour %s", symbol)
        return None

    score = _score_indicators(indicators)
    probability = _score_to_probability(score)
    window = _expected_window(indicators)
    signal = _signal(probability, window)

    enriched = {**base_row}
    enriched.update(indicators)
    enriched.update(
        {
            "explosion_score": score,
            "prob_explosion_pct": probability,
            "expected_window": window,
            "signal": signal,
        }
    )
    return enriched


def _process_watchlist(df: pd.DataFrame) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for symbol, group in df.groupby("symbol"):
        base_row = group.iloc[0].to_dict()
        analyzed = _analyze_symbol(symbol, base_row)
        if analyzed:
            results.append(analyzed)
    return results


def run_entry_signal_pipeline(input_path: Path, output_path: Path, min_score: int) -> Optional[Path]:
    try:
        watchlist = _load_watchlist(input_path, min_score)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Impossible de charger la watchlist: %s", exc)
        return None

    if watchlist.empty:
        LOGGER.warning("Aucun ticker éligible pour les signaux (profil C, score >= %s)", min_score)
        return None

    LOGGER.info("%s tickers à analyser", watchlist["symbol"].nunique())
    enriched_rows = _process_watchlist(watchlist)
    if not enriched_rows:
        LOGGER.warning("Aucun ticker n'a pu être analysé")
        return None

    results_df = pd.DataFrame(enriched_rows)
    results_df.sort_values(by=["explosion_score", "prob_explosion_pct"], ascending=False, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Signaux sauvegardés dans %s", output_path)
    return output_path


def main() -> None:  # pragma: no cover - CLI helper
    args = _parse_args()
    run_entry_signal_pipeline(Path(args.input), Path(args.output), args.min_score)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
