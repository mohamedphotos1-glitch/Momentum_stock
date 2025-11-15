"""Command line interface for the weekly momentum scanner."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from .config_profiles import PROFILES, ProfileParams
from .momentum_metrics import compute_all_metrics, score_with_profile
from .price_loader import fetch_price_history
from .universe_loader import load_universe_from_csv
from .utils_logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly US momentum scanner")
    parser.add_argument("--profile", choices=PROFILES.keys(), required=True, help="Profil de momentum à utiliser")
    parser.add_argument(
        "--universe",
        default="univers_us.csv",
        help="Nom du fichier d'univers dans le dossier data/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de tickers pour des tests rapides",
    )
    return parser.parse_args()


def _save_results(df: pd.DataFrame, profile: ProfileParams) -> Path:
    results_dir = Path("momentum_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"momentum_weekly_profile_{profile.name}.csv"
    output_path = results_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def _summarize(df: pd.DataFrame, profile: ProfileParams, requested_count: int) -> None:
    processed = len(df)
    passed = int((df["score"] >= profile.score_min).sum()) if processed else 0
    LOGGER.info(
        "Scan terminé - Demandés: %s | Avec données: %s | Profil %s retenus: %s",
        requested_count,
        processed,
        profile.name,
        passed,
    )

    if processed:
        distribution = df["score"].value_counts().sort_index()
        LOGGER.info("Distribution des scores:\n%s", distribution.to_string())


def run_scan(profile_key: str, universe_file: str, limit: int | None = None) -> None:
    profile = PROFILES[profile_key]
    tickers = load_universe_from_csv(universe_file)
    if not tickers:
        LOGGER.error("Aucun ticker dans l'univers, arrêt du scan")
        return

    if limit is not None:
        tickers = tickers[:limit]
    requested_count = len(tickers)

    scan_date = datetime.utcnow().date().isoformat()
    results: List[dict] = []

    for ticker in tickers:
        price_data = fetch_price_history(ticker)
        if price_data is None:
            continue
        metrics = compute_all_metrics(price_data)
        if not metrics:
            continue
        scored = score_with_profile(metrics, profile)
        if not scored:
            continue
        scored.update({
            "symbol": ticker,
            "scan_date": scan_date,
            "profile": profile.name,
        })
        results.append(scored)

    if not results:
        LOGGER.warning("Aucun résultat exploitable pour le profil %s", profile.name)
        return

    df = pd.DataFrame(results)
    df = df[
        [
            "scan_date",
            "profile",
            "symbol",
            "last_close",
            "avg_vol_20",
            "price_range_ok",
            "volume_ok",
            "r_3m",
            "r_3m_ok",
            "r_6m",
            "r_6m_ok",
            "r_12m",
            "r_12m_ok",
            "drawdown_1y",
            "drawdown_1y_ok",
            "close_to_high_1y",
            "close_to_high_1y_ok",
            "ma50",
            "ma150",
            "ma_trend_ok",
            "score",
        ]
    ].sort_values(by="score", ascending=False)

    output_path = _save_results(df, profile)
    LOGGER.info("Résultats sauvegardés dans %s", output_path)
    _summarize(df, profile, requested_count=requested_count)


def main() -> None:
    args = _parse_args()
    run_scan(profile_key=args.profile, universe_file=args.universe, limit=args.limit)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
