"""
Portfolio tracker that ingests weekly weight files, validates allocations,
computes NAV paths, and stores the result as a CSV.

The script is derived from the logic in ``notebooks/baseline.ipynb`` but
adapted to handle weekly rebalanced portfolios whose weights live under
``data/weights/weight_week*``. It expects a wide price table (dates x tickers)
that contains at least the tickers referenced by the weight files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import pandas as pd

# --- Defaults & constants ---
STARTING_CAPITAL = 10_000.0  # USD
WEIGHT_DIR_GLOB = "weight_week*"
WEIGHTS_ROOT = Path("data/weights")
OUTPUT_CSV = Path("data/portfolio_nav.csv")
# Acceptable numeric drift due to JSON rounding
WEIGHT_TOLERANCE = 1e-6
# Candidates for price CSV discovery (first existing one is used if --prices not passed)
PRICE_CSV_CANDIDATES = (
    Path("data/prices.csv"),
    Path("data/price.csv"),
    Path("notebooks/price.csv"),
    Path("price.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly portfolio NAV tracker.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_ROOT,
        help="Root directory containing weight_week*/team.json files.",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="CSV with price history (dates as first column, tickers as remaining columns). "
        "If omitted, a best-effort search over common locations is performed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Destination CSV for the weekly NAV table.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=STARTING_CAPITAL,
        help="Initial capital per team in USD.",
    )
    return parser.parse_args()


def discover_price_csv(explicit: Path | None) -> Path:
    """Find a price CSV either from the explicit path or common defaults."""
    if explicit is not None:
        csv_path = explicit.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Price CSV not found: {csv_path}")
        return csv_path

    for candidate in PRICE_CSV_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate a price CSV. Pass one explicitly with --prices."
    )


def extract_week_number(name: str) -> Tuple[int, str]:
    """
    Extract the numeric suffix from weight_week* style directory names.
    Returns (numeric_week, original_name) so sorting stays stable.
    """
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    raise ValueError(f"Failed to parse week number from directory name: {name}")


def load_weights(weights_root: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Load and validate weight files.

    Returns:
        Ordered dict keyed by week number -> team name -> {ticker: weight}.
    """
    if not weights_root.exists():
        raise FileNotFoundError(f"Weights root does not exist: {weights_root}")

    week_dirs = sorted(
        (p for p in weights_root.iterdir() if p.is_dir() and p.name.startswith("weight_week")),
        key=lambda p: extract_week_number(p.name)[0],
    )
    if not week_dirs:
        raise ValueError(f"No week directories found under {weights_root}")

    weeks: MutableMapping[int, Dict[str, Dict[str, float]]] = OrderedDict()
    for week_dir in week_dirs:
        week_num, _ = extract_week_number(week_dir.name)
        team_files = sorted(
            (p for p in week_dir.glob("*.json") if p.is_file()),
            key=lambda p: p.stem,
        )
        if not team_files:
            raise ValueError(f"No JSON weight files found in {week_dir}")

        team_weights: Dict[str, Dict[str, float]] = {}
        for weight_path in team_files:
            raw = json.loads(weight_path.read_text())
            weights = {ticker: float(weight) for ticker, weight in raw.items()}
            _validate_weights(weights, weight_path)
            team_weights[weight_path.stem] = weights
        weeks[week_num] = team_weights

    return weeks


def _validate_weights(weights: Mapping[str, float], location: Path) -> None:
    if not weights:
        raise ValueError(f"Empty weights in {location}")

    total_weight = sum(weights.values())
    if not math.isfinite(total_weight):
        raise ValueError(f"Invalid weight total (NaN/inf) in {location}")

    if abs(total_weight - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"Weights in {location} sum to {total_weight:.8f}, expected 1.0 ± {WEIGHT_TOLERANCE}"
        )

    negatives = [ticker for ticker, weight in weights.items() if weight < -WEIGHT_TOLERANCE]
    if negatives:
        raise ValueError(f"Negative weights detected in {location}: {negatives}")


def collect_universe(weeks: Mapping[int, Mapping[str, Mapping[str, float]]]) -> Iterable[str]:
    tickers = set()
    for week_allocs in weeks.values():
        for weights in week_allocs.values():
            tickers.update(weights.keys())
    return sorted(tickers)


def load_price_history(price_csv: Path, tickers: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(price_csv, index_col=0, parse_dates=[0])
    if df.empty:
        raise ValueError(f"Price CSV {price_csv} is empty.")

    df.index.name = "Date"
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    needed = set(tickers)
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing tickers in price data ({price_csv}): {', '.join(sorted(missing))}"
        )
    subset = df.loc[:, sorted(needed)].copy()
    subset = subset.ffill().dropna(how="any")
    if subset.empty:
        raise ValueError("Price data became empty after forward filling missing values.")
    return subset


def compute_weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    weekly_prices = prices.resample("W-FRI").last()
    weekly_prices = weekly_prices.ffill().dropna(how="any")
    weekly_returns = weekly_prices.pct_change().dropna(how="all")
    if weekly_returns.empty:
        raise ValueError("Weekly returns are empty; check the price history coverage.")
    return weekly_returns


def compute_nav_table(
    weeks: Mapping[int, Mapping[str, Mapping[str, float]]],
    weekly_returns: pd.DataFrame,
    starting_capital: float,
) -> pd.DataFrame:
    week_numbers = list(weeks.keys())
    num_weeks = len(week_numbers)
    if num_weeks == 0:
        raise ValueError("No week weights provided.")

    if len(weekly_returns) < num_weeks:
        raise ValueError(
            f"Weekly returns ({len(weekly_returns)}) shorter than week weights ({num_weeks})."
        )

    aligned_returns = weekly_returns.iloc[-num_weeks:].copy()
    aligned_returns.index = week_numbers

    nav_state: Dict[str, float] = {}
    records = []
    for week_num, returns_row in aligned_returns.iterrows():
        team_allocations = weeks[week_num]
        for team, weights in team_allocations.items():
            nav_before = nav_state.get(team, starting_capital)
            portfolio_return = 0.0
            for ticker, weight in weights.items():
                ret = returns_row.get(ticker)
                if pd.isna(ret):
                    raise ValueError(
                        f"Missing return for ticker {ticker} in week {week_num} for team {team}."
                    )
                portfolio_return += weight * ret
            nav_after = nav_before * (1.0 + portfolio_return)
            nav_state[team] = nav_after
            records.append({"week": week_num, "team": str(team), "NAV": float(nav_after)})
    return pd.DataFrame(records)


def save_nav_table(nav_table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    nav_table.sort_values(["week", "team"]).to_csv(destination, index=False)


def main() -> None:
    args = parse_args()
    weeks = load_weights(args.weights)
    tickers = collect_universe(weeks)

    price_csv = discover_price_csv(args.prices)
    prices = load_price_history(price_csv, tickers)
    weekly_returns = compute_weekly_returns(prices)

    nav_table = compute_nav_table(weeks, weekly_returns, args.capital)
    save_nav_table(nav_table, args.output)

    print(f"Wrote NAV table to {args.output} ({len(nav_table)} rows)")


if __name__ == "__main__":
    main()
