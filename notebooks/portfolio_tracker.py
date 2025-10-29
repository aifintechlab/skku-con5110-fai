"""
Portfolio tracker for weekly team submissions.

Loads JSON weight files from ``data/weights/weight_week*``, validates that
allocations sum to 1, walks daily price data to build each team's NAV path,
and then
  * saves the end-of-week NAV snapshot to ``data/portfolio_nav.csv``,
  * saves a performance summary table to ``data/portfolio_performance.csv``,
  * writes a line chart of NAV paths to ``data/portfolio_nav.png``.

The weekly schedule follows the specification shared alongside the notebook:
weights submitted on Sunday are deployed starting the following Monday and
held through that week's Sunday. Use ``--week4-start`` to override the initial
week if future cohorts run on a different calendar.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# --- Defaults & constants ---
STARTING_CAPITAL = 10_000.0  # USD
WEIGHTS_ROOT = Path("data/weights")
WEEKLY_NAV_CSV = Path("data/portfolio_nav.csv")
PERF_TABLE_CSV = Path("data/portfolio_performance.csv")
NAV_FIG_PATH = Path("data/portfolio_nav.png")
WEIGHT_TOLERANCE = 1e-6
TRADING_DAYS_PER_YEAR = 252
WEEK_LENGTH_DAYS = 7

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
        help="CSV with daily price history (dates as first column, tickers as remaining columns). "
        "If omitted, a best-effort search over common locations is performed.",
    )
    parser.add_argument(
        "--nav-csv",
        type=Path,
        default=WEEKLY_NAV_CSV,
        help="Destination CSV for the weekly NAV snapshot (week/team/NAV).",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=PERF_TABLE_CSV,
        help="Destination CSV for the performance summary table.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=NAV_FIG_PATH,
        help="Path to save the NAV line chart.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=STARTING_CAPITAL,
        help="Initial capital per team in USD.",
    )
    parser.add_argument(
        "--week4-start",
        type=str,
        default="2025-09-29",
        help="Calendar date (YYYY-MM-DD) when week 4 trading starts.",
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


def derive_week_schedule(week_numbers: Sequence[int], week4_start: str) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build a mapping of week -> (start_date, end_date) using the supplied week 4 start.
    Later weeks advance in seven day increments.
    """
    base_start = pd.Timestamp(week4_start).normalize()
    schedule: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for week_num in sorted(week_numbers):
        if week_num < 4:
            raise ValueError(
                f"Week {week_num} precedes the configured base week (4). Update --week4-start logic."
            )
        offset_days = (week_num - 4) * WEEK_LENGTH_DAYS
        start_date = base_start + pd.Timedelta(days=offset_days)
        end_date = start_date + pd.Timedelta(days=WEEK_LENGTH_DAYS - 1)
        schedule[week_num] = (start_date, end_date)
    return schedule


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


def compute_daily_nav(
    prices: pd.DataFrame,
    weeks: Mapping[int, Mapping[str, Mapping[str, float]]],
    schedule: Mapping[int, Tuple[pd.Timestamp, pd.Timestamp]],
    starting_capital: float,
) -> pd.DataFrame:
    teams = sorted({team for week_allocs in weeks.values() for team in week_allocs})
    nav_df = pd.DataFrame(index=prices.index, columns=teams, dtype=float)
    nav_state: Dict[str, float] = {team: starting_capital for team in teams}

    returns = prices.pct_change().fillna(0.0)

    for week_num in sorted(schedule):
        if week_num not in weeks:
            raise ValueError(f"No weights found for week {week_num}.")

        start, end = schedule[week_num]
        start_idx = prices.index.searchsorted(start)
        end_idx = prices.index.searchsorted(end, side="right")
        week_dates = prices.index[start_idx:end_idx]
        if len(week_dates) == 0:
            raise ValueError(
                f"No price data between {start.date()} and {end.date()} for week {week_num}."
            )

        team_weights = weeks[week_num]
        weight_vectors = {
            team: pd.Series(weights, dtype=float).reindex(prices.columns, fill_value=0.0)
            for team, weights in team_weights.items()
        }

        for pos, current_date in enumerate(week_dates):
            ret_row = returns.loc[current_date]
            for team in teams:
                weights = weight_vectors.get(team)
                if pos == 0:
                    # Record the NAV at the rebalance date before applying returns.
                    nav_df.loc[current_date, team] = nav_state[team]
                    continue
                if weights is None:
                    # If a team skipped submitting weights this week, carry capital forward unchanged.
                    nav_df.loc[current_date, team] = nav_state[team]
                    continue
                portfolio_return = float(
                    ret_row.reindex(weights.index, fill_value=0.0).mul(weights).sum()
                )
                nav_state[team] = nav_state[team] * (1.0 + portfolio_return)
                nav_df.loc[current_date, team] = nav_state[team]

    first_start = min(start for start, _ in schedule.values())
    nav_df = nav_df.loc[nav_df.index >= first_start].sort_index()
    nav_df = nav_df.ffill()
    return nav_df


def compute_performance_table(nav_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in nav_df.columns:
        nav = nav_df[team].dropna()
        if nav.shape[0] < 2:
            continue
        rets = nav.pct_change().dropna()
        total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
        if len(rets) == 0:
            ann_return = math.nan
            ann_vol = math.nan
        else:
            ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / len(rets)) - 1.0
            ann_vol = rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) if len(rets) > 1 else math.nan
        if ann_vol is None or math.isnan(ann_vol) or ann_vol == 0:
            sharpe = math.nan
        else:
            sharpe = ann_return / ann_vol
        cumulative_max = nav.cummax()
        drawdown = nav / cumulative_max - 1.0
        mdd = drawdown.min()
        rows.append(
            {
                "team": team,
                "Days": int(nav.shape[0]),
                "TotalReturn": total_return,
                "AnnReturn": ann_return,
                "AnnVol": ann_vol,
                "Sharpe": sharpe,
                "MDD": mdd,
                "FinalNAV": nav.iloc[-1],
            }
        )
    perf = pd.DataFrame(rows).set_index("team").sort_values("FinalNAV", ascending=False)
    return perf


def summarize_weekly_nav(nav_df: pd.DataFrame, schedule: Mapping[int, Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    records = []
    for week_num, (_, end_date) in sorted(schedule.items()):
        mask = (nav_df.index >= schedule[week_num][0]) & (nav_df.index <= end_date)
        nav_slice = nav_df.loc[mask]
        if nav_slice.empty:
            raise ValueError(f"No NAV observations found for week {week_num}.")
        last_nav = nav_slice.iloc[-1]
        for team, nav in last_nav.items():
            records.append({"week": week_num, "team": team, "NAV": float(nav)})
    return pd.DataFrame(records)


def plot_nav_paths(nav_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for team in nav_df.columns:
        series = nav_df[team].dropna()
        if series.empty:
            continue
        plt.plot(series.index, series.values, linewidth=1.8, label=team)
    plt.title("Team Portfolio NAV Paths (Weekly Rebalanced)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_nav_table(nav_table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    nav_table.sort_values(["week", "team"]).to_csv(destination, index=False)


def main() -> None:
    args = parse_args()
    weeks = load_weights(args.weights)
    tickers = collect_universe(weeks)

    schedule = derive_week_schedule(weeks.keys(), args.week4_start)
    price_csv = discover_price_csv(args.prices)
    prices = load_price_history(price_csv, tickers)

    nav_df = compute_daily_nav(prices, weeks, schedule, args.capital)
    plot_nav_paths(nav_df, args.figure)

    perf_table = compute_performance_table(nav_df)
    perf_table.to_csv(args.performance, float_format="%.6f")

    weekly_nav = summarize_weekly_nav(nav_df, schedule)
    save_nav_table(weekly_nav, args.nav_csv)

    print(f"Wrote weekly NAV snapshot to {args.nav_csv} ({len(weekly_nav)} rows).")
    print(f"Wrote performance table to {args.performance} ({len(perf_table)} teams).")
    print(f"Wrote NAV chart to {args.figure}.")


if __name__ == "__main__":
    main()
