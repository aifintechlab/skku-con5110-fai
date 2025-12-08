"""
Generate portfolio NAV paths, weekly snapshots, performance stats, and a simple SVG chart
without relying on heavy third-party libraries.

This script reads:
  * weight submissions from data/weights/weight_week*/<team>.json
  * a daily price table from data/prices.csv

Outputs:
  * data/portfolio_nav.csv             (week, team, NAV at end of week)
  * data/portfolio_performance.csv     (summary metrics per team)
  * data/portfolio_nav.svg             (line chart showing NAV paths)
  * data/portfolio_nav_timeseries.csv  (daily NAV values for every team; optional but handy)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


# --- constants ---
CASH_LABEL = "CASH"
CASH_KEYWORDS = {"cash", "money"}
IGNORE_META = {
    "week_of",
    "week",
    "date",
    "notes",
    "note",
    "comment",
    "comments",
    "team",
    "name",
    "id",
    "portfolio_name",
}
CONTAINER_KEYS = {"portfolio", "weights", "allocations", "allocation", "holdings"}
TRADING_DAYS_PER_YEAR = 252
WEEK_LENGTH_DAYS = 7
RF_RATE = 0.0435
VAR_CONFIDENCE = 0.95


@dataclass
class PriceTable:
    dates: List[date]
    tickers: List[str]
    prices: Dict[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NAV paths and summaries for all teams.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/weights"),
        help="Root directory containing weight_week*/team.json files.",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/prices.csv"),
        help="CSV with daily prices (Date column + tickers).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Starting capital per team (USD).",
    )
    parser.add_argument(
        "--week4-start",
        type=str,
        default="2025-09-29",
        help="Calendar date (YYYY-MM-DD) when week 4 trading starts.",
    )
    parser.add_argument(
        "--nav-csv",
        type=Path,
        default=Path("data/portfolio_nav.csv"),
        help="Weekly NAV snapshot output (week, team, NAV).",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=Path("data/portfolio_performance.csv"),
        help="Performance summary CSV output.",
    )
    parser.add_argument(
        "--timeseries",
        type=Path,
        default=Path("data/portfolio_nav_timeseries.csv"),
        help="Daily NAV output (optional but useful).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/portfolio_nav.svg"),
        help="SVG chart destination.",
    )
    return parser.parse_args()


def derive_week_schedule(week_numbers: Iterable[int], week4_start: str) -> Dict[int, Tuple[date, date]]:
    base_start = datetime.strptime(week4_start, "%Y-%m-%d").date()
    schedule: Dict[int, Tuple[date, date]] = {}
    for week_num in sorted(week_numbers):
        if week_num < 4:
            raise ValueError(f"Week {week_num} precedes week 4; adjust --week4-start if needed.")
        offset = (week_num - 4) * WEEK_LENGTH_DAYS
        start = base_start + timedelta(days=offset)
        end = start + timedelta(days=WEEK_LENGTH_DAYS - 1)
        schedule[week_num] = (start, end)
    return schedule


def load_weights(weights_root: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    if not weights_root.exists():
        raise FileNotFoundError(f"Weights root does not exist: {weights_root}")

    weeks: MutableMapping[int, Dict[str, Dict[str, float]]] = OrderedDict()
    for week_dir in sorted(weights_root.glob("weight_week*")):
        if not week_dir.is_dir():
            continue
        week_num = extract_week_number(week_dir.name)
        team_weights: Dict[str, Dict[str, float]] = {}
        for path in sorted(week_dir.glob("*.json")):
            weights = dict(load_weight_mapping(path))
            validate_weights(weights, path)
            team_weights[path.stem] = normalize_weights(weights)
        if team_weights:
            weeks[week_num] = team_weights
    if not weeks:
        raise ValueError(f"No week directories with JSON files found under {weights_root}")
    return dict(weeks)


def extract_week_number(name: str) -> int:
    digits = ""
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if digits:
        return int(digits)
    raise ValueError(f"Cannot parse week number from directory name: {name}")


def load_weight_mapping(path: Path) -> Dict[str, float]:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty weight file: {path}")

    candidates = [text]
    if "=" in text:
        candidates.append(text.split("=", 1)[1].strip())

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    for candidate in candidates:
        cleaned = candidate.strip().rstrip(";.")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        weights = extract_weight_mapping(data)
        if weights:
            return weights
    raise ValueError(f"Unable to parse weight file as JSON mapping: {path}")


def extract_weight_mapping(obj) -> Dict[str, float]:
    def normalise_ticker(raw: str) -> str:
        ticker = str(raw).strip()
        if ticker.lower() in CASH_KEYWORDS:
            return CASH_LABEL
        return ticker

    def from_mapping(data: Mapping) -> Dict[str, float]:
        acc: Dict[str, float] = {}
        for key, value in data.items():
            key_str = str(key).strip()
            if key_str.lower() in IGNORE_META:
                continue
            if key_str.lower() in CONTAINER_KEYS and isinstance(value, (Mapping, list)):
                acc.update(extract_weight_mapping(value))
                continue
            if isinstance(value, Mapping):
                weight_val = find_weight_field(value)
                if weight_val is not None:
                    acc[normalise_ticker(key_str)] = weight_val
                else:
                    acc.update(extract_weight_mapping(value))
                continue
            if isinstance(value, (int, float)):
                acc[normalise_ticker(key_str)] = float(value)
                continue
            if isinstance(value, str):
                try:
                    acc[normalise_ticker(key_str)] = float(value)
                except ValueError:
                    pass
        return acc

    if isinstance(obj, Mapping):
        return from_mapping(obj)
    if isinstance(obj, list):
        merged: Dict[str, float] = {}
        for item in obj:
            merged.update(extract_weight_mapping(item))
        return merged
    return {}


def find_weight_field(candidate: Mapping) -> float | None:
    for key, value in candidate.items():
        if str(key).strip().lower() == "weight":
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def validate_weights(weights: Mapping[str, float], location: Path) -> None:
    if not weights:
        raise ValueError(f"No weights found in {location}")
    total = sum(weights.values())
    if not math.isfinite(total):
        raise ValueError(f"Invalid weight total (NaN/inf) in {location}")
    neg_tolerance = 1e-6
    negatives = [ticker for ticker, weight in weights.items() if weight < -neg_tolerance]
    if negatives:
        raise ValueError(f"Negative weights in {location}: {negatives}")


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value before normalization.")
    return {ticker: weight / total for ticker, weight in weights.items()}


def load_price_table(price_csv: Path, required_tickers: Iterable[str]) -> PriceTable:
    if not price_csv.exists():
        raise FileNotFoundError(f"Price CSV not found: {price_csv}")

    with price_csv.open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Price CSV {price_csv} is empty.")

        if not header or header[0].lower() != "date":
            raise ValueError("First column must be 'Date'.")

        tickers = header[1:]
        missing = sorted(set(required_tickers).difference(tickers))
        if missing:
            raise ValueError(
                f"Missing tickers in price data ({price_csv}): {', '.join(missing)}"
            )

        date_list: List[date] = []
        prices: Dict[str, List[float]] = {ticker: [] for ticker in tickers}

        for row in reader:
            if not row:
                continue
            dt = datetime.strptime(row[0], "%Y-%m-%d").date()
            date_list.append(dt)
            for ticker, cell in zip(tickers, row[1:]):
                value = cell.strip()
                if value == "":
                    raise ValueError(
                        f"Missing price for ticker {ticker} on {dt} in {price_csv}."
                    )
                prices[ticker].append(float(value))

    return PriceTable(dates=date_list, tickers=tickers, prices=prices)


def build_week_price_indices(
    price_table: PriceTable, schedule: Mapping[int, Tuple[date, date]]
) -> Dict[int, List[int]]:
    date_to_index = {dt: idx for idx, dt in enumerate(price_table.dates)}
    week_indices: Dict[int, List[int]] = {}
    for week_num, (start, end) in sorted(schedule.items()):
        idxs = [
            date_to_index[dt]
            for dt in price_table.dates
            if start <= dt <= end and dt in date_to_index
        ]
        if not idxs:
            raise ValueError(
                f"No price data covering {start} to {end} for week {week_num}."
            )
        week_indices[week_num] = idxs
    return week_indices


def compute_nav_paths(
    price_table: PriceTable,
    weeks: Mapping[int, Mapping[str, Mapping[str, float]]],
    schedule: Mapping[int, Tuple[date, date]],
    starting_capital: float,
    week_indices: Mapping[int, List[int]],
) -> Dict[str, List[Tuple[date, float]]]:
    teams = sorted({team for week_allocs in weeks.values() for team in week_allocs})
    nav_series: Dict[str, List[Tuple[date, float]]] = {team: [] for team in teams}
    nav_state: Dict[str, float] = {team: starting_capital for team in teams}

    for week_num in sorted(schedule):
        if week_num not in weeks:
            raise ValueError(f"No weights provided for week {week_num}.")
        if week_num not in week_indices:
            raise ValueError(f"No price indices recorded for week {week_num}.")
        indices = week_indices[week_num]

        week_allocs = weeks[week_num]

        for team in teams:
            weights = dict(week_allocs.get(team, {}))
            weights.pop(CASH_LABEL, None)
            for pos, idx in enumerate(indices):
                current_date = price_table.dates[idx]
                if pos == 0:
                    nav_series[team].append((current_date, nav_state[team]))
                    continue
                if weights:
                    portfolio_return = 0.0
                    prev_idx = idx - 1
                    for ticker, w in weights.items():
                        price_today = price_table.prices[ticker][idx]
                        price_prev = price_table.prices[ticker][prev_idx]
                        if price_prev == 0:
                            raise ValueError(
                                f"Zero price for {ticker} on {price_table.dates[prev_idx]}"
                            )
                        portfolio_return += w * (price_today / price_prev - 1.0)
                    nav_state[team] *= (1.0 + portfolio_return)
                nav_series[team].append((current_date, nav_state[team]))

    return nav_series


def _quantile(values: List[float], quantile: float) -> float:
    if not values:
        return float("nan")
    if quantile <= 0:
        return min(values)
    if quantile >= 1:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * quantile
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_vals[int(idx)]
    weight = idx - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _skewness(returns: List[float]) -> float:
    n = len(returns)
    if n < 3:
        return float("nan")
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 == 0:
        return 0.0
    m3 = sum((r - mean) ** 3 for r in returns) / n
    return math.sqrt(n * (n - 1)) / (n - 2) * (m3 / (m2 ** 1.5))


def compute_week_growth(
    price_table: PriceTable, indices: Sequence[int], tickers: Iterable[str]
) -> Dict[str, float]:
    start_idx = indices[0]
    end_idx = indices[-1]
    if start_idx == end_idx:
        return {ticker: 1.0 for ticker in tickers}
    growth: Dict[str, float] = {}
    for ticker in tickers:
        series = price_table.prices[ticker]
        start_price = series[start_idx]
        end_price = series[end_idx]
        if start_price == 0:
            raise ValueError(f"Zero price for {ticker} on {price_table.dates[start_idx]}")
        growth[ticker] = end_price / start_price
    return growth


def compute_turnover(
    weeks: Mapping[int, Mapping[str, Mapping[str, float]]],
    price_table: PriceTable,
    week_indices: Mapping[int, List[int]],
    tickers: Iterable[str],
) -> Dict[str, float]:
    teams = sorted({team for week_allocs in weeks.values() for team in week_allocs})
    totals: Dict[str, float] = {team: 0.0 for team in teams}
    counts: Dict[str, int] = {team: 0 for team in teams}
    ticker_set = sorted({ticker for ticker in tickers if ticker != CASH_LABEL})
    prev_allocs: Dict[str, Dict[str, float]] | None = None
    prev_week: int | None = None
    growth_cache: Dict[int, Dict[str, float]] = {}

    for week in sorted(week_indices):
        if week not in weeks:
            continue
        week_allocs = weeks[week]
        cleaned: Dict[str, Dict[str, float]] = {}
        for team in teams:
            weights = dict(week_allocs.get(team, {}))
            weights.pop(CASH_LABEL, None)
            cleaned[team] = weights

        if prev_allocs is not None and prev_week is not None:
            if prev_week not in week_indices:
                raise ValueError(f"Missing price indices for week {prev_week} during turnover computation.")
            if prev_week not in growth_cache:
                growth_cache[prev_week] = compute_week_growth(
                    price_table, week_indices[prev_week], ticker_set
                )
            growth = growth_cache[prev_week]
            for team in teams:
                prev_weights = prev_allocs.get(team, {})
                curr_weights = cleaned.get(team, {})
                drifted: Dict[str, float] = {}
                if prev_weights:
                    grown_values = {
                        ticker: prev_weights.get(ticker, 0.0) * growth.get(ticker, 1.0)
                        for ticker in prev_weights
                    }
                    total = sum(grown_values.values())
                    if total > 0:
                        drifted = {ticker: value / total for ticker, value in grown_values.items()}
                tickers_union = set(drifted) | set(curr_weights)
                if not tickers_union:
                    continue
                turnover = 0.5 * sum(
                    abs(curr_weights.get(t, 0.0) - drifted.get(t, 0.0)) for t in tickers_union
                )
                totals[team] += turnover
                counts[team] += 1

        prev_allocs = cleaned
        prev_week = week

    return {team: (totals[team] / counts[team] if counts[team] else 0.0) for team in teams}


def compute_performance(
    nav_series: Mapping[str, List[Tuple[date, float]]],
    turnover_map: Mapping[str, float] | None = None,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for team, series in nav_series.items():
        if len(series) < 2:
            continue
        nav_values = [nav for _, nav in series]
        returns = []
        for prev, curr in zip(nav_values[:-1], nav_values[1:]):
            if prev == 0:
                continue
            returns.append(curr / prev - 1.0)

        if not returns:
            continue

        mean_return = sum(returns) / len(returns)
        variance = (
            sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            if len(returns) > 1
            else 0.0
        )
        ann_vol = math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)
        ann_return = mean_return * TRADING_DAYS_PER_YEAR

        total_return = nav_values[-1] / nav_values[0] - 1.0
        days = len(nav_values)
        sharpe = ((ann_return - RF_RATE) / ann_vol) if ann_vol > 0 else float("nan")
        skew = _skewness(returns)
        tail_q = 1.0 - VAR_CONFIDENCE
        var_threshold = _quantile(returns, tail_q)
        tail_returns = [r for r in returns if r <= var_threshold]
        es = sum(tail_returns) / len(tail_returns) if tail_returns else var_threshold
        turnover = turnover_map.get(team) if turnover_map else float("nan")

        peak = nav_values[0]
        mdd = 0.0
        for value in nav_values:
            if value > peak:
                peak = value
            drawdown = value / peak - 1.0
            if drawdown < mdd:
                mdd = drawdown

        records.append(
            {
                "team": team,
                "Days": days,
                "TotalReturn": total_return,
                "AnnReturn": ann_return,
                "AnnVol": ann_vol,
                "Sharpe": sharpe,
                "MDD": mdd,
                "FinalNAV": nav_values[-1],
                "Skew": skew,
                "VaR": var_threshold,
                "ES": es,
                "Turnover": turnover,
            }
        )
    records.sort(key=lambda row: row["FinalNAV"], reverse=True)
    return records


def summarize_weekly(nav_series: Mapping[str, List[Tuple[date, float]]], schedule: Mapping[int, Tuple[date, date]]) -> List[Dict[str, object]]:
    week_records: List[Dict[str, object]] = []
    lookup: Dict[str, Dict[date, float]] = {
        team: {dt: nav for dt, nav in series} for team, series in nav_series.items()
    }
    for week_num, (start_date, end_date) in sorted(schedule.items()):
        for team, nav_map in lookup.items():
            candidates = [dt for dt in nav_map if start_date <= dt <= end_date]
            if not candidates:
                raise ValueError(
                    f"No NAV observations for team {team} during week {week_num} "
                    f"({start_date} ~ {end_date})."
                )
            anchor_date = max(candidates)
            week_records.append(
                {"week": week_num, "team": team, "NAV": float(nav_map[anchor_date])}
            )
    return week_records


def clamp_schedule_to_prices(
    schedule: Mapping[int, Tuple[date, date]], last_price_date: date
) -> Dict[int, Tuple[date, date]]:
    clamped: Dict[int, Tuple[date, date]] = {}
    for week, (start, end) in schedule.items():
        if start > last_price_date:
            continue
        clamped[week] = (start, min(end, last_price_date))
    return clamped


def write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_timeseries_csv(path: Path, nav_series: Mapping[str, List[Tuple[date, float]]]) -> None:
    all_dates = sorted({dt for series in nav_series.values() for dt, _ in series})
    teams = sorted(nav_series.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date"] + teams)
        for dt in all_dates:
            row = [dt.isoformat()]
            for team in teams:
                value = next((nav for d, nav in nav_series[team] if d == dt), "")
                row.append(f"{value:.6f}" if value != "" else "")
            writer.writerow(row)


def render_svg(nav_series: Mapping[str, List[Tuple[date, float]]], output: Path) -> None:
    all_dates = sorted({dt for series in nav_series.values() for dt, _ in series})
    if len(all_dates) < 2:
        print("Not enough dates to render SVG chart.")
        return
    teams = sorted(nav_series.keys())

    nav_values = [nav for series in nav_series.values() for _, nav in series]
    min_nav_actual = min(nav_values)
    max_nav = max(nav_values)
    if math.isclose(max_nav, min_nav_actual):
        max_nav = min_nav_actual + 1.0

    desired_min = 9000.0
    min_nav = min(desired_min, min_nav_actual)

    width, height = 960, 540
    margin = 60
    inner_w = width - 2 * margin
    inner_h = height - 2 * margin

    def scale_x(idx: int) -> float:
        return margin + inner_w * idx / (len(all_dates) - 1)

    def scale_y(nav: float) -> float:
        return margin + inner_h * (1 - (nav - min_nav) / (max_nav - min_nav))

    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 12px; }</style>',
        '<rect x="0" y="0" width="{0}" height="{1}" fill="#ffffff" />'.format(width, height),
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        f'<text x="{margin}" y="{margin - 20}" font-size="16" font-weight="600">Team Portfolio NAV Paths</text>',
        f'<text x="{width/2}" y="{height - margin + 40}" text-anchor="middle">Date</text>',
        f'<text x="{margin - 40}" y="{margin - 10}" text-anchor="end">NAV (USD)</text>',
    ]

    # Axis labels (min/max and start/end dates)
    svg_lines.append(
        f'<text x="{margin - 10}" y="{scale_y(min_nav)}" text-anchor="end">{min_nav:,.0f}</text>'
    )
    svg_lines.append(
        f'<text x="{margin - 10}" y="{scale_y(max_nav)}" text-anchor="end">{max_nav:,.0f}</text>'
    )
    svg_lines.append(
        f'<text x="{margin}" y="{height - margin + 20}" text-anchor="middle">{all_dates[0].isoformat()}</text>'
    )
    svg_lines.append(
        f'<text x="{width - margin}" y="{height - margin + 20}" text-anchor="middle">{all_dates[-1].isoformat()}</text>'
    )

    legend_x = margin + 10
    legend_y = margin + 10
    legend_line_height = 18

    for idx, team in enumerate(teams):
        color = color_palette[idx % len(color_palette)]
        series = nav_series[team]
        nav_map = {dt: nav for dt, nav in series}
        points = []
        for i, dt in enumerate(all_dates):
            if dt not in nav_map:
                continue
            x = scale_x(i)
            y = scale_y(nav_map[dt])
            points.append(f"{x:.2f},{y:.2f}")
        if len(points) < 2:
            continue
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{" ".join(points)}" />'
        )
        svg_lines.append(
            f'<rect x="{legend_x}" y="{legend_y + idx * legend_line_height - 10}" width="14" height="14" fill="{color}"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 20}" y="{legend_y + idx * legend_line_height}" alignment-baseline="middle">{team}</text>'
        )

    svg_lines.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg_lines))
    print(f"Wrote NAV SVG chart to {output}")


def main() -> None:
    args = parse_args()
    weeks = load_weights(args.weights)
    schedule = derive_week_schedule(weeks.keys(), args.week4_start)
    tickers = {ticker for week in weeks.values() for weights in week.values() for ticker in weights if ticker != CASH_LABEL}
    price_table = load_price_table(args.prices, tickers)
    last_price_date = price_table.dates[-1]
    schedule = clamp_schedule_to_prices(schedule, last_price_date)
    if not schedule:
        raise ValueError("Price history does not cover the earliest scheduled week.")

    week_indices = build_week_price_indices(price_table, schedule)
    nav_series = compute_nav_paths(price_table, weeks, schedule, args.capital, week_indices)
    turnover_map = compute_turnover(weeks, price_table, week_indices, tickers)
    performance_rows = compute_performance(nav_series, turnover_map)
    weekly_rows = summarize_weekly(nav_series, schedule)

    weekly_rows_rounded = [
        {"week": row["week"], "team": row["team"], "NAV": round(row["NAV"], 3)}
        for row in weekly_rows
    ]
    write_csv(args.nav_csv, weekly_rows_rounded, ["week", "team", "NAV"])

    perf_headers = [
        "team",
        "Days",
        "TotalReturn",
        "AnnReturn",
        "AnnVol",
        "Sharpe",
        "MDD",
        "FinalNAV",
        "Skew",
        "VaR",
        "ES",
        "Turnover",
    ]
    performance_rows_rounded = []
    for row in performance_rows:
        rounded = row.copy()
        for key in ["TotalReturn", "AnnReturn", "AnnVol", "Sharpe", "MDD", "FinalNAV", "Skew", "VaR", "ES", "Turnover"]:
            if key in rounded and rounded[key] is not None:
                rounded[key] = round(rounded[key], 3)
        performance_rows_rounded.append(rounded)

    write_csv(args.performance, performance_rows_rounded, perf_headers)
    write_timeseries_csv(args.timeseries, nav_series)
    render_svg(nav_series, args.figure)

    print(
        f"Wrote weekly NAV to {args.nav_csv} ({len(weekly_rows)} rows)\n"
        f"Wrote performance table to {args.performance} ({len(performance_rows)} teams)"
    )


if __name__ == "__main__":
    main()
