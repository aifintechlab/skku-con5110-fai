"""
Download daily close prices for every ticker referenced in the weekly weight files.

This replaces the earlier synthetic generator. It walks through
``data/weights/weight_week*/*.json`` to gather the unique tickers (excluding any
cash placeholders), queries Yahoo Finance via ``yfinance``, and produces a
wide CSV with one column per ticker. Missing observations are forward-filled so
downstream backtests start with a clean table.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, MutableSet, Sequence

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "yfinance is required for this script. Install with `pip install yfinance pandas`."
    ) from exc


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

# yfinance behaves better when requests are chunked; >100 tickers often times out.
CHUNK_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance close prices for all submitted tickers.")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path("data/weights"),
        help="Root directory containing weight_week*/team.json files.",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2025, 9, 29),
        help="First trading day to download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2025, 10, 30),
        help="Last trading day to download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prices.csv"),
        help="Destination CSV for the consolidated price table.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of tickers to fetch per yfinance call (smaller reduces timeout risk).",
    )
    parser.add_argument(
        "--no-forward-fill",
        action="store_true",
        help="Disable forward-filling of missing prices.",
    )
    return parser.parse_args()


def gather_tickers(weights_root: Path) -> Sequence[str]:
    if not weights_root.exists():
        raise FileNotFoundError(f"Weights root not found: {weights_root}")

    tickers: MutableSet[str] = set()
    for path in weights_root.glob("weight_week*/*.json"):
        weights = load_weight_mapping(path)
        for ticker in weights:
            if ticker != CASH_LABEL:
                tickers.add(ticker)
    if not tickers:
        raise ValueError(f"No tickers extracted from {weights_root}")
    return sorted(tickers)


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
    raise ValueError(f"Unable to parse weight file as JSON: {path}")


def extract_weight_mapping(obj) -> Dict[str, float]:
    def normalise_ticker(ticker: str) -> str:
        t_clean = str(ticker).strip()
        if t_clean.lower() in CASH_KEYWORDS:
            return CASH_LABEL
        return t_clean

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
                maybe_weight = find_weight_field(value)
                if maybe_weight is not None:
                    acc[normalise_ticker(key_str)] = maybe_weight
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


def chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def download_close_prices(
    tickers: Sequence[str],
    start: date,
    end: date,
    chunk_size: int,
) -> pd.DataFrame:
    if start > end:
        raise ValueError("start_date must be on/before end_date.")

    dfs: List[pd.DataFrame] = []
    yf_end = end + timedelta(days=1)  # yfinance end date is exclusive

    for batch in chunked(tickers, chunk_size):
        close = _download_batch(batch, start, yf_end)
        if not close.empty:
            dfs.append(close)

    if not dfs:
        raise RuntimeError("Failed to download prices for the requested tickers.")

    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[~combined.index.duplicated()].sort_index()
    # Limit to requested window again (defensive)
    combined = combined[(combined.index.date >= start) & (combined.index.date <= end)]
    combined.index = combined.index.tz_localize(None)
    combined.index.name = "Date"
    combined = _recover_missing_tickers(combined, tickers, start, end)
    return combined


def extract_close_from_yf(df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if isinstance(df, pd.Series):
        return df.to_frame(name="Close")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"]
        frames = []
        for ticker in tickers:
            if ticker in df.columns.get_level_values(0):
                sub = df[ticker]
                if "Close" in sub.columns:
                    frames.append(sub["Close"].rename(ticker))
        if frames:
            return pd.concat(frames, axis=1)
    return df


def _download_batch(batch: Sequence[str], start: date, yf_end: date) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=list(batch),
                start=start.isoformat(),
                end=yf_end.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)
            continue
        close = extract_close_from_yf(raw, batch)
        if close.empty:
            time.sleep(RETRY_DELAY * attempt)
            continue
        return close
    return pd.DataFrame()


def _recover_missing_tickers(prices: pd.DataFrame, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
    yf_end = end + timedelta(days=1)
    missing = []
    for ticker in tickers:
        if ticker not in prices.columns:
            missing.append(ticker)
            continue
        series = prices[ticker]
        if series.isna().all():
            missing.append(ticker)
    if not missing:
        return prices

    recovered = []
    for ticker in missing:
        data = _download_batch([ticker], start, yf_end)
        if data.empty or ticker not in data.columns or data[ticker].isna().all():
            print(f"[WARN] Failed to recover ticker {ticker} after retries.")
            continue
        recovered.append(data[[ticker]])

    if recovered:
        prices = pd.concat([prices] + recovered, axis=1)
        prices = prices.loc[:, sorted(prices.columns)]

    still_missing = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns or prices[ticker].isna().all()
    ]
    if still_missing:
        raise RuntimeError(
            f"Unable to download prices for tickers: {', '.join(still_missing)}. "
            "Please re-run later."
        )
    return prices


def convert_krw_to_usd(prices: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    kr_tickers = [col for col in prices.columns if col.endswith(".KS")]
    if not kr_tickers:
        return prices

    fx_end = end + timedelta(days=1)
    fx = yf.download(
        tickers="KRW=X",
        start=start.isoformat(),
        end=fx_end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if fx.empty:
        raise RuntimeError("Failed to download KRW=X FX rate for KRW tickers.")

    if isinstance(fx.columns, pd.MultiIndex):
        fx_close = fx["Close"]
        if isinstance(fx_close, pd.Series):
            fx_series = fx_close
        else:
            if "KRW=X" in fx_close.columns:
                fx_series = fx_close["KRW=X"]
            else:
                fx_series = fx_close.squeeze()
    else:
        fx_series = fx["Close"].squeeze()

    if not isinstance(fx_series, pd.Series):
        fx_series = pd.Series(fx_series)

    fx_series = fx_series.tz_localize(None).rename("KRW=X")
    fx_series = fx_series.reindex(prices.index).ffill()
    if fx_series.isna().any():
        raise RuntimeError("KRW=X FX series has missing values after alignment.")

    prices.loc[:, kr_tickers] = prices.loc[:, kr_tickers].div(fx_series, axis=0)
    return prices


def drop_crypto_only_rows(prices: pd.DataFrame) -> pd.DataFrame:
    non_crypto = [col for col in prices.columns if not col.endswith("-USD")]
    if not non_crypto:
        return prices
    mask = prices[non_crypto].isna().all(axis=1)
    if mask.any():
        prices = prices.loc[~mask]
    return prices


def main() -> None:
    args = parse_args()
    tickers = gather_tickers(args.weights_root)
    prices = download_close_prices(tickers, args.start_date, args.end_date, args.chunk_size)
    prices = drop_crypto_only_rows(prices)
    prices = convert_krw_to_usd(prices, args.start_date, args.end_date)
    if not args.no_forward_fill:
        prices = prices.ffill()
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("Price table is empty after cleaning.")
    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_tickers:
        raise RuntimeError(
            "Price table is missing the following tickers: " + ", ".join(missing_tickers)
        )
    cols_with_na = prices.columns[prices.isna().any()].tolist()
    if cols_with_na:
        raise RuntimeError(
            f"Price table still contains missing values for tickers: {', '.join(cols_with_na)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.output, float_format="%.6f")
    print(
        f"Wrote {args.output} with {prices.shape[0]} rows and {prices.shape[1]} tickers "
        "(Yahoo Finance close prices)."
    )


if __name__ == "__main__":
    main()
