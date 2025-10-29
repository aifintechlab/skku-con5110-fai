"""
Utility to generate price data, portfolio analytics, and a static site
summarising the latest results. Intended for use both locally and in CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SITE_DIR = REPO_ROOT / "site"


def run_price_download(start_date: str, end_date: str, weights_root: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_price_history.py"),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--weights-root",
        str(weights_root),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def run_portfolio_visualizer(week4_start: str, capital: float, prices: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "portfolio_visualizer.py"),
        "--week4-start",
        week4_start,
        "--capital",
        str(capital),
        "--prices",
        str(prices),
    ]
    subprocess.run(cmd, check=True)


def build_index_html(target_dir: Path, svg_path: Path, perf_csv: Path, weekly_csv: Path) -> None:
    perf = pd.read_csv(perf_csv)
    weekly = pd.read_csv(weekly_csv)

    perf_table = perf.to_html(
        index=False,
        classes="table table-striped",
        justify="center",
        border=0,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x,
    )
    weekly_table = weekly.to_html(
        index=False,
        classes="table table-striped",
        justify="center",
        border=0,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x,
    )

    last_updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Weekly Portfolio Tracker</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 0; background: #f6f7fb; color: #1f2933; }}
    header {{ background: #1f2933; color: #fff; padding: 1.8rem 1rem; text-align: center; }}
    main {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem 3rem; }}
    section {{ background: #fff; border-radius: 12px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08); padding: 1.8rem; margin-bottom: 2rem; }}
    h1, h2 {{ margin: 0 0 1rem; }}
    .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 1rem; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border-radius: 8px; }}
    .table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    .table th, .table td {{ padding: 0.55rem 0.7rem; border-bottom: 1px solid #e2e8f0; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
    .actions a {{ display: inline-block; margin-right: 1rem; text-decoration: none; color: #2563eb; }}
    footer {{ text-align: center; color: #94a3b8; font-size: 0.85rem; padding-bottom: 3rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Weekly Portfolio Tracker</h1>
    <p class="meta">Latest update: {last_updated}</p>
  </header>
  <main>
    <section>
      <h2>Portfolio NAV Paths</h2>
      <p class="meta">Weekly rebalanced NAV lines for every team.</p>
      <img src="{svg_path.name}" alt="Portfolio NAV Chart">
      <div class="actions">
        <a href="portfolio_nav_timeseries.csv">Download daily NAVs (CSV)</a>
        <a href="portfolio_nav.csv">Download weekly snapshot (CSV)</a>
      </div>
    </section>
    <section>
      <h2>Performance Summary</h2>
      <p class="meta">Total return, annualised return/volatility, Sharpe (RF = 4.35%), and drawdown per team.</p>
      {perf_table}
      <div class="actions">
        <a href="{perf_csv.name}">Download performance table (CSV)</a>
      </div>
    </section>
    <section>
      <h2>Weekly NAV Snapshot</h2>
      <p class="meta">Ending NAV per week for each team.</p>
      {weekly_table}
    </section>
  </main>
  <footer>
    Built from team weight submissions – updated automatically via GitHub Actions.
  </footer>
</body>
</html>
"""
    (target_dir / "index.html").write_text(html, encoding="utf-8")


def build_site(
    start_date: str,
    end_date: str,
    week4_start: str,
    capital: float,
    weights_root: Path,
    prices_csv: Path,
    skip_download: bool,
    existing_prices: Path | None,
) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if skip_download:
        if existing_prices is None:
            raise ValueError("--skip-download requires --existing-prices")
        existing_path = existing_prices.resolve()
        target_path = prices_csv.resolve()
        if existing_path != target_path:
            shutil.copy(existing_path, target_path)
    else:
        run_price_download(start_date, end_date, weights_root, prices_csv)

    run_portfolio_visualizer(week4_start, capital, prices_csv)

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    # Copy artefacts
    artifacts = [
        ("portfolio_nav.svg", SITE_DIR / "portfolio_nav.svg"),
        ("portfolio_performance.csv", SITE_DIR / "portfolio_performance.csv"),
        ("portfolio_nav.csv", SITE_DIR / "portfolio_nav.csv"),
        ("portfolio_nav_timeseries.csv", SITE_DIR / "portfolio_nav_timeseries.csv"),
    ]
    for name, target in artifacts:
        shutil.copy(DATA_DIR / name, target)

    build_index_html(
        SITE_DIR,
        SITE_DIR / "portfolio_nav.svg",
        SITE_DIR / "portfolio_performance.csv",
        SITE_DIR / "portfolio_nav.csv",
    )


def parse_args() -> argparse.Namespace:
    today = dt.date.today()
    default_start = dt.date(2025, 9, 29).isoformat()
    default_end = dt.date(2025, 11, 2).isoformat()
    parser = argparse.ArgumentParser(description="Generate static site with latest NAV analytics.")
    parser.add_argument("--start-date", default=default_start, help="Start date for price download (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=default_end, help="End date for price download (YYYY-MM-DD).")
    parser.add_argument(
        "--week4-start",
        default="2025-09-30",
        help="Calendar date corresponding to week 4 trading start (YYYY-MM-DD).",
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital per team.")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=REPO_ROOT / "data" / "weights",
        help="Directory containing weight_week*/ JSON files.",
    )
    parser.add_argument(
        "--prices-csv",
        type=Path,
        default=DATA_DIR / "prices.csv",
        help="Destination for the downloaded price table.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip price download (useful when prices.csv already exists).",
    )
    parser.add_argument(
        "--existing-prices",
        type=Path,
        default=None,
        help="Path to an existing price CSV to reuse when --skip-download is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_site(
        start_date=args.start_date,
        end_date=args.end_date,
        week4_start=args.week4_start,
        capital=args.capital,
        weights_root=args.weights_root,
        prices_csv=args.prices_csv,
        skip_download=args.skip_download,
        existing_prices=args.existing_prices,
    )


if __name__ == "__main__":
    main()
