# Repository Overview for LLM Assistants

This project builds a static portfolio dashboard from weekly weight submissions. Key assets:
- `data/weights/weight_week*/` — team weight JSON files (weeks 4–11, etc.). Cash is marked as `money`/`cash`.
- `data/prices.csv` — daily adjusted close prices (yfinance `auto_adjust=True`), KRW tickers converted to USD via `KRW=X`, crypto-only rows (weekends) removed.
- Generated outputs: `data/portfolio_nav*.csv`, `data/portfolio_performance.csv`, `data/portfolio_nav.svg`, and the static site under `site/`.

## Main Scripts
- `scripts/build_price_history.py`
  - Reads all tickers from `data/weights/weight_week*/`.
  - Downloads prices from yfinance (default start/end from CLI), retries batches, converts `.KS` tickers to USD, drops crypto-only rows, forward-fills, validates no missing tickers/values.
  - Writes `data/prices.csv`.
  - Args: `--start-date`, `--end-date`, `--weights-root`, `--output`, `--chunk-size`, `--no-forward-fill`.

- `scripts/portfolio_visualizer.py`
  - Loads `data/prices.csv`, weights, and builds NAV paths with weekly rebalancing.
  - Clamps weekly schedule to available price data; missing prices raise errors.
  - Saves weekly NAV (`data/portfolio_nav.csv`), daily NAV timeseries, performance table (TotalReturn, simple annualized return/vol, Sharpe with RF 4.35%, MDD, FinalNAV), and `data/portfolio_nav.svg`.
  - Key params: `--week4-start`, `--capital`, `--prices`.

- `scripts/build_site.py`
  - Orchestrates: (optional) price download via `build_price_history.py`, then `portfolio_visualizer.py`, copies artifacts to `site/`, and builds `site/index.html`.
  - Default date range: start fixed at 2025-09-30, end = latest business day; schedules are clamped to available prices.
  - Args mirror the price download options plus `--skip-download`/`--existing-prices`.

## GitHub Actions
- Workflow: `.github/workflows/deploy.yml`
  - Runs on push/schedule (`cron: 30 9 * * 1-5`) and on dispatch.
  - Steps: install `requirements.txt` (pandas, yfinance), run `python scripts/build_site.py`, upload `site/` as Pages artifact, deploy with `actions/deploy-pages@v4`.
  - Environment: ensure Pages is set to “GitHub Actions” and `github-pages` environment allows `main`.

## Data Conventions
- Prices are adjusted (total-return style) via yfinance `auto_adjust=True`. KRW tickers are scaled by `KRW=X`. Weekend crypto-only rows are dropped. Cash weights are ignored in the price table.
- Week schedules: week 4 starts at `--week4-start`; each week runs 7 days. Schedule is clamped to the last price date to avoid failures when the price horizon is shorter.
- Sharpe uses RF = 0.0435. Annualized return uses simple mean daily return * 252; vol uses std_daily * sqrt(252).

## Typical Commands
- Rebuild everything locally with existing prices:
  ```bash
  python3.11 scripts/build_site.py --skip-download --existing-prices data/prices.csv --week4-start 2025-09-30
  ```
- Redownload prices (network required) then rebuild:
  ```bash
  python3.11 scripts/build_site.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --week4-start 2025-09-30
  ```

## Common Issues & Notes
- yfinance can intermittently fail; `build_price_history.py` retries batches and single tickers. If a ticker still fails, the script raises an explicit error listing missing tickers.
- Ensure `data/prices.csv` covers the latest week; otherwise the schedule is clamped and NAV stops early.
- If running in Colab, use `notebooks/build_price_from_colab.ipynb` to fetch prices from Drive-hosted weights, handle KRW/crypto adjustments, and download `data/prices.csv`.

## What to Adjust for New Research
- Update/extend `data/weights/weight_week*/` as needed; rerun `build_site.py`.
- If you prefer raw (unadjusted) prices, set `auto_adjust=False` in `build_price_history.py` and regenerate `prices.csv`.
- To change RF or annualization, edit constants in `scripts/portfolio_visualizer.py` (`RF_RATE`, `TRADING_DAYS_PER_YEAR`).
