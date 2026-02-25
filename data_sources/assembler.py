"""
Panel assembler: merges price data (yfinance) with forward or trailing
fundamentals (BYOD / yfinance) into a single tidy monthly panel.

Data-tier logic
---------------
  Tier 1 – ticker found in BYOD file: forward estimates used directly.
  Tier 3 – ticker NOT in BYOD:         trailing (TTM) metrics from yfinance,
            with 3Y trailing CAGR as a revenue-growth proxy.

Assembly steps
--------------
  1. Load BYOD (if path provided).
  2. Fetch monthly adjusted-close prices from yfinance for all tickers.
  3. For each ticker:
       Tier 1 – forward-fill BYOD fundamentals to monthly price dates.
       Tier 3 – compute TTM series from yfinance quarterly statements,
                resample to monthly, forward-fill.
  4. Compute market_cap = price × shares_outstanding where not supplied.
  5. Stack all tickers into a single DataFrame sorted by (ticker, date).

Output columns (best-effort; NaN where unavailable)
----------------------------------------------------
  date, ticker, price, shares_outstanding, market_cap, total_debt, cash,
  enterprise_value,
  fwd_eps_1y,  fwd_revenue_1y,  fwd_ebitda_1y,  fwd_fcf_1y,  fwd_revenue_4y,
  ttm_revenue, ttm_ebitda, ttm_net_income, ttm_fcf,
  data_tier
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data_sources.byod import load_byod, validate_byod
from data_sources.yahoo import fetch_price_history, fetch_trailing_fundamentals

logger = logging.getLogger(__name__)

# Fundamental columns supplied by BYOD
_BYOD_FUND_COLS = [
    "shares_outstanding", "total_debt", "cash",
    "fwd_eps_1y", "fwd_revenue_1y", "fwd_ebitda_1y", "fwd_fcf_1y",
    "fwd_revenue_4y", "market_cap", "enterprise_value",
]

# Fundamental columns supplied by yfinance trailing
_YF_TRAIL_COLS = [
    "shares_outstanding", "total_debt", "cash",
    "ttm_revenue", "ttm_ebitda", "ttm_net_income", "ttm_fcf",
]


def _align_byod_to_prices(
    price_series: pd.Series,
    byod_ticker: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Align BYOD fundamentals to the monthly price index via forward-fill.

    Parameters
    ----------
    price_series : pd.Series
        Monthly prices indexed by date.
    byod_ticker  : pd.DataFrame
        Subset of BYOD for one ticker, not yet indexed by date.
    freq         : str
        Target resampling frequency.

    Returns
    -------
    pd.DataFrame indexed by date, same index as price_series.
    """
    fund = byod_ticker.set_index("date").sort_index()

    # Keep only fundamental columns (drop ticker, price from BYOD –
    # we use yfinance price instead, which is daily-clean)
    keep = [c for c in _BYOD_FUND_COLS if c in fund.columns]
    fund = fund[keep]

    # Resample BYOD to target freq (handles quarterly/annual BYOD)
    fund = fund.resample(freq).last()

    # Reindex to the full monthly price index, then ffill
    merged = fund.reindex(price_series.index).ffill()
    merged.insert(0, "price", price_series.values)
    return merged


def _build_tier3_panel(
    ticker: str,
    price_series: pd.Series,
    freq: str,
) -> pd.DataFrame:
    """
    Build a Tier-3 panel from yfinance trailing fundamentals.

    Returns a DataFrame indexed by date with price + TTM fundamentals.
    """
    trail = fetch_trailing_fundamentals(ticker, freq=freq)

    base = pd.DataFrame({"price": price_series})

    if trail.empty:
        logger.warning("%s: no trailing fundamentals; only prices available.", ticker)
        return base

    keep = [c for c in _YF_TRAIL_COLS if c in trail.columns]
    merged = base.join(trail[keep], how="left").ffill()

    # Backfill balance-sheet stock items so the full date range is covered.
    # Annual extension in fetch_trailing_fundamentals may not reach far enough
    # back; bfill propagates the earliest known value to earlier months.
    # Flow items (ttm_revenue etc.) are NOT backfilled – they would give a
    # flat/zero CAGR rather than a meaningful estimate.
    stock_cols = [c for c in ["shares_outstanding", "total_debt", "cash"] if c in merged.columns]
    if stock_cols:
        merged[stock_cols] = merged[stock_cols].bfill()

    return merged


def build_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    byod_path: Optional[str] = None,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Assemble the full panel for the given universe.

    Parameters
    ----------
    tickers    : list[str]  All tickers (primary + basket + market proxy).
    start_date : str        ISO date string '2018-01-01'.
    end_date   : str        ISO date string '2024-12-31'.
    byod_path  : str | None Path to BYOD CSV/Excel, or None.
    freq       : str        Pandas offset alias; default 'ME' (month-end).

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, price, shares_outstanding, market_cap,
                 total_debt, cash, enterprise_value,
                 fwd_* (if BYOD), ttm_* (if Tier-3),
                 data_tier.
        Sorted by (ticker, date).
    """
    tickers = [t.upper().strip() for t in tickers]
    tickers_unique = list(dict.fromkeys(tickers))  # preserve order, deduplicate

    # ── Load BYOD ────────────────────────────────────────────────────────────
    byod_df: Optional[pd.DataFrame] = None
    byod_tickers: set[str] = set()

    if byod_path:
        try:
            byod_df = load_byod(byod_path)
            issues = validate_byod(byod_df)
            for w in issues:
                logger.warning("BYOD validation: %s", w)
            byod_tickers = set(byod_df["ticker"].unique())
            logger.info("BYOD tickers found: %s", sorted(byod_tickers))
        except Exception as exc:
            logger.error("Failed to load BYOD: %s – falling back to Tier 3.", exc)

    # ── Fetch price history for all tickers ─────────────────────────────────
    price_frames: dict[str, pd.Series] = {}
    for ticker in tickers_unique:
        pf = fetch_price_history(ticker, start_date, end_date, freq=freq)
        if not pf.empty and ticker in pf.columns:
            price_frames[ticker] = pf[ticker].dropna()
        else:
            logger.warning("No prices for %s; excluded from panel.", ticker)

    if not price_frames:
        logger.error("No price data retrieved for any ticker.")
        return pd.DataFrame()

    # ── Build per-ticker panels ──────────────────────────────────────────────
    panels: list[pd.DataFrame] = []

    for ticker in tickers_unique:
        if ticker not in price_frames:
            continue

        px = price_frames[ticker]

        if byod_df is not None and ticker in byod_tickers:
            # ── Tier 1: BYOD ────────────────────────────────────────────────
            ticker_byod = byod_df[byod_df["ticker"] == ticker].copy()
            panel = _align_byod_to_prices(px, ticker_byod, freq)
            panel["data_tier"] = 1
            logger.info("%s: Tier 1 (BYOD) – %d rows", ticker, len(panel))
        else:
            # ── Tier 3: yfinance trailing ────────────────────────────────────
            panel = _build_tier3_panel(ticker, px, freq)
            panel["data_tier"] = 3
            logger.info("%s: Tier 3 (trailing) – %d rows", ticker, len(panel))

        # Derive market_cap if not supplied
        if "market_cap" not in panel.columns or panel["market_cap"].isna().all():
            if "shares_outstanding" in panel.columns:
                panel["market_cap"] = panel["price"] * panel["shares_outstanding"]
            else:
                panel["market_cap"] = np.nan

        # Derive EV if not supplied
        if "enterprise_value" not in panel.columns or panel["enterprise_value"].isna().all():
            td = panel.get("total_debt",  pd.Series(np.nan, index=panel.index))
            ca = panel.get("cash",        pd.Series(np.nan, index=panel.index))
            mc = panel.get("market_cap",  pd.Series(np.nan, index=panel.index))
            ev = mc + td.fillna(0) - ca.fillna(0)
            panel["enterprise_value"] = np.where(ev > 0, ev, np.nan)

        panel.index.name = "date"
        panel = panel.reset_index()
        panel["ticker"] = ticker
        panels.append(panel)

    if not panels:
        logger.error("Panel is empty after assembly.")
        return pd.DataFrame()

    out = pd.concat(panels, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(
        "Panel assembled: %d rows | %d tickers | %s – %s",
        len(out),
        out["ticker"].nunique(),
        out["date"].min().date(),
        out["date"].max().date(),
    )
    return out
