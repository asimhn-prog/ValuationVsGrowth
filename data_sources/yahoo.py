"""
Yahoo Finance data source (via yfinance).

Provides:
  - Historical monthly prices (adjusted close)
  - Best-effort trailing fundamentals for Tier-3 fallback:
      TTM revenue, EBITDA, FCF, net income, shares, total_debt, cash
  - Best-effort current forward EPS / P/E from yfinance .info

Limitations
-----------
yfinance does NOT reliably provide consensus forward revenue, EBITDA, or FCF
estimates, nor 3Y-forward revenue. Use BYOD for those.  The forwardEps field
in .info is present for some US equities but is point-in-time only (no history).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Row-name mappings for yfinance financial statements ──────────────────────
_INCOME_ROWS = {
    "ttm_revenue":    ["Total Revenue"],
    "ttm_gross":      ["Gross Profit"],
    "ttm_ebit":       ["EBIT", "Operating Income"],
    "ttm_net_income": ["Net Income"],
}
_CF_ROWS = {
    "ttm_op_cf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "ttm_capex": ["Capital Expenditure"],
    "ttm_da":    ["Depreciation And Amortization", "Depreciation & Amortization"],
}
_BS_ROWS = {
    "total_debt":         ["Total Debt", "Long Term Debt And Capital Lease Obligation"],
    "cash":               ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
}


def _first_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    """Return the first matching row from a DataFrame, or None."""
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _ttm(series: pd.Series) -> pd.Series:
    """Rolling sum of last 4 quarters (TTM)."""
    s = series.sort_index().dropna()
    return s.rolling(4, min_periods=4).sum()


def fetch_price_history(
    ticker: str,
    start: str,
    end: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Fetch adjusted-close price history and resample to ``freq``.

    Parameters
    ----------
    ticker : str
    start  : str  ISO date, e.g. '2018-01-01'
    end    : str  ISO date, e.g. '2024-12-31'
    freq   : str  pandas offset alias (default 'ME' = month-end)

    Returns
    -------
    pd.DataFrame  single column named ``ticker``, indexed by date.
    """
    try:
        t = yf.Ticker(ticker)
        raw = t.history(start=start, end=end, auto_adjust=True)
        if raw.empty:
            logger.warning("No price data for %s", ticker)
            return pd.DataFrame(columns=[ticker])
        monthly = raw[["Close"]].resample(freq).last()
        if monthly.index.tz is not None:
            monthly.index = monthly.index.tz_localize(None)
        monthly.columns = [ticker]
        monthly.index.name = "date"
        return monthly
    except Exception as exc:
        logger.error("fetch_price_history failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=[ticker])


def fetch_basic_info(ticker: str) -> dict:
    """
    Fetch current snapshot from yfinance .info.

    Returns a flat dict with best-effort values; missing fields are None.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":             ticker.upper(),
            "forward_eps":        info.get("forwardEps"),
            "trailing_eps":       info.get("trailingEps"),
            "forward_pe":         info.get("forwardPE"),
            "trailing_pe":        info.get("trailingPE"),
            "price":              info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap":         info.get("marketCap"),
            "total_debt":         info.get("totalDebt"),
            "cash":               info.get("totalCash"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "trailing_revenue":   info.get("totalRevenue"),
            "trailing_ebitda":    info.get("ebitda"),
            "free_cash_flow":     info.get("freeCashflow"),
            "enterprise_value":   info.get("enterpriseValue"),
        }
    except Exception as exc:
        logger.error("fetch_basic_info failed for %s: %s", ticker, exc)
        return {"ticker": ticker.upper()}


def fetch_trailing_fundamentals(
    ticker: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Tier-3 fallback: build a monthly panel of trailing (TTM) fundamentals.

    Uses quarterly income statement, cash flow, and balance sheet from yfinance.
    TTM = sum of last 4 quarters for flow items; last available for stock items.

    Returned columns
    ----------------
    ttm_revenue, ttm_ebitda, ttm_net_income, ttm_fcf,
    total_debt, cash, shares_outstanding
    """
    t = yf.Ticker(ticker)
    try:
        inc_q = t.quarterly_income_stmt      # cols = quarters, index = line items
        cf_q  = t.quarterly_cashflow
        bs_q  = t.quarterly_balance_sheet
    except Exception as exc:
        logger.warning("Cannot fetch quarterly statements for %s: %s", ticker, exc)
        return pd.DataFrame()

    rows: dict[str, pd.Series] = {}

    # ── Income statement TTM ─────────────────────────────────────────────────
    if inc_q is not None and not inc_q.empty:
        for col, candidates in _INCOME_ROWS.items():
            s = _first_row(inc_q, candidates)
            if s is not None:
                rows[col] = _ttm(s)

    # ── Cash-flow TTM ────────────────────────────────────────────────────────
    if cf_q is not None and not cf_q.empty:
        op_cf = _first_row(cf_q, _CF_ROWS["ttm_op_cf"])
        capex = _first_row(cf_q, _CF_ROWS["ttm_capex"])
        da    = _first_row(cf_q, _CF_ROWS["ttm_da"])

        if op_cf is not None:
            op_ttm  = _ttm(op_cf)
            rows["ttm_op_cf"] = op_ttm

        if op_cf is not None and capex is not None:
            # capex is negative in yfinance; FCF = opCF + capex
            rows["ttm_fcf"] = _ttm(op_cf) + _ttm(capex)

        if da is not None:
            da_ttm = _ttm(da)
            rows["ttm_da"] = da_ttm
            if "ttm_ebit" in rows:
                rows["ttm_ebitda"] = rows["ttm_ebit"] + da_ttm

    # ── Balance sheet (point-in-time, forward-filled) ────────────────────────
    if bs_q is not None and not bs_q.empty:
        for col, candidates in _BS_ROWS.items():
            s = _first_row(bs_q, candidates)
            if s is not None:
                rows[col] = s.sort_index().dropna()

    if not rows:
        logger.warning("No trailing fundamentals for %s", ticker)
        return pd.DataFrame()

    # ── Extend all series backward with annual data ───────────────────────────
    # yfinance quarterly statements only cover ~5 recent quarters.  Annual
    # statements (5 fiscal years) are used to fill earlier dates so that a
    # 3-year trailing CAGR and EV-based metrics are available for the full
    # requested date range.
    def _extend_backward(rows_dict: dict, col: str, ann_series: pd.Series) -> None:
        """Prepend annual data points that pre-date the existing quarterly data."""
        ann_data = ann_series.sort_index().dropna()
        ann_data.index = pd.to_datetime(ann_data.index)
        existing = rows_dict.get(col, pd.Series(dtype=float))
        existing_valid = existing.dropna()
        cutoff = existing_valid.index.min() if not existing_valid.empty else pd.Timestamp.max
        ann_early = ann_data[ann_data.index < cutoff]
        if ann_early.empty:
            return
        if col in rows_dict:
            rows_dict[col] = pd.concat([ann_early, rows_dict[col]]).sort_index()
        else:
            rows_dict[col] = ann_early

    # Annual income statement → extend TTM income items
    try:
        ann_inc = t.income_stmt
        if ann_inc is not None and not ann_inc.empty:
            for col, candidates in _INCOME_ROWS.items():
                ann_row = _first_row(ann_inc, candidates)
                if ann_row is not None:
                    _extend_backward(rows, col, ann_row)
            # Also try to extend ttm_ebitda directly (annual stmt often has it)
            ann_ebitda = _first_row(ann_inc, ["EBITDA", "Normalized EBITDA"])
            if ann_ebitda is not None and "ttm_ebitda" not in rows:
                _extend_backward(rows, "ttm_ebitda", ann_ebitda)
            elif ann_ebitda is not None:
                _extend_backward(rows, "ttm_ebitda", ann_ebitda)
    except Exception as exc:
        logger.warning("%s: could not extend income stmt with annual data: %s", ticker, exc)

    # Annual cash-flow statement → extend FCF
    try:
        ann_cf = t.cashflow
        if ann_cf is not None and not ann_cf.empty:
            ann_op_cf = _first_row(ann_cf, _CF_ROWS["ttm_op_cf"])
            ann_capex = _first_row(ann_cf, _CF_ROWS["ttm_capex"])
            if ann_op_cf is not None:
                _extend_backward(rows, "ttm_op_cf", ann_op_cf)
            if ann_op_cf is not None and ann_capex is not None:
                ann_fcf = ann_op_cf.sort_index() + ann_capex.sort_index()
                _extend_backward(rows, "ttm_fcf", ann_fcf.dropna())
    except Exception as exc:
        logger.warning("%s: could not extend CF with annual data: %s", ticker, exc)

    # Annual balance sheet → extend shares, debt, cash (most critical for EV)
    try:
        ann_bs = t.balance_sheet
        if ann_bs is not None and not ann_bs.empty:
            for col, candidates in _BS_ROWS.items():
                ann_row = _first_row(ann_bs, candidates)
                if ann_row is not None:
                    _extend_backward(rows, col, ann_row)
    except Exception as exc:
        logger.warning("%s: could not extend balance sheet with annual data: %s", ticker, exc)

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().resample(freq).last().ffill()
    df.index.name = "date"

    # Rename ebit→ebitda if da was unavailable
    if "ttm_ebitda" not in df.columns and "ttm_ebit" in df.columns:
        df = df.rename(columns={"ttm_ebit": "ttm_ebitda"})
        logger.info("%s: EBITDA approximated as EBIT (D&A unavailable)", ticker)

    return df


# ── Self-check ────────────────────────────────────────────────────────────────
def self_check_price(df: pd.DataFrame, ticker: str) -> list[str]:
    """Validate a price DataFrame returned by fetch_price_history."""
    issues: list[str] = []
    if df.empty:
        issues.append(f"{ticker}: price DataFrame is empty")
        return issues
    if ticker not in df.columns:
        issues.append(f"{ticker}: column missing")
    else:
        pct_null = df[ticker].isna().mean()
        if pct_null > 0.1:
            issues.append(f"{ticker}: {pct_null:.0%} NaN prices")
        if (df[ticker].dropna() <= 0).any():
            issues.append(f"{ticker}: non-positive prices found")
    return issues
