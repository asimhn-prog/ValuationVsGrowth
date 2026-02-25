"""
Compute forward (or trailing) valuation yield metrics.

Metric definitions
------------------
  ey          = fwd_eps_1y    / price                  (earnings yield; 1/P×E)
  ebitda_yld  = fwd_ebitda_1y / enterprise_value
  sales_yld   = fwd_revenue_1y / enterprise_value
  fcf_yld     = fwd_fcf_1y   / enterprise_value        (fallback: / market_cap)

Mixed-tier handling
-------------------
  When a panel contains both Tier-1 (BYOD forward) and Tier-3 (yfinance
  trailing) rows, each metric is computed **per row**:
    - Use forward estimate if the row's fwd_* column is non-NaN.
    - Otherwise fall back to the trailing (TTM) equivalent.
  This ensures basket tickers (Tier 3) always produce valid yields even when
  the primary ticker is Tier 1.

Tier labels
-----------
  Each metric column 'X' has a companion 'X_tier':
    1 → forward estimate (BYOD or yfinance forwardEps)
    3 → trailing/TTM proxy

Negative / zero denominator handling
--------------------------------------
  Any row where the denominator is ≤ 0 or NaN yields NaN for that metric.
  A debug log records the count.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metric display labels
YIELD_METRIC_LABELS: dict[str, str] = {
    "ey":        "Earnings Yield (E/P)",
    "ebitda_yld": "EBITDA / EV Yield",
    "sales_yld":  "Sales / EV Yield",
    "fcf_yld":    "FCF Yield",
}

# Human-readable multiple name for each yield
MULTIPLE_LABELS: dict[str, str] = {
    "ey":        "Forward P/E",
    "ebitda_yld": "EV / EBITDA",
    "sales_yld":  "EV / Sales",
    "fcf_yld":    "EV / FCF  (or P / FCF)",
}

ALL_METRICS = list(YIELD_METRIC_LABELS.keys())


def _safe_divide(
    num: pd.Series,
    den: pd.Series,
    label: str,
) -> pd.Series:
    """
    Element-wise division, returning NaN wherever den ≤ 0 or either is NaN.
    """
    valid = (den > 0) & den.notna() & num.notna()
    result = np.where(valid, num / den, np.nan)
    n_bad = (~valid).sum()
    if n_bad:
        logger.debug("%s: %d invalid rows (denominator ≤ 0 or NaN).", label, n_bad)
    return pd.Series(result, index=num.index, dtype=float)


def _blend(
    df: pd.DataFrame,
    fwd_col: Optional[str],
    ttm_col: Optional[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Return (numerator_series, tier_series) blending forward and TTM columns.

    Forward values take priority; TTM fills rows where forward is NaN.
    tier=1 for rows using fwd_col, tier=3 for rows using ttm_col.
    """
    nan_s = pd.Series(np.nan, index=df.index, dtype=float)

    fwd  = df[fwd_col].copy().astype(float) if fwd_col and fwd_col in df.columns else nan_s.copy()
    tier = pd.Series(1, index=df.index, dtype=float)

    if ttm_col and ttm_col in df.columns:
        missing = fwd.isna()
        if missing.any():
            fwd[missing]  = df.loc[missing, ttm_col].astype(float)
            tier[missing] = 3

    return fwd, tier


def compute_valuation_yields(
    df: pd.DataFrame,
    fcf_use_ev: bool = True,
) -> pd.DataFrame:
    """
    Add yield columns (ey, ebitda_yld, sales_yld, fcf_yld) and their _tier
    companions to the panel DataFrame.

    Each metric is computed per row: forward estimates are used where
    available (Tier 1), with TTM trailing values as fallback (Tier 3).
    This correctly handles mixed-tier panels (e.g. BYOD primary + yfinance
    basket).

    Parameters
    ----------
    df         : pd.DataFrame  Panel (must contain price, enterprise_value,
                               and forward or TTM fundamental columns).
    fcf_use_ev : bool          If True, FCF yield denominator = EV;
                               if False, denominator = market_cap.
                               Falls back to market_cap if EV is NaN.

    Returns
    -------
    pd.DataFrame  Copy with additional columns per metric.
    """
    df = df.copy()

    # ── Earnings Yield ────────────────────────────────────────────────────────
    # TTM EPS = ttm_net_income / shares_outstanding
    ttm_eps: Optional[pd.Series] = None
    if "ttm_net_income" in df.columns and "shares_outstanding" in df.columns:
        ttm_eps = df["ttm_net_income"] / df["shares_outstanding"].replace(0, np.nan)

    ey_fwd_col = "fwd_eps_1y" if "fwd_eps_1y" in df.columns else None
    if ey_fwd_col or ttm_eps is not None:
        fwd_eps = df[ey_fwd_col].copy().astype(float) if ey_fwd_col else pd.Series(np.nan, index=df.index)
        ey_tier = pd.Series(1, index=df.index, dtype=float)
        if ttm_eps is not None:
            missing = fwd_eps.isna()
            if missing.any():
                fwd_eps[missing] = ttm_eps[missing]
                ey_tier[missing] = 3
        if fwd_eps.notna().any():
            df["ey"]      = _safe_divide(fwd_eps, df["price"], "EY")
            df["ey_tier"] = ey_tier
        else:
            df["ey"]      = np.nan
            df["ey_tier"] = np.nan
    else:
        df["ey"]      = np.nan
        df["ey_tier"] = np.nan

    # ── EBITDA / EV Yield ─────────────────────────────────────────────────────
    ebitda_num, ebitda_tier = _blend(df, "fwd_ebitda_1y", "ttm_ebitda")
    if ebitda_num.notna().any():
        df["ebitda_yld"]      = _safe_divide(ebitda_num, df["enterprise_value"], "EBITDA_YLD")
        df["ebitda_yld_tier"] = ebitda_tier
    else:
        df["ebitda_yld"]      = np.nan
        df["ebitda_yld_tier"] = np.nan

    # ── Sales / EV Yield ──────────────────────────────────────────────────────
    sales_num, sales_tier = _blend(df, "fwd_revenue_1y", "ttm_revenue")
    if sales_num.notna().any():
        df["sales_yld"]      = _safe_divide(sales_num, df["enterprise_value"], "SALES_YLD")
        df["sales_yld_tier"] = sales_tier
    else:
        df["sales_yld"]      = np.nan
        df["sales_yld_tier"] = np.nan

    # ── FCF Yield ─────────────────────────────────────────────────────────────
    fcf_num, fcf_tier = _blend(df, "fwd_fcf_1y", "ttm_fcf")
    if fcf_num.notna().any():
        if fcf_use_ev:
            denom = df["enterprise_value"].copy()
            # Row-level fallback to market_cap where EV is NaN
            if "market_cap" in df.columns:
                fallback = denom.isna() & df["market_cap"].notna()
                if fallback.any():
                    denom[fallback] = df.loc[fallback, "market_cap"]
                    logger.info(
                        "FCF_YLD: market_cap used for %d rows (EV unavailable).",
                        fallback.sum(),
                    )
        else:
            denom = df.get("market_cap", pd.Series(np.nan, index=df.index))
        df["fcf_yld"]      = _safe_divide(fcf_num, denom, "FCF_YLD")
        df["fcf_yld_tier"] = fcf_tier
    else:
        df["fcf_yld"]      = np.nan
        df["fcf_yld_tier"] = np.nan

    return df


def yields_to_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inverse-yield (multiple) columns: pe, ev_ebitda, ev_sales, ev_fcf.

    Returns NaN where yield ≤ 0.
    """
    df = df.copy()
    mapping = {
        "ey":        "pe",
        "ebitda_yld": "ev_ebitda",
        "sales_yld":  "ev_sales",
        "fcf_yld":    "ev_fcf",
    }
    for yld, mult in mapping.items():
        if yld in df.columns:
            df[mult] = np.where(
                df[yld].notna() & (df[yld] > 0),
                1.0 / df[yld],
                np.nan,
            )
    return df
