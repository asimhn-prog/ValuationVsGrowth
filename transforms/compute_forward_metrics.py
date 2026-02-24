"""
Compute forward (or trailing) valuation yield metrics.

Metric definitions
------------------
  ey          = fwd_eps_1y    / price                  (earnings yield; 1/P×E)
  ebitda_yld  = fwd_ebitda_1y / enterprise_value
  sales_yld   = fwd_revenue_1y / enterprise_value
  fcf_yld     = fwd_fcf_1y   / enterprise_value        (fallback: / market_cap)

Tier labels
-----------
  Each metric column 'X' has a companion 'X_tier':
    1 → forward estimate (BYOD or yfinance forwardEps)
    3 → trailing/TTM proxy

Forward-horizon definition
--------------------------
  "1-year forward" means FY+1 / NTM (next-twelve-months) consensus estimates
  as of the observation date.  When only trailing data is available (Tier 3),
  the metric is labelled TTM and the _tier column is set to 3.

  Note: for Tier-3 EPS we use TTM net income / shares, which is a trailing
  approximation. It will typically differ from buy-side NTM estimates.

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


def compute_valuation_yields(
    df: pd.DataFrame,
    fcf_use_ev: bool = True,
) -> pd.DataFrame:
    """
    Add yield columns (ey, ebitda_yld, sales_yld, fcf_yld) and their _tier
    companions to the panel DataFrame.

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
    if "fwd_eps_1y" in df.columns and df["fwd_eps_1y"].notna().any():
        df["ey"]      = _safe_divide(df["fwd_eps_1y"], df["price"], "EY")
        df["ey_tier"] = 1
    elif "ttm_net_income" in df.columns and "shares_outstanding" in df.columns:
        ttm_eps = df["ttm_net_income"] / df["shares_outstanding"].replace(0, np.nan)
        df["ey"]      = _safe_divide(ttm_eps, df["price"], "EY_TTM")
        df["ey_tier"] = 3
        logger.info("EY: Tier-3 proxy (TTM net income / shares).")
    else:
        df["ey"]      = np.nan
        df["ey_tier"] = np.nan

    # ── EBITDA / EV Yield ─────────────────────────────────────────────────────
    if "fwd_ebitda_1y" in df.columns and df["fwd_ebitda_1y"].notna().any():
        df["ebitda_yld"]      = _safe_divide(
            df["fwd_ebitda_1y"], df["enterprise_value"], "EBITDA_YLD"
        )
        df["ebitda_yld_tier"] = 1
    elif "ttm_ebitda" in df.columns and df["ttm_ebitda"].notna().any():
        df["ebitda_yld"]      = _safe_divide(
            df["ttm_ebitda"], df["enterprise_value"], "EBITDA_YLD_TTM"
        )
        df["ebitda_yld_tier"] = 3
        logger.info("EBITDA_YLD: Tier-3 proxy (TTM EBITDA).")
    else:
        df["ebitda_yld"]      = np.nan
        df["ebitda_yld_tier"] = np.nan

    # ── Sales / EV Yield ──────────────────────────────────────────────────────
    if "fwd_revenue_1y" in df.columns and df["fwd_revenue_1y"].notna().any():
        df["sales_yld"]      = _safe_divide(
            df["fwd_revenue_1y"], df["enterprise_value"], "SALES_YLD"
        )
        df["sales_yld_tier"] = 1
    elif "ttm_revenue" in df.columns and df["ttm_revenue"].notna().any():
        df["sales_yld"]      = _safe_divide(
            df["ttm_revenue"], df["enterprise_value"], "SALES_YLD_TTM"
        )
        df["sales_yld_tier"] = 3
        logger.info("SALES_YLD: Tier-3 proxy (TTM revenue).")
    else:
        df["sales_yld"]      = np.nan
        df["sales_yld_tier"] = np.nan

    # ── FCF Yield ─────────────────────────────────────────────────────────────
    fcf_col  = "fwd_fcf_1y"  if ("fwd_fcf_1y"  in df.columns and df["fwd_fcf_1y"].notna().any()) \
          else "ttm_fcf"     if ("ttm_fcf"      in df.columns and df["ttm_fcf"].notna().any()) \
          else None
    fcf_tier = 1 if fcf_col == "fwd_fcf_1y" else (3 if fcf_col == "ttm_fcf" else None)

    if fcf_col is not None:
        if fcf_use_ev:
            denom = df["enterprise_value"].copy()
            # Fall back to market_cap row-by-row where EV is NaN
            fallback = denom.isna() & df.get("market_cap", pd.Series(dtype=float)).notna()
            if fallback.any():
                denom[fallback] = df.loc[fallback, "market_cap"]
                logger.info("FCF_YLD: market_cap used for %d rows (EV unavailable).", fallback.sum())
        else:
            denom = df.get("market_cap", pd.Series(np.nan, index=df.index))

        df["fcf_yld"]      = _safe_divide(df[fcf_col], denom, "FCF_YLD")
        df["fcf_yld_tier"] = fcf_tier
        if fcf_tier == 3:
            logger.info("FCF_YLD: Tier-3 proxy (TTM FCF).")
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
