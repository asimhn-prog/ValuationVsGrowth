"""
Compute 3-year forward revenue CAGR.

Definition
----------
  3Y Forward CAGR = (fwd_revenue_4y / fwd_revenue_1y) ^ (1/3) - 1

  Where:
    fwd_revenue_1y  = 1-year forward (NTM) revenue consensus   [FY+1]
    fwd_revenue_4y  = 4-year forward revenue consensus          [FY+4]

  This annualises the revenue growth from FY+1 to FY+4 – exactly 3 years
  of forward growth – which is the cleanest definition consistent with the
  X-axis label "3-year forward revenue CAGR".

Tier-3 fallback
---------------
  When forward revenue is unavailable, we fall back to the 3-year *trailing*
  revenue CAGR computed from TTM revenue history.  This is a lagging proxy –
  it reflects where growth *has been*, not where analysts expect it to go.
  It is labelled clearly as a Tier-3 approximation.

  Formula (trailing): (ttm_revenue_t / ttm_revenue_{t-36months}) ^ (1/3) - 1

NaN handling
------------
  Any row where fwd_revenue_1y ≤ 0, fwd_revenue_4y ≤ 0, or either is NaN
  produces NaN for fwd_cagr_3y.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_N_YEARS = 3


def compute_forward_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``fwd_cagr_3y`` and ``fwd_cagr_tier`` columns to the panel.

    The panel must be sorted by (ticker, date) and have a 'date' column.

    Parameters
    ----------
    df : pd.DataFrame  Panel with optional fwd_revenue_1y / fwd_revenue_4y
                       and/or ttm_revenue columns.

    Returns
    -------
    pd.DataFrame  Copy with additional columns:
                    fwd_cagr_3y   – annualised 3-year revenue CAGR
                    fwd_cagr_tier – 1 (forward) or 3 (trailing proxy)
    """
    df = df.copy()

    has_fwd = (
        "fwd_revenue_1y" in df.columns
        and "fwd_revenue_4y" in df.columns
        and df["fwd_revenue_1y"].notna().any()
        and df["fwd_revenue_4y"].notna().any()
    )

    if has_fwd:
        # ── Tier 1: forward CAGR ─────────────────────────────────────────────
        r1 = df["fwd_revenue_1y"]
        r4 = df["fwd_revenue_4y"]
        valid = (r1 > 0) & (r4 > 0) & r1.notna() & r4.notna()
        df["fwd_cagr_3y"] = np.where(
            valid, (r4 / r1) ** (1.0 / _N_YEARS) - 1, np.nan
        )
        df["fwd_cagr_tier"] = np.where(valid, 1, np.nan)
        n_inv = (~valid).sum()
        if n_inv:
            logger.warning(
                "%d rows have invalid forward revenue for CAGR (set to NaN).", n_inv
            )
        logger.info("fwd_cagr_3y: Tier 1 (forward estimates).")

    elif "ttm_revenue" in df.columns and df["ttm_revenue"].notna().any():
        # ── Tier 3: trailing CAGR ────────────────────────────────────────────
        logger.info(
            "fwd_cagr_3y: Tier-3 proxy using 3Y trailing TTM revenue CAGR."
        )
        df = df.sort_values(["ticker", "date"]).copy()
        lag_periods = 36  # 36 monthly periods ≈ 3 years

        cagr_parts: list[pd.Series] = []
        tier_parts: list[pd.Series] = []

        for ticker, grp in df.groupby("ticker", sort=False):
            rev = grp["ttm_revenue"].values.astype(float)
            idx = grp.index

            if len(rev) <= lag_periods:
                cagr_parts.append(pd.Series(np.nan, index=idx))
                tier_parts.append(pd.Series(np.nan, index=idx))
                continue

            rev_lag   = np.full_like(rev, np.nan)
            rev_lag[lag_periods:] = rev[:-lag_periods]

            valid  = (rev > 0) & (rev_lag > 0)
            cagr   = np.where(
                valid, (rev / rev_lag) ** (1.0 / _N_YEARS) - 1, np.nan
            )
            tier   = np.where(~np.isnan(cagr), 3, np.nan)

            cagr_parts.append(pd.Series(cagr, index=idx))
            tier_parts.append(pd.Series(tier, index=idx))

        df["fwd_cagr_3y"]   = pd.concat(cagr_parts).sort_index()
        df["fwd_cagr_tier"] = pd.concat(tier_parts).sort_index()

    else:
        logger.warning(
            "Neither forward nor trailing revenue found; fwd_cagr_3y = NaN."
        )
        df["fwd_cagr_3y"]   = np.nan
        df["fwd_cagr_tier"] = np.nan

    return df
