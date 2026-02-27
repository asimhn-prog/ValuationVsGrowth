"""
Compute revenue CAGR (3-year preferred, with 2Y/1Y fallback).

Tier-1 (forward estimates)
--------------------------
  3Y Forward CAGR = (fwd_revenue_4y / fwd_revenue_1y) ^ (1/3) - 1

  T_0 = fwd_revenue_1y (NTM, FY+1)
  T_3 = fwd_revenue_4y (FY+4)
  Annualises revenue growth over exactly 3 forward years.

Tier-3 fallback (trailing TTM)
-------------------------------
  When forward estimates are unavailable we use trailing TTM revenue history.
  Lookback is DATE-based (not row-count-based) so it is correctly annualised
  at any sampling frequency (daily, monthly, quarterly):

    3Y CAGR  = (rev_t / rev_{t-3yr}) ^ (1/actual_years) - 1   [preferred]
    2Y CAGR  = (rev_t / rev_{t-2yr}) ^ (1/actual_years) - 1   [fallback]
    1Y CAGR  = (rev_t / rev_{t-1yr}) ^ (1/actual_years) - 1   [last resort]

  actual_years is computed from the real calendar distance between the two
  observations, so the annualisation is exact rather than assumed.
  The horizon actually used is recorded in ``fwd_cagr_years`` (3, 2, or 1).

Mixed-tier handling
-------------------
  When the panel contains both Tier-1 (BYOD forward) and Tier-3 (yfinance
  trailing) rows, CAGR is computed **per row**:
    1. Forward CAGR is applied to any row that has valid fwd_revenue_1y AND
       fwd_revenue_4y.
    2. Trailing CAGR (3Y→2Y→1Y fallback) is then computed for rows still
       NaN, using ttm_revenue history per ticker.
  This ensures basket tickers (Tier 3) get a valid CAGR even when the primary
  is Tier 1 with forward estimates.

NaN handling
------------
  Rows where either revenue value is ≤ 0 or NaN produce NaN for fwd_cagr_3y.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_N_YEARS = 3


def compute_forward_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``fwd_cagr_3y``, ``fwd_cagr_years``, and ``fwd_cagr_tier`` columns
    to the panel.

    The panel must be sorted by (ticker, date) and have a 'date' column.
    Mixed-tier panels (Tier 1 + Tier 3 rows) are handled correctly: forward
    CAGR is used per row where available; trailing CAGR fills the rest.

    Parameters
    ----------
    df : pd.DataFrame  Panel with optional fwd_revenue_1y / fwd_revenue_4y
                       and/or ttm_revenue columns.

    Returns
    -------
    pd.DataFrame  Copy with additional columns:
                    fwd_cagr_3y    – annualised revenue CAGR
                    fwd_cagr_years – horizon used (3, 2, or 1)
                    fwd_cagr_tier  – 1 (forward) or 3 (trailing proxy)
    """
    df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    n = len(df)
    cagr_arr = np.full(n, np.nan)
    year_arr = np.full(n, np.nan)
    tier_arr = np.full(n, np.nan)

    # ── Step 1: Forward CAGR (row-by-row) ────────────────────────────────────
    has_fwd_cols = (
        "fwd_revenue_1y" in df.columns
        and "fwd_revenue_4y" in df.columns
    )
    if has_fwd_cols:
        r1 = df["fwd_revenue_1y"].values.astype(float)
        r4 = df["fwd_revenue_4y"].values.astype(float)
        valid_fwd = (r1 > 0) & (r4 > 0) & np.isfinite(r1) & np.isfinite(r4)
        cagr_arr = np.where(valid_fwd, (r4 / r1) ** (1.0 / _N_YEARS) - 1, cagr_arr)
        year_arr = np.where(valid_fwd, float(_N_YEARS), year_arr)
        tier_arr = np.where(valid_fwd, 1.0, tier_arr)
        n_fwd = int(valid_fwd.sum())
        if n_fwd:
            logger.info(
                "fwd_cagr_3y: %d rows using Tier-1 (forward estimates).", n_fwd
            )

    # ── Step 2: Trailing CAGR for rows still NaN ─────────────────────────────
    still_nan_mask = np.isnan(cagr_arr)
    has_ttm = "ttm_revenue" in df.columns and df["ttm_revenue"].notna().any()

    if has_ttm and still_nan_mask.any():
        logger.info(
            "fwd_cagr_3y: filling %d rows with Tier-3 trailing CAGR "
            "(3Y -> 2Y -> 1Y fallback).",
            int(still_nan_mask.sum()),
        )
        for ticker, grp in df.groupby("ticker", sort=False):
            idx  = grp.index.to_numpy()
            # Only process tickers that still have at least one NaN CAGR row
            if not still_nan_mask[idx].any():
                continue

            dates = grp["date"].values  # numpy datetime64[ns], sorted ascending
            rev   = grp["ttm_revenue"].values.astype(float)
            m     = len(rev)

            local_cagr = cagr_arr[idx].copy()
            local_year = year_arr[idx].copy()

            # Date-based lookback: avoids the row-offset assumption that the data
            # is monthly.  Works correctly for daily, monthly, and quarterly freq.
            for target_years in [3, 2, 1]:
                # Approximate cutoff: target_years × 365.25 days back
                lookback = np.timedelta64(int(target_years * 365.25), "D")
                cutoff_dates = dates - lookback  # shape (m,)

                # For each row i, find the last j where dates[j] <= cutoff_dates[i]
                # and rev[j] is valid, using searchsorted for O(log m) per row.
                j_upper = np.searchsorted(dates, cutoff_dates, side="right") - 1

                rev_lag      = np.full(m, np.nan)
                actual_years = np.full(m, np.nan)

                for i in range(m):
                    if j_upper[i] < 0:
                        continue
                    # Scan backward from j_upper[i] to find the most recent valid rev
                    for j in range(j_upper[i], -1, -1):
                        if rev[j] > 0 and np.isfinite(rev[j]):
                            rev_lag[i] = rev[j]
                            days = (dates[i] - dates[j]).astype("timedelta64[D]").astype(int)
                            actual_years[i] = days / 365.25
                            break

                fill = (
                    np.isnan(local_cagr)
                    & (rev > 0) & np.isfinite(rev)
                    & np.isfinite(rev_lag)
                    & (actual_years >= 0.5)   # require at least 6 months of history
                )
                local_cagr = np.where(
                    fill, (rev / rev_lag) ** (1.0 / actual_years) - 1, local_cagr
                )
                local_year = np.where(fill, float(target_years), local_year)

            # Mark newly filled rows as Tier 3
            newly_filled = ~np.isnan(local_cagr) & np.isnan(tier_arr[idx])
            local_tier = np.where(newly_filled, 3.0, tier_arr[idx])

            cagr_arr[idx] = local_cagr
            year_arr[idx] = local_year
            tier_arr[idx] = local_tier

        n_t3 = int((tier_arr == 3.0).sum())
        if n_t3:
            logger.info("fwd_cagr_3y: %d rows using Tier-3 trailing CAGR.", n_t3)

    if np.isnan(cagr_arr).all():
        logger.warning(
            "Neither forward nor trailing revenue found; fwd_cagr_3y = NaN."
        )

    df["fwd_cagr_3y"]    = cagr_arr
    df["fwd_cagr_years"] = year_arr
    df["fwd_cagr_tier"]  = tier_arr

    return df
