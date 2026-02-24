"""
Enterprise Value sanity-check and repair.

EV = market_cap + total_debt - cash

If the panel already contains a valid (> 0) enterprise_value, it is kept.
Otherwise EV is (re)computed from components.  Rows where EV cannot be
computed are set to NaN with a warning.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_ev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a valid ``enterprise_value`` column in the panel.

    Steps
    -----
    1. If ``enterprise_value`` is missing or ≤ 0, compute from components.
    2. If components (market_cap, total_debt, cash) are missing, attempt to
       derive market_cap from price × shares_outstanding.
    3. Any remaining non-positive EV is set to NaN.

    Parameters
    ----------
    df : pd.DataFrame  Panel with at minimum a 'price' column.

    Returns
    -------
    pd.DataFrame  Copy with a clean 'enterprise_value' column.
    """
    df = df.copy()

    # ── Derive market_cap if absent ──────────────────────────────────────────
    if "market_cap" not in df.columns or df["market_cap"].isna().all():
        if "price" in df.columns and "shares_outstanding" in df.columns:
            df["market_cap"] = df["price"] * df["shares_outstanding"]
            logger.debug("market_cap derived from price × shares_outstanding.")
        else:
            df["market_cap"] = np.nan

    # ── Compute EV from components ───────────────────────────────────────────
    td = df.get("total_debt", pd.Series(0.0, index=df.index)).fillna(0)
    ca = df.get("cash",       pd.Series(0.0, index=df.index)).fillna(0)
    mc = df["market_cap"]
    computed = mc + td - ca

    if "enterprise_value" not in df.columns:
        df["enterprise_value"] = np.where(computed > 0, computed, np.nan)
    else:
        # Replace missing or non-positive with computed value
        bad = df["enterprise_value"].isna() | (df["enterprise_value"] <= 0)
        df.loc[bad, "enterprise_value"] = np.where(
            computed[bad] > 0, computed[bad], np.nan
        )

    n_nan = df["enterprise_value"].isna().sum()
    if n_nan:
        logger.warning(
            "%d rows have enterprise_value ≤ 0 or uncomputable (set to NaN).", n_nan
        )

    return df
