"""
Winsorisation utilities.

Two methods
-----------
  A) percentile  – clip values below quantile(p_low) and above quantile(1-p_high).
  B) stddev      – clip values outside mean ± n_std × std_dev.

Each method may be applied independently to any column in the panel.
Winsorisation is applied *cross-sectionally* across all rows supplied (i.e.
all tickers and dates together), not per-ticker.  NaN values are preserved.

Usage
-----
  from transforms.winsor import winsorise_panel

  df_clean = winsorise_panel(
      df,
      columns=["fwd_cagr_3y", "ey", "ebitda_yld"],
      method="percentile",
      p_low=0.025, p_high=0.025,
  )
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WinsorMethod = Literal["percentile", "stddev"]


def winsorise_series(
    s: pd.Series,
    method: WinsorMethod = "percentile",
    p_low: float = 0.025,
    p_high: float = 0.025,
    n_std: float = 2.5,
) -> pd.Series:
    """
    Winsorise a single numeric Series.

    Parameters
    ----------
    s       : pd.Series
    method  : 'percentile' | 'stddev'
    p_low   : lower tail trim fraction (percentile method)
    p_high  : upper tail trim fraction (percentile method)
    n_std   : σ multiplier (stddev method)

    Returns
    -------
    pd.Series  Same index and NaN pattern; values clipped to [lo, hi].
    """
    s = s.copy()
    valid = s.dropna()
    if valid.empty:
        return s

    if method == "percentile":
        if not (0.0 <= p_low < 0.5 and 0.0 <= p_high < 0.5):
            raise ValueError(
                f"p_low and p_high must be in [0, 0.5). Got {p_low}, {p_high}."
            )
        lo = float(valid.quantile(p_low))
        hi = float(valid.quantile(1.0 - p_high))

    elif method == "stddev":
        mu  = float(valid.mean())
        sig = float(valid.std(ddof=1))
        if sig == 0:
            return s  # no variation; nothing to clip
        lo = mu - n_std * sig
        hi = mu + n_std * sig

    else:
        raise ValueError(f"Unknown winsor method '{method}'. Use 'percentile' or 'stddev'.")

    n_clipped = int(((s < lo) | (s > hi)).sum())
    if n_clipped:
        logger.debug(
            "Winsor '%s' [%s]: clipped %d values to [%.4g, %.4g].",
            s.name, method, n_clipped, lo, hi,
        )

    return s.clip(lower=lo, upper=hi)


def winsorise_panel(
    df: pd.DataFrame,
    columns: list[str],
    method: WinsorMethod = "percentile",
    p_low: float = 0.025,
    p_high: float = 0.025,
    n_std: float = 2.5,
) -> pd.DataFrame:
    """
    Apply winsorisation to specified columns of a panel DataFrame.

    Parameters
    ----------
    df      : pd.DataFrame  Panel to winsorise (original is not modified).
    columns : list[str]     Column names to winsorise.
    method  : WinsorMethod  'percentile' or 'stddev'.
    p_low   : float         Lower percentile clip (percentile method).
    p_high  : float         Upper percentile clip (percentile method).
    n_std   : float         Std-dev multiplier (stddev method).

    Returns
    -------
    pd.DataFrame  Copy with winsorised columns; untouched columns unchanged.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Winsor: column '%s' not in DataFrame; skipped.", col)
            continue
        df[col] = winsorise_series(
            df[col].rename(col),
            method=method,
            p_low=p_low,
            p_high=p_high,
            n_std=n_std,
        )
    return df


def get_winsor_bounds(
    df: pd.DataFrame,
    columns: list[str],
    method: WinsorMethod = "percentile",
    p_low: float = 0.025,
    p_high: float = 0.025,
    n_std: float = 2.5,
) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """
    Return the (lo, hi) clip bounds for each column without modifying df.
    Useful for reporting the effective winsor range in the UI.
    """
    bounds: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for col in columns:
        if col not in df.columns:
            bounds[col] = (None, None)
            continue
        valid = df[col].dropna()
        if valid.empty:
            bounds[col] = (None, None)
            continue
        if method == "percentile":
            lo = float(valid.quantile(p_low))
            hi = float(valid.quantile(1.0 - p_high))
        else:
            mu  = float(valid.mean())
            sig = float(valid.std(ddof=1))
            lo  = mu - n_std * sig
            hi  = mu + n_std * sig
        bounds[col] = (lo, hi)
    return bounds
