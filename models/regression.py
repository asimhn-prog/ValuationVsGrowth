"""
OLS regression of valuation yield on revenue CAGR.

Two regression modes
--------------------
Cross-sectional (primary)
  At each date t, fit OLS across all basket tickers' snapshot values:
    yield_i(t) = alpha(t) + beta(t) * cagr_i(t) + ε
  Produces a time series of (alpha, beta, R², n_obs) — one row per date.
  The primary security's "fair" yield at t is:
    fair(t) = alpha(t) + beta(t) * cagr_primary(t)

Pooled (fallback / scatter reference)
  Single OLS estimated on all (ticker, date) observations pooled.
  Used when cross-sectional has too few data points, and to anchor
  the regression line on the snapshot scatter chart.

Notation matching the UI labels
--------------------------------
  alpha  → "c" (intercept / constant)
  beta   → "m" (slope)
  cagr   → "x" (primary's growth rate at that date)
  fair   → "y" = c + m·x  (fitted / fair valuation)

Premium / discount definitions
-------------------------------
  premium_yield    = actual_yield   - fair_yield        (absolute yield spread)
  premium_pct      = premium_yield  / |fair_yield|      (percentage)
  premium_multiple = actual_mult    / fair_mult   - 1   (multiple basis)
  premium_zscore   = z-score of premium_yield over the full history
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

_X_COL = "fwd_cagr_3y"


@dataclass
class RegressionResult:
    """Stores OLS output for one metric."""

    metric:   str
    x_col:    str
    alpha:    float
    beta:     float
    r2:       float
    n_obs:    int
    se_alpha: float = field(default=np.nan)
    se_beta:  float = field(default=np.nan)
    p_alpha:  float = field(default=np.nan)
    p_beta:   float = field(default=np.nan)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RegressionResult({self.metric}: α={self.alpha:.4f}, "
            f"β={self.beta:.4f}, R²={self.r2:.3f}, n={self.n_obs})"
        )


def run_ols(
    df: pd.DataFrame,
    x_col: str = _X_COL,
    y_col: str = "ey",
    min_obs: int = 10,
) -> Optional[RegressionResult]:
    """
    Run OLS: y = alpha + beta * x on the supplied DataFrame.

    Parameters
    ----------
    df      : pd.DataFrame  Panel (all tickers / dates used jointly).
    x_col   : str           CAGR column name.
    y_col   : str           Yield metric column name.
    min_obs : int           Minimum number of valid observations.

    Returns
    -------
    RegressionResult | None
    """
    data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < min_obs:
        logger.warning(
            "OLS %s ~ %s: only %d valid obs (min %d); skipped.",
            y_col, x_col, len(data), min_obs,
        )
        return None

    X = sm.add_constant(data[x_col], has_constant="add")
    y = data[y_col]

    try:
        fit = sm.OLS(y, X).fit()
    except Exception as exc:
        logger.error("OLS failed for %s: %s", y_col, exc)
        return None

    return RegressionResult(
        metric=y_col,
        x_col=x_col,
        alpha=float(fit.params.get("const", np.nan)),
        beta=float(fit.params.get(x_col, np.nan)),
        r2=float(fit.rsquared),
        n_obs=int(fit.nobs),
        se_alpha=float(fit.bse.get("const", np.nan)),
        se_beta=float(fit.bse.get(x_col, np.nan)),
        p_alpha=float(fit.pvalues.get("const", np.nan)),
        p_beta=float(fit.pvalues.get(x_col, np.nan)),
    )


def run_rolling_ols(
    df: pd.DataFrame,
    x_col: str = _X_COL,
    y_col: str = "ey",
    window: int = 36,
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling-window OLS over time.

    Uses an expanding-to-fixed window: for each unique date, the regression
    uses at most the last ``window`` distinct dates of observations.

    Parameters
    ----------
    df     : pd.DataFrame  Panel sorted by date.
    window : int           Number of most-recent months to include.

    Returns
    -------
    pd.DataFrame  Columns: date, alpha, beta, r2, n_obs.
                  Indexed by RangeIndex.
    """
    df_s = df.sort_values("date")
    all_dates = sorted(df_s["date"].unique())
    results: list[dict] = []

    for i, date in enumerate(all_dates):
        window_dates = all_dates[max(0, i - window + 1): i + 1]
        subset = df_s[df_s["date"].isin(window_dates)]
        res = run_ols(subset, x_col=x_col, y_col=y_col, min_obs=min_obs)
        if res is not None:
            results.append(
                {
                    "date":  pd.Timestamp(date),
                    "alpha": res.alpha,
                    "beta":  res.beta,
                    "r2":    res.r2,
                    "n_obs": res.n_obs,
                }
            )

    return pd.DataFrame(results)


def run_cross_sectional_ols(
    df: pd.DataFrame,
    x_col: str = _X_COL,
    y_col: str = "ey",
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    For each date in the basket panel, fit a cross-sectional OLS across all
    basket tickers whose (x, y) values are both valid at that date.

    Parameters
    ----------
    df      : pd.DataFrame  Basket panel (all tickers × dates).
    x_col   : str           CAGR column (x-axis).
    y_col   : str           Yield metric column (y-axis).
    min_obs : int           Minimum tickers with valid data to run regression.

    Returns
    -------
    pd.DataFrame  Columns: date, c (intercept/alpha), m (slope/beta), r2, n_obs.
                  One row per date where a regression was possible.
    """
    results: list[dict] = []
    for date, grp in df.groupby("date"):
        data = grp[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < min_obs:
            continue
        X = sm.add_constant(data[x_col], has_constant="add")
        try:
            fit = sm.OLS(data[y_col], X).fit()
            results.append(
                {
                    "date":  pd.Timestamp(date),
                    "c":     float(fit.params.get("const", np.nan)),   # intercept
                    "m":     float(fit.params.get(x_col, np.nan)),     # slope
                    "r2":    float(fit.rsquared),
                    "n_obs": int(fit.nobs),
                }
            )
        except Exception as exc:
            logger.debug(
                "Cross-sec OLS failed at %s for %s: %s", date, y_col, exc
            )
    return pd.DataFrame(results)


def compute_cross_sectional_predictions(
    df_security: pd.DataFrame,
    cross_sec_df: pd.DataFrame,
    x_col: str = _X_COL,
    y_col: str = "ey",
) -> pd.DataFrame:
    """
    Apply per-date cross-sectional coefficients to the primary security.

    For each date t where cross_sec_df has valid coefficients:
      x        = primary's CAGR at t
      fair_yield(t) = c(t) + m(t) * x
      premium(t)    = actual_yield(t) - fair_yield(t)

    Added columns
    -------------
    x_cagr              primary's CAGR used for fair-value calculation
    predicted_{y_col}   fair-value yield   (= c + m·x)
    premium_yield       actual - fair (yield spread)
    premium_pct         premium_yield / |fair_yield|
    actual_multiple     1 / actual_yield
    predicted_multiple  1 / fair_yield
    premium_multiple    actual_mult / fair_mult - 1
    premium_zscore      z-score of premium_yield over the full history
    """
    df = df_security.copy().sort_values("date")
    pred_col = f"predicted_{y_col}"

    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(
            "compute_cross_sectional_predictions: missing '%s' or '%s'.", x_col, y_col
        )
        return df

    if cross_sec_df.empty:
        df[pred_col] = np.nan
    else:
        cs = cross_sec_df.set_index("date")
        c_map = cs["c"].to_dict()
        m_map = cs["m"].to_dict()
        dates  = df["date"]
        alphas = dates.map(c_map)
        betas  = dates.map(m_map)
        x_vals = df[x_col]
        valid  = alphas.notna() & betas.notna() & x_vals.notna()
        df[pred_col] = np.where(valid, alphas + betas * x_vals, np.nan)

    df["x_cagr"] = df[x_col]  # record the x used for transparency

    df["premium_yield"] = df[y_col] - df[pred_col]
    df["premium_pct"] = np.where(
        df[pred_col].abs() > 1e-10,
        df["premium_yield"] / df[pred_col].abs(),
        np.nan,
    )

    df["actual_multiple"] = np.where(
        df[y_col].notna() & (df[y_col] > 1e-10), 1.0 / df[y_col], np.nan
    )
    df["predicted_multiple"] = np.where(
        df[pred_col].notna() & (df[pred_col] > 1e-10), 1.0 / df[pred_col], np.nan
    )
    df["premium_multiple"] = np.where(
        df["predicted_multiple"].notna()
        & (df["predicted_multiple"].abs() > 1e-10),
        df["actual_multiple"] / df["predicted_multiple"] - 1.0,
        np.nan,
    )

    mu  = df["premium_yield"].mean()
    sig = df["premium_yield"].std(ddof=1)
    df["premium_zscore"] = np.where(
        sig > 1e-10, (df["premium_yield"] - mu) / sig, np.nan
    )
    return df


def compute_predictions(
    df_security: pd.DataFrame,
    reg: RegressionResult,
    x_col: str = _X_COL,
    y_col: str = "ey",
) -> pd.DataFrame:
    """
    Apply regression coefficients to the primary security's CAGR history to
    produce predicted yields and premium/discount metrics.

    Added columns
    -------------
    predicted_{y_col}   alpha + beta * cagr
    premium_yield       actual - predicted
    premium_pct         premium_yield / |predicted|
    actual_multiple     1 / actual_yield    (NaN if yield ≤ 0)
    predicted_multiple  1 / predicted_yield (NaN if yield ≤ 0)
    premium_multiple    actual_mult / predicted_mult - 1
    premium_zscore      z-score of premium_yield over the full series
    """
    df = df_security.copy()

    if x_col not in df.columns or y_col not in df.columns:
        logger.warning("compute_predictions: missing '%s' or '%s'.", x_col, y_col)
        return df

    pred_col = f"predicted_{y_col}"
    df[pred_col] = reg.alpha + reg.beta * df[x_col]

    df["premium_yield"] = df[y_col] - df[pred_col]
    df["premium_pct"] = np.where(
        df[pred_col].abs() > 1e-10,
        df["premium_yield"] / df[pred_col].abs(),
        np.nan,
    )

    # Multiple-based premium
    df["actual_multiple"] = np.where(
        df[y_col].notna() & (df[y_col] > 1e-10),
        1.0 / df[y_col],
        np.nan,
    )
    df["predicted_multiple"] = np.where(
        df[pred_col].notna() & (df[pred_col] > 1e-10),
        1.0 / df[pred_col],
        np.nan,
    )
    df["premium_multiple"] = np.where(
        df["predicted_multiple"].notna() & (df["predicted_multiple"].abs() > 1e-10),
        df["actual_multiple"] / df["predicted_multiple"] - 1.0,
        np.nan,
    )

    # Z-score of premium_yield
    mu  = df["premium_yield"].mean()
    sig = df["premium_yield"].std(ddof=1)
    df["premium_zscore"] = np.where(
        sig > 1e-10, (df["premium_yield"] - mu) / sig, np.nan
    )

    return df


def _latest_row(df: pd.DataFrame, y_col: str) -> Optional[pd.Series]:
    """Return the last non-NaN row for y_col."""
    sub = df.dropna(subset=[y_col])
    return sub.iloc[-1] if not sub.empty else None


def run_all_metrics(
    basket_df: pd.DataFrame,
    security_df: pd.DataFrame,
    metrics: list[str],
    x_col: str = _X_COL,
    rolling_window: Optional[int] = None,
    min_obs: int = 10,
    cross_sec_min_obs: int = 5,
) -> dict[str, dict]:
    """
    Run the full regression pipeline for every metric.

    For each metric, two regression approaches are used:
      1. Cross-sectional OLS per date (primary) — fits basket snapshot at each t,
         produces per-date (c, m, R²) and applies them to the primary security.
      2. Pooled OLS across all dates (fallback + scatter anchor) — single α/β
         estimated from the full basket history.

    Parameters
    ----------
    basket_df         : pd.DataFrame  Basket + market data (used for regression).
    security_df       : pd.DataFrame  Primary security data (predictions applied).
    metrics           : list[str]     Metric column names.
    x_col             : str           CAGR column.
    rolling_window    : int | None    If set, also compute rolling pooled OLS.
    min_obs           : int           Min obs for pooled OLS.
    cross_sec_min_obs : int           Min tickers per date for cross-sec OLS.

    Returns
    -------
    dict  keyed by metric name, each value is:
      {
        'reg':         RegressionResult | None,   pooled OLS
        'cross_sec':   pd.DataFrame,              per-date (c, m, r2, n_obs)
        'predictions': pd.DataFrame,              primary + fair/premium cols
        'rolling':     pd.DataFrame | None,
        'summary_row': dict,
      }
    """
    output: dict[str, dict] = {}

    for metric in metrics:
        logger.info("Regression: %s ~ %s", metric, x_col)

        # ── Pooled OLS (kept for scatter reference) ───────────────────────────
        reg = run_ols(basket_df, x_col=x_col, y_col=metric, min_obs=min_obs)

        # ── Cross-sectional OLS per date (primary prediction method) ─────────
        cross_sec = run_cross_sectional_ols(
            basket_df, x_col=x_col, y_col=metric, min_obs=cross_sec_min_obs
        )

        if not cross_sec.empty:
            preds = compute_cross_sectional_predictions(
                security_df, cross_sec, x_col=x_col, y_col=metric
            )
            logger.info(
                "%s: cross-sec predictions on %d dates (mean R²=%.3f)",
                metric, len(cross_sec), cross_sec["r2"].mean(),
            )
        elif reg is not None:
            logger.info("%s: cross-sec unavailable; falling back to pooled OLS.", metric)
            preds = compute_predictions(security_df, reg, x_col=x_col, y_col=metric)
        else:
            preds = security_df.copy()

        if reg is None and cross_sec.empty:
            output[metric] = {
                "reg": None, "cross_sec": cross_sec,
                "predictions": preds, "rolling": None,
                "summary_row": {"metric": metric, "r2": np.nan},
            }
            continue

        rolling: Optional[pd.DataFrame] = None
        if rolling_window:
            rolling = run_rolling_ols(
                basket_df, x_col=x_col, y_col=metric,
                window=rolling_window, min_obs=min_obs,
            )

        latest   = _latest_row(preds, metric)
        pred_col = f"predicted_{metric}"

        # R² for summary: prefer mean cross-sec R²
        cs_r2_mean = cross_sec["r2"].mean() if not cross_sec.empty else np.nan
        summary_r2 = cs_r2_mean if not np.isnan(cs_r2_mean) else (reg.r2 if reg else np.nan)

        # Latest cross-sec coefficients
        latest_cs = cross_sec.iloc[-1] if not cross_sec.empty else None

        def _g(row, key, scale=1.0):
            if row is None or key not in row.index:
                return np.nan
            v = row[key]
            return round(float(v) * scale, 6) if pd.notna(v) else np.nan

        summary_row: dict = {
            "metric":                  metric,
            "r2":                      round(float(summary_r2), 4),
            "c_latest":                _g(latest_cs, "c"),       # intercept at latest date
            "m_latest":                _g(latest_cs, "m"),       # slope at latest date
            "x_latest":                _g(latest, x_col),        # primary CAGR at latest date
            "n_obs":                   int(latest_cs["n_obs"]) if latest_cs is not None else (reg.n_obs if reg else 0),
            "latest_actual_yield":     _g(latest, metric),
            "latest_predicted_yield":  _g(latest, pred_col),
            "latest_premium_yield":    _g(latest, "premium_yield"),
            "latest_premium_pct":      _g(latest, "premium_pct", scale=100),
            "latest_premium_zscore":   _g(latest, "premium_zscore"),
            "latest_actual_multiple":  _g(latest, "actual_multiple"),
            "latest_predicted_multiple": _g(latest, "predicted_multiple"),
            "latest_premium_multiple": _g(latest, "premium_multiple", scale=100),
        }

        output[metric] = {
            "reg":         reg,
            "cross_sec":   cross_sec,
            "predictions": preds,
            "rolling":     rolling,
            "summary_row": summary_row,
        }

    return output
