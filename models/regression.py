"""
OLS regression of valuation yield on 3-year forward CAGR.

Model
-----
  yield_metric = alpha + beta * fwd_cagr_3y  + ε

The regression is estimated on the *basket* (and/or market) panel and then
used to *predict* the primary security's fair-value yield given its CAGR.

Outputs per metric
------------------
  RegressionResult   – full-sample OLS coefficients + diagnostics
  rolling_ols()      – time-varying alpha / beta / R² (rolling window)
  compute_predictions() – predicted yield + premium/discount for a security
  run_all_metrics()  – orchestrates the above for a list of metrics

Premium / discount definitions
-------------------------------
  premium_yield    = actual_yield   - predicted_yield    (absolute yield spread)
  premium_pct      = premium_yield  / |predicted_yield|  (percentage)
  premium_multiple = actual_mult    / predicted_mult - 1  (multiple vs predicted)
  premium_zscore   = z-score of premium_yield over the history
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
) -> dict[str, dict]:
    """
    Run the full regression pipeline for every metric.

    Parameters
    ----------
    basket_df     : pd.DataFrame  Basket + market data (used for regression).
    security_df   : pd.DataFrame  Primary security data (predictions applied here).
    metrics       : list[str]     Metric column names (e.g. ['ey', 'ebitda_yld']).
    x_col         : str           CAGR column.
    rolling_window: int | None    If set, also compute rolling OLS.
    min_obs       : int           Minimum obs for a regression to be valid.

    Returns
    -------
    dict  keyed by metric name, each value is:
      {
        'reg':         RegressionResult | None,
        'predictions': pd.DataFrame   (security + predictions),
        'rolling':     pd.DataFrame | None,
        'summary_row': dict,
      }
    """
    output: dict[str, dict] = {}

    for metric in metrics:
        logger.info("Regression: %s ~ %s", metric, x_col)

        reg = run_ols(basket_df, x_col=x_col, y_col=metric, min_obs=min_obs)
        if reg is None:
            output[metric] = {
                "reg": None, "predictions": security_df.copy(),
                "rolling": None, "summary_row": {"metric": metric, "r2": np.nan},
            }
            continue

        preds = compute_predictions(security_df, reg, x_col=x_col, y_col=metric)

        rolling: Optional[pd.DataFrame] = None
        if rolling_window:
            rolling = run_rolling_ols(
                basket_df, x_col=x_col, y_col=metric,
                window=rolling_window, min_obs=min_obs,
            )

        latest = _latest_row(preds, metric)
        pred_col = f"predicted_{metric}"

        def _g(row, key, scale=1.0):
            """Safe getter with scaling."""
            if row is None or key not in row.index:
                return np.nan
            v = row[key]
            return round(float(v) * scale, 6) if pd.notna(v) else np.nan

        summary_row: dict = {
            "metric":                  metric,
            "r2":                      round(reg.r2, 4),
            "alpha":                   round(reg.alpha, 6),
            "beta":                    round(reg.beta, 4),
            "n_obs":                   reg.n_obs,
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
            "predictions": preds,
            "rolling":     rolling,
            "summary_row": summary_row,
        }

    return output
