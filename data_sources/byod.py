"""
Bring-Your-Own-Data (BYOD) ingestion module.

Accepts a user-supplied CSV or Excel file with forward fundamental data
exported from Bloomberg (or any other source). Validates schema, coerces
types, derives optional columns, and flags quality issues.

── Required columns ────────────────────────────────────────────────────────────
  date               Observation date (YYYY-MM-DD or any parseable format)
  ticker             Security identifier (Bloomberg or Yahoo-compatible)
  price              Closing price on that date
  shares_outstanding Shares outstanding (same units as revenue, e.g. absolute)
  total_debt         Total financial debt (same currency as revenue)
  cash               Cash and equivalents
  fwd_eps_1y         1-year forward consensus EPS
  fwd_revenue_1y     1-year forward consensus revenue
  fwd_ebitda_1y      1-year forward consensus EBITDA
  fwd_fcf_1y         1-year forward consensus free cash flow
  fwd_revenue_4y     4-year forward consensus revenue (FY+4)

── Optional columns ────────────────────────────────────────────────────────────
  market_cap         Computed as price × shares_outstanding if absent
  enterprise_value   Computed as market_cap + total_debt - cash if absent
  currency           Informational; not used in calculations

── Bloomberg export guidance ──────────────────────────────────────────────────
  1. In BQNT / Excel Bloomberg add-in, pull BDH (historical) fields:
       PX_LAST, BS_SH_OUT, BS_TOT_LIABILITIES2, CASH_AND_NEAR_CASH_ITEM,
       BEST_EPS, BEST_SALES, BEST_EBITDA, CF_FREE_CASH_FLOW,
       BEST_SALES_4YR_FWD (or BEST_SALES + BEST_SALES_CAGR_3Y)
  2. Use monthly periodicity for the date range.
  3. Export to CSV, rename columns to the schema above.
  4. For fwd_revenue_4y: Bloomberg field BEST_SALES gives FY+1; for FY+4,
     use BEST_SALES with BEST_PERIOD_OVERRIDE = +4Y, or compute from
     BEST_SALES × (1 + BEST_SALES_GROWTH_RATE)^3.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS: list[str] = [
    "date",
    "ticker",
    "price",
    "shares_outstanding",
    "fwd_revenue_1y",
]

# All other fundamental columns are optional; metrics that need a missing column
# will produce NaN for that metric rather than failing entirely.
OPTIONAL_COLS: list[str] = [
    "total_debt",
    "cash",
    "fwd_eps_1y",
    "fwd_ebitda_1y",
    "fwd_fcf_1y",
    "fwd_revenue_4y",   # needed for 3Y fwd CAGR; falls back to TTM if absent
    "market_cap",
    "enterprise_value",
    "currency",
]

_NUMERIC_COLS: list[str] = [
    c for c in REQUIRED_COLS + OPTIONAL_COLS
    if c not in ("date", "ticker", "currency")
]


def load_byod(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate a BYOD file (.csv or .xlsx).

    Parameters
    ----------
    path : str | Path
        Path to the BYOD file.

    Returns
    -------
    pd.DataFrame
        Validated, normalised panel.  Index is a RangeIndex.
        (date, ticker) pairs are unique.

    Raises
    ------
    FileNotFoundError  – if the file does not exist.
    ValueError         – if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BYOD file not found: {path}")

    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .xlsx.")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Required-column check ────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"BYOD file is missing required columns: {missing}\n"
            f"See config/byod_template.csv for the expected schema."
        )

    # ── Type coercion ────────────────────────────────────────────────────────
    df["date"]   = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with unparseable dates
    bad_dates = df["date"].isna()
    if bad_dates.any():
        logger.warning("Dropped %d rows with unparseable dates.", bad_dates.sum())
        df = df[~bad_dates]

    # ── Derive optional columns if absent ────────────────────────────────────
    if "market_cap" not in df.columns:
        df["market_cap"] = df["price"] * df["shares_outstanding"]

    if "enterprise_value" not in df.columns:
        # total_debt and cash are optional; treat as 0 if absent
        td = df["total_debt"].fillna(0) if "total_debt" in df.columns else 0
        ca = df["cash"].fillna(0) if "cash" in df.columns else 0
        df["enterprise_value"] = df["market_cap"] + td - ca

    # ── Guard: non-positive EV / price ───────────────────────────────────────
    bad_ev = df["enterprise_value"] <= 0
    if bad_ev.any():
        logger.warning(
            "%d rows have enterprise_value ≤ 0; EV-based metrics → NaN.",
            bad_ev.sum(),
        )
        df.loc[bad_ev, "enterprise_value"] = np.nan

    bad_px = df["price"] <= 0
    if bad_px.any():
        logger.warning("%d rows have price ≤ 0; EPS yield → NaN.", bad_px.sum())
        df.loc[bad_px, "price"] = np.nan

    # ── Deduplication ────────────────────────────────────────────────────────
    dupes = df.duplicated(subset=["date", "ticker"])
    if dupes.any():
        logger.warning(
            "%d duplicate (date, ticker) rows; keeping first.", dupes.sum()
        )
        df = df[~dupes].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(
        "BYOD loaded: %d rows | %d tickers | %s – %s",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def validate_byod(df: pd.DataFrame) -> list[str]:
    """
    Self-check a loaded BYOD DataFrame.

    Returns
    -------
    list[str]
        Warning messages; empty list means no issues detected.
        Only REQUIRED_COLS trigger NaN warnings.  Optional columns that are
        absent or all-NaN are silently skipped (they simply won't power the
        corresponding metric).
    """
    warnings: list[str] = []

    # Required columns: flag if missing or >50% NaN
    for col in REQUIRED_COLS:
        if col not in df.columns:
            warnings.append(f"MISSING required column: '{col}'")
            continue
        if col in _NUMERIC_COLS:
            pct_null = df[col].isna().mean() * 100
            if pct_null > 50:
                warnings.append(
                    f"Required column '{col}' is {pct_null:.1f}% NaN – "
                    "check your export."
                )

    # Optional columns: note which are present (informational, not a warning)
    present_optional = [
        c for c in OPTIONAL_COLS
        if c in df.columns and c not in ("market_cap", "enterprise_value", "currency")
        and df[c].notna().any()
    ]
    if present_optional:
        warnings.append(
            f"ℹ Optional columns found (will be used where available): "
            f"{', '.join(present_optional)}"
        )

    for col in ["fwd_revenue_1y", "fwd_revenue_4y"]:
        if col in df.columns:
            n_neg = (df[col].dropna() < 0).sum()
            if n_neg:
                warnings.append(f"{n_neg} negative values in '{col}'.")

    if "fwd_revenue_4y" in df.columns and "fwd_revenue_1y" in df.columns:
        ratio = df["fwd_revenue_4y"] / df["fwd_revenue_1y"]
        extreme = (ratio > 10) | (ratio < 0.1)
        n_ext = extreme.sum()
        if n_ext:
            warnings.append(
                f"{n_ext} rows have fwd_revenue_4y / fwd_revenue_1y outside [0.1, 10] "
                f"– check units or outliers."
            )

    return warnings
