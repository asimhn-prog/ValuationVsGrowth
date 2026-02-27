"""
Valuation vs Growth Explorer – Streamlit front-end.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup (allow running from repo root or app/ directory) ──────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.assembler import build_panel
from data_sources.byod import load_byod, validate_byod, REQUIRED_COLS as BYOD_REQUIRED_COLS
from transforms.compute_ev import ensure_ev
from transforms.compute_forward_metrics import (
    ALL_METRICS,
    MULTIPLE_LABELS,
    YIELD_METRIC_LABELS,
    compute_valuation_yields,
)
from transforms.compute_cagr import compute_forward_cagr
from transforms.winsor import winsorise_panel, get_winsor_bounds
from models.regression import run_all_metrics, RegressionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG_DIR = ROOT / "config"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Valuation vs Growth",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>div.stButton > button {width:100%;}</style>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data …", ttl=3600)
def load_panel(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    byod_bytes: bytes | None,
    byod_filename: str | None,
    freq: str,
) -> pd.DataFrame:
    """Cached panel build (re-runs only when inputs change)."""
    byod_path = None
    if byod_bytes is not None:
        # Preserve the original file extension so load_byod picks the right reader
        suffix = Path(byod_filename).suffix.lower() if byod_filename else ".csv"
        if suffix not in {".csv", ".xlsx", ".xls"}:
            suffix = ".csv"
        tmp = OUTPUT_DIR / f"_byod_upload{suffix}"
        tmp.write_bytes(byod_bytes)
        byod_path = str(tmp)
    return build_panel(list(tickers), start, end, byod_path=byod_path, freq=freq)


def run_pipeline(
    panel: pd.DataFrame,
    fcf_use_ev: bool,
    winsor_cfg: dict,
) -> pd.DataFrame:
    """Compute EV, yields, CAGR, and optionally winsorise."""
    panel = ensure_ev(panel)
    panel = compute_valuation_yields(panel, fcf_use_ev=fcf_use_ev)
    panel = compute_forward_cagr(panel)

    if winsor_cfg.get("enabled"):
        cols = winsor_cfg.get("apply_to", ALL_METRICS + ["fwd_cagr_3y"])
        cols = [c for c in cols if c in panel.columns]
        panel = winsorise_panel(
            panel,
            columns=cols,
            method=winsor_cfg.get("method", "percentile"),
            p_low=winsor_cfg.get("p_low", 0.025),
            p_high=winsor_cfg.get("p_high", 0.025),
            n_std=winsor_cfg.get("n_std", 2.5),
        )
    return panel


def tier_label(tier: int | float) -> str:
    return "▲ Forward (Tier 1)" if tier == 1 else "▽ Trailing (Tier 3)" if tier == 3 else "–"


def fmt_pct(v) -> str:
    if pd.isna(v):
        return "–"
    return f"{v:+.1f}%"


def fmt_x(v) -> str:
    if pd.isna(v):
        return "–"
    return f"{v:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "security": "#2563EB",   # blue
    "basket":   "#16A34A",   # green
    "market":   "#9333EA",   # purple
    "regression": "#EF4444", # red
    "current":  "#F59E0B",   # amber
}


def scatter_plot(
    panel: pd.DataFrame,
    primary: str,
    basket: list[str],
    market: str,
    metric: str,
    show_regression: bool = True,
    reg_result=None,
) -> go.Figure:
    """
    Snapshot scatter: one point per ticker at its latest available date.
    X = fwd_cagr_3y, Y = metric yield.  Ticker labels shown on chart.
    """
    fig = go.Figure()
    x_col = "fwd_cagr_3y"
    basket_tickers = [t for t in basket if t != primary]

    def _add_group(tickers, color, symbol, size, group_name):
        xs, ys, labels, hover = [], [], [], []
        for ticker in tickers:
            sub = panel[panel["ticker"] == ticker].dropna(subset=[x_col, metric])
            if sub.empty:
                continue
            row = sub.iloc[-1]
            xs.append(row[x_col] * 100)
            ys.append(row[metric] * 100)
            labels.append(ticker)
            hover.append(row["date"].strftime("%Y-%m-%d"))
        if not xs:
            return
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                name=group_name,
                text=labels,
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(color=color, size=size, symbol=symbol),
                customdata=list(zip(labels, hover)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Date: %{customdata[1]}<br>"
                    f"Rev. CAGR p.a.: %{{x:.2f}}%<br>"
                    f"{YIELD_METRIC_LABELS.get(metric, metric)}: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    _add_group(basket_tickers,  _COLORS["basket"],   "diamond", 10, "Basket")
    _add_group([market],        _COLORS["market"],   "square",  10, f"Market ({market})")
    _add_group([primary],       _COLORS["security"], "star",    16, primary)

    # Regression line (fitted on full basket history)
    if show_regression and reg_result is not None:
        ref_data = panel[
            panel["ticker"].isin(basket_tickers + [market])
        ].dropna(subset=[x_col, metric])
        if not ref_data.empty:
            x_range = np.linspace(ref_data[x_col].min(), ref_data[x_col].max(), 100)
            y_line  = reg_result.alpha + reg_result.beta * x_range
            fig.add_trace(
                go.Scatter(
                    x=x_range * 100, y=y_line * 100,
                    mode="lines",
                    name=f"OLS fit (R²={reg_result.r2:.2f})",
                    line=dict(color=_COLORS["regression"], width=2, dash="dash"),
                    hovertemplate="Rev. CAGR p.a.: %{x:.2f}%<br>Fitted yield: %{y:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"{YIELD_METRIC_LABELS.get(metric, metric)} vs Revenue CAGR p.a. — Current Snapshot",
            font=dict(size=16),
        ),
        xaxis=dict(title="Revenue CAGR, % p.a.", ticksuffix="%", gridcolor="#e5e7eb"),
        yaxis=dict(title=f"{YIELD_METRIC_LABELS.get(metric, metric)} (%)", ticksuffix="%", gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=-0.2),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
    )
    return fig


def basket_timeseries_plot(
    panel: pd.DataFrame,
    primary: str,
    basket: list[str],
    market: str,
    metric: str,
    basket_agg: str = "mean",
) -> go.Figure:
    """
    Time-series of basket-average yield (mean or median) and primary security
    yield over the full history.
    """
    fig = go.Figure()
    basket_tickers = [t for t in basket if t != primary]

    # Basket aggregate over time
    basket_data = panel[panel["ticker"].isin(basket_tickers)]
    if not basket_data.empty and metric in basket_data.columns:
        grp = basket_data.groupby("date")[metric]
        agg_series = grp.median() if basket_agg == "median" else grp.mean()
        agg = agg_series.dropna().reset_index()
        agg.columns = ["date", metric]
        if not agg.empty:
            fig.add_trace(
                go.Scatter(
                    x=agg["date"],
                    y=agg[metric] * 100,
                    mode="lines",
                    name=f"Basket {basket_agg.capitalize()}",
                    line=dict(color=_COLORS["basket"], width=2),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        f"Basket {basket_agg}: %{{y:.2f}}%<extra></extra>"
                    ),
                )
            )

    # Primary security
    sec = panel[panel["ticker"] == primary].dropna(subset=[metric])
    if not sec.empty:
        fig.add_trace(
            go.Scatter(
                x=sec["date"],
                y=sec[metric] * 100,
                mode="lines",
                name=primary,
                line=dict(color=_COLORS["security"], width=2),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"{primary}: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    # Market proxy
    mkt = panel[panel["ticker"] == market].dropna(subset=[metric])
    if not mkt.empty:
        fig.add_trace(
            go.Scatter(
                x=mkt["date"],
                y=mkt[metric] * 100,
                mode="lines",
                name=f"Market ({market})",
                line=dict(color=_COLORS["market"], width=1.5, dash="dot"),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"Market: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{YIELD_METRIC_LABELS.get(metric, metric)} Over Time",
            font=dict(size=16),
        ),
        xaxis=dict(title="Date", gridcolor="#e5e7eb"),
        yaxis=dict(
            title=f"{YIELD_METRIC_LABELS.get(metric, metric)} (%)",
            ticksuffix="%",
            gridcolor="#e5e7eb",
        ),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
    )
    return fig


def premium_timeseries_plot(
    preds_df: pd.DataFrame,
    primary: str,
    metric: str,
    cross_sec_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Two-row chart (shared x-axis):
      Row 1 – Left axis:  Observed yield & Fair yield (regression line)
               Right axis: Spread = observed − fair  (shaded + line)
      Row 2 – Cross-sectional R² per period
    """
    metric_label = YIELD_METRIC_LABELS.get(metric, metric)
    pred_col     = f"predicted_{metric}"
    has_r2       = (
        cross_sec_df is not None
        and not cross_sec_df.empty
        and "r2" in cross_sec_df.columns
    )

    fig = make_subplots(
        rows=2 if has_r2 else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.65, 0.35] if has_r2 else [1.0],
        specs=(
            [[{"secondary_y": True}], [{"secondary_y": False}]]
            if has_r2
            else [[{"secondary_y": True}]]
        ),
        subplot_titles=(
            [f"{primary} — {metric_label} | Fair vs Observed", "Cross-sectional R²"]
            if has_r2
            else [f"{primary} — {metric_label} | Fair vs Observed"]
        ),
    )

    data = preds_df.dropna(subset=["date"])

    # ── Row 1, left axis: Observed yield ─────────────────────────────────────
    if metric in data.columns:
        obs = data.dropna(subset=[metric])
        fig.add_trace(
            go.Scatter(
                x=obs["date"], y=obs[metric] * 100,
                mode="lines", name="Observed yield",
                line=dict(color=_COLORS["security"], width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Observed: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    # ── Row 1, left axis: Fair yield ─────────────────────────────────────────
    if pred_col in data.columns:
        fair = data.dropna(subset=[pred_col])
        fig.add_trace(
            go.Scatter(
                x=fair["date"], y=fair[pred_col] * 100,
                mode="lines", name="Fair yield (cross-sec regression)",
                line=dict(color=_COLORS["regression"], width=2, dash="dash"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Fair: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    # ── Row 1, right axis: Spread (observed − fair) ───────────────────────────
    if "premium_yield" in data.columns:
        prem = data.dropna(subset=["premium_yield"])
        pos  = prem["premium_yield"].clip(lower=0) * 100
        neg  = prem["premium_yield"].clip(upper=0) * 100

        fig.add_trace(
            go.Scatter(
                x=prem["date"], y=pos, fill="tozeroy",
                fillcolor="rgba(239,68,68,0.12)", line=dict(width=0),
                name="Rich (obs > fair)", showlegend=True, hoverinfo="skip",
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=prem["date"], y=neg, fill="tozeroy",
                fillcolor="rgba(22,163,74,0.12)", line=dict(width=0),
                name="Cheap (obs < fair)", showlegend=True, hoverinfo="skip",
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=prem["date"], y=prem["premium_yield"] * 100,
                mode="lines", name="Spread (obs − fair)",
                line=dict(color="#6B7280", width=1.5),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Spread: %{y:+.2f}%<extra></extra>"
                ),
            ),
            row=1, col=1, secondary_y=True,
        )

    # ── Row 2: R² per period ──────────────────────────────────────────────────
    if has_r2:
        fig.add_trace(
            go.Scatter(
                x=cross_sec_df["date"], y=cross_sec_df["r2"],
                mode="lines+markers", name="R² (cross-sec)",
                line=dict(color=_COLORS["basket"], width=2),
                marker=dict(size=5),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>R²: %{y:.3f}"
                    "<br>n=%{customdata}<extra></extra>"
                ),
                customdata=cross_sec_df["n_obs"],
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=680 if has_r2 else 480,
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_yaxes(
        title_text=f"{metric_label} (%)",
        ticksuffix="%",
        gridcolor="#e5e7eb",
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Spread (%)",
        ticksuffix="%",
        gridcolor=None,
        showgrid=False,
        zeroline=True, zerolinecolor="#9ca3af", zerolinewidth=1,
        row=1, col=1, secondary_y=True,
    )
    if has_r2:
        fig.update_yaxes(
            title_text="R²",
            range=[0, 1],
            gridcolor="#e5e7eb",
            row=2, col=1,
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def basket_gap_timeseries(
    basket_panel: pd.DataFrame,
    preds_df: pd.DataFrame,
    cross_sec_df: pd.DataFrame,
    metric: str,
    primary: str,
) -> go.Figure:
    """
    Time-series of the gap (actual multiple − predicted multiple) for every
    basket ticker, plus the primary security.

    For each date the cross-sectional coefficients are applied to every basket
    ticker to get its 'fair' multiple; the gap is actual − fair.

    • Shaded band = 25th–75th percentile across basket tickers each month.
    • Green line   = basket mean gap.
    • Blue line    = primary security gap.
    • Dashed zero  = fairly-valued level.
    """
    fig = go.Figure()
    x_col = "fwd_cagr_3y"
    mult_label = MULTIPLE_LABELS.get(metric, metric)

    # ── Basket gaps ────────────────────────────────────────────────────────────
    if cross_sec_df is not None and not cross_sec_df.empty:
        cs = cross_sec_df.set_index("date")[["c", "m"]]

        work = basket_panel[["date", "ticker", metric, x_col]].copy()
        work = work.join(cs, on="date")          # adds c and m per date
        work["fair_yield"] = work["c"] + work["m"] * work[x_col]

        # Keep only rows where both actual and fair yields are positive
        valid = (
            work[metric].notna() & (work[metric] > 0) &
            work["fair_yield"].notna() & (work["fair_yield"] > 0) &
            work[x_col].notna()
        )
        work = work[valid].copy()
        work["gap"] = 1.0 / work[metric] - 1.0 / work["fair_yield"]

        agg = (
            work.groupby("date")["gap"]
            .agg(
                mean="mean",
                p25=lambda s: s.quantile(0.25),
                p75=lambda s: s.quantile(0.75),
            )
            .reset_index()
        )

        if not agg.empty:
            # Shaded IQR band
            fig.add_trace(go.Scatter(
                x=pd.concat([agg["date"], agg["date"].iloc[::-1]]),
                y=pd.concat([agg["p75"], agg["p25"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(22, 163, 74, 0.12)",
                line=dict(color="rgba(22, 163, 74, 0)"),
                name="Basket IQR (25th–75th %ile)",
                hoverinfo="skip",
            ))
            # Mean basket gap
            fig.add_trace(go.Scatter(
                x=agg["date"],
                y=agg["mean"].round(2),
                mode="lines",
                name="Basket mean gap",
                line=dict(color=_COLORS["basket"], width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Basket mean gap: %{y:.1f}x<extra></extra>",
            ))

    # ── Primary security gap ───────────────────────────────────────────────────
    if (
        preds_df is not None
        and "actual_multiple" in preds_df.columns
        and "predicted_multiple" in preds_df.columns
    ):
        prim = preds_df.dropna(subset=["actual_multiple", "predicted_multiple"]).copy()
        prim["gap"] = prim["actual_multiple"] - prim["predicted_multiple"]
        if not prim.empty:
            fig.add_trace(go.Scatter(
                x=prim["date"],
                y=prim["gap"].round(2),
                mode="lines",
                name=f"{primary}",
                line=dict(color=_COLORS["security"], width=2.5),
                hovertemplate=(
                    f"Date: %{{x|%Y-%m-%d}}<br>"
                    f"{primary} gap: %{{y:.1f}}x<extra></extra>"
                ),
            ))

    # Zero reference
    fig.add_hline(y=0, line=dict(color="#6b7280", dash="dash", width=1))

    fig.update_layout(
        title=dict(
            text=f"Multiple Gap: Actual vs Predicted {mult_label}",
            font=dict(size=15),
        ),
        xaxis=dict(title="Date", gridcolor="#e5e7eb"),
        yaxis=dict(
            title=f"Actual − Predicted {mult_label} (turns)",
            gridcolor="#e5e7eb",
            zeroline=False,
        ),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📈 Valuation vs Growth Explorer")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Universe")

        primary = st.text_input("Primary ticker", value="AAPL").upper().strip()

        st.subheader("Basket")
        basket_raw = st.text_area(
            "Basket tickers (comma-separated)",
            value="MSFT, GOOGL, META, AMZN, NVDA",
            height=80,
        )
        basket_list = [t.strip().upper() for t in basket_raw.split(",") if t.strip()]

        # Basket save / load
        c1, c2 = st.columns(2)
        if c1.button("💾 Save basket"):
            cfg = {"primary": primary, "basket": basket_list}
            (CONFIG_DIR / "saved_basket.json").write_text(json.dumps(cfg, indent=2))
            st.success("Saved to config/saved_basket.json")
        if c2.button("📂 Load basket"):
            fp = CONFIG_DIR / "saved_basket.json"
            if fp.exists():
                cfg = json.loads(fp.read_text())
                st.session_state["primary"]     = cfg.get("primary", primary)
                st.session_state["basket_raw"]  = ", ".join(cfg.get("basket", []))
                st.rerun()
            else:
                st.warning("No saved basket found.")

        market = st.text_input("Market proxy", value="SPY").upper().strip()

        st.divider()
        st.header("Date & Frequency")
        start_date = st.date_input("Start date", value=pd.Timestamp("2021-01-01"))
        end_date   = st.date_input("End date",   value=pd.Timestamp.today().normalize())
        freq_label = st.selectbox("Frequency", ["Monthly", "Quarterly", "Daily"])
        freq = {"Monthly": "ME", "Quarterly": "QE", "Daily": "B"}[freq_label]
        if freq_label == "Daily":
            st.caption(
                "Daily loads more data points. Fundamental metrics step at quarterly "
                "intervals between earnings updates."
            )

        st.divider()
        st.header("BYOD Data")
        st.markdown(
            "_Upload a Bloomberg CSV export. See `config/byod_template.csv` for schema._"
        )
        byod_file  = st.file_uploader("Upload BYOD CSV / Excel", type=["csv", "xlsx"])
        byod_bytes = None

        if byod_file is not None:
            byod_bytes = byod_file.read()
            # ── Validate immediately and surface any issues ───────────────────
            try:
                _suffix = Path(byod_file.name).suffix
                _tmp    = OUTPUT_DIR / f"_byod_check{_suffix}"
                _tmp.write_bytes(byod_bytes)
                _preview = load_byod(_tmp)
                _issues  = validate_byod(_preview)
                _tickers = sorted(_preview["ticker"].unique())
                st.success(
                    f"✓ BYOD loaded — {len(_tickers)} ticker(s), "
                    f"{len(_preview)} rows  |  "
                    f"{_preview['date'].min().date()} → {_preview['date'].max().date()}"
                )
                # Cross-reference BYOD tickers against sidebar universe
                _byod_set     = set(_tickers)
                _sidebar_set  = set([primary] + basket_list + [market])
                _matched      = sorted(_byod_set & _sidebar_set)
                _not_in_byod  = sorted(_sidebar_set - _byod_set)
                _extra_byod   = sorted(_byod_set - _sidebar_set)
                if _matched:
                    st.caption(f"✓ Matched to sidebar (→ Tier 1): {', '.join(_matched)}")
                if _not_in_byod:
                    st.warning(
                        f"⚠ Sidebar tickers **not** found in BYOD: "
                        f"{', '.join(_not_in_byod)}  \n"
                        "These will use Tier 3 (yfinance trailing). "
                        "Check that ticker names match exactly (e.g. `AAPL`, not `AAPL US`)."
                    )
                if _extra_byod:
                    st.caption(f"ℹ BYOD tickers not in current sidebar: {', '.join(_extra_byod)}")
                for _w in _issues:
                    if _w.startswith("ℹ"):
                        st.caption(_w)
                    else:
                        st.warning(f"⚠ {_w}")
            except Exception as _e:
                st.error(f"❌ BYOD file error: {_e}")
                st.info(
                    f"Required columns: **{', '.join(BYOD_REQUIRED_COLS)}**  \n"
                    "All other fundamental columns are optional. "
                    "Column names are case-insensitive and spaces are converted to underscores."
                )
                byod_bytes = None   # don't pass bad file downstream

        st.divider()
        st.header("Settings")
        fcf_use_ev = st.toggle("FCF yield denominator = EV (else Mkt Cap)", value=True)

        metric = st.selectbox(
            "Valuation metric",
            options=ALL_METRICS,
            format_func=lambda m: YIELD_METRIC_LABELS.get(m, m),
        )

        basket_agg = st.selectbox(
            "Basket time-series aggregation", ["mean", "median"],
            format_func=lambda v: {"mean": "Mean per date", "median": "Median per date"}[v],
        )

        show_reg     = st.toggle("Show regression line",  value=True)
        rolling_win  = st.number_input("Rolling OLS window (months; 0 = full sample)", min_value=0, max_value=120, value=0, step=6)
        rolling_win  = int(rolling_win) if rolling_win > 0 else None

        st.divider()
        st.subheader("Winsorisation")
        winsor_on = st.toggle("Enable winsorisation", value=False)
        w_method  = st.selectbox("Method", ["percentile", "stddev"], disabled=not winsor_on)
        w_p_low   = st.slider("Lower percentile", 0.0, 0.10, 0.025, step=0.005, disabled=not winsor_on or w_method == "stddev")
        w_p_high  = st.slider("Upper percentile", 0.0, 0.10, 0.025, step=0.005, disabled=not winsor_on or w_method == "stddev")
        w_nstd    = st.slider("Std-dev clip (σ)", 1.0, 4.0, 2.5, step=0.25, disabled=not winsor_on or w_method == "percentile")

        winsor_cfg = {
            "enabled": winsor_on,
            "method":  w_method,
            "p_low":   w_p_low,
            "p_high":  w_p_high,
            "n_std":   w_nstd,
            "apply_to": ALL_METRICS + ["fwd_cagr_3y"],
        }

        st.divider()
        run_btn = st.button("🚀 Run Analysis", type="primary")

    # ── Main content ──────────────────────────────────────────────────────────
    if not run_btn:
        st.info(
            "Configure your universe and settings in the sidebar, then click **Run Analysis**."
        )
        st.markdown(
            "**Quick start** – no BYOD needed:\n"
            "1. Enter a primary ticker (e.g. `AAPL`).\n"
            "2. Enter basket tickers.\n"
            "3. Click Run.  \n\n"
            "*Note: without a BYOD file, all metrics use **trailing (TTM)** data (Tier 3). "
            "For NTM forward estimates, upload a Bloomberg CSV export.*"
        )
        return

    all_tickers = list(dict.fromkeys([primary] + basket_list + [market]))

    with st.spinner("Loading data …"):
        try:
            panel_raw = load_panel(
                tuple(all_tickers),
                str(start_date),
                str(end_date),
                byod_bytes,
                byod_file.name if byod_file is not None else None,
                freq,
            )
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            logger.exception("Panel load error")
            return

    if panel_raw.empty:
        st.error("No data returned. Check your tickers and date range.")
        return

    panel = run_pipeline(panel_raw, fcf_use_ev=fcf_use_ev, winsor_cfg=winsor_cfg)

    # Tier info banner
    tiers = panel.drop_duplicates("ticker")[["ticker", "data_tier"]]
    tier1_tickers = [r.ticker for _, r in tiers.iterrows() if r.data_tier == 1]
    tier3_tickers = [r.ticker for _, r in tiers.iterrows() if r.data_tier == 3]
    if tier1_tickers:
        st.success(f"✓ Tier 1 — BYOD forward estimates: **{', '.join(tier1_tickers)}**")
    if tier3_tickers:
        st.info(f"▽ Tier 3 — yfinance trailing: **{', '.join(tier3_tickers)}**")

    # ── Data quality diagnostic ───────────────────────────────────────────────
    with st.expander("🔍 Data diagnostic (click to inspect if charts are blank)"):
        diag_cols = ["fwd_cagr_3y"] + ALL_METRICS
        rows_diag = []
        for tkr in all_tickers:
            sub = panel[panel["ticker"] == tkr]
            tier_val = sub["data_tier"].iloc[0] if not sub.empty else "?"
            row_d = {"Ticker": tkr, "Tier": int(tier_val) if not pd.isna(tier_val) else "?",
                     "Rows": len(sub)}
            for col in diag_cols:
                if col in sub.columns:
                    nn = sub[col].notna().sum()
                    latest = sub[col].dropna().iloc[-1] if nn > 0 else None
                    row_d[f"{col} (non-NaN)"] = nn
                    row_d[f"{col} (latest)"] = f"{latest*100:.2f}%" if latest is not None else "NaN"
                else:
                    row_d[f"{col} (non-NaN)"] = "col missing"
                    row_d[f"{col} (latest)"] = "col missing"
            rows_diag.append(row_d)
        st.dataframe(pd.DataFrame(rows_diag), use_container_width=True)

        # Intermediate column coverage per ticker
        inter_cols = ["shares_outstanding", "market_cap", "enterprise_value", "ttm_revenue"]
        rows_inter = []
        for tkr in all_tickers:
            sub = panel_raw[panel_raw["ticker"] == tkr]
            r = {"Ticker": tkr}
            for col in inter_cols:
                if col in sub.columns:
                    nn = sub[col].notna().sum()
                    r[f"{col} (non-NaN)"] = f"{nn}/{len(sub)}"
                else:
                    r[f"{col} (non-NaN)"] = "missing"
            rows_inter.append(r)
        st.caption("**Intermediate columns (raw panel before transforms):**")
        st.dataframe(pd.DataFrame(rows_inter), use_container_width=True)

        fwd_cols = [c for c in panel.columns if c.startswith("fwd_")]
        ttm_cols = [c for c in panel.columns if c.startswith("ttm_")]
        st.caption(f"fwd_* columns: {fwd_cols}")
        st.caption(f"ttm_* columns: {ttm_cols}")
        st.caption(f"enterprise_value non-NaN: {panel['enterprise_value'].notna().sum()} / {len(panel)}")

    # ── Regression ────────────────────────────────────────────────────────────
    basket_and_market = [t for t in all_tickers if t != primary]
    basket_panel   = panel[panel["ticker"].isin(basket_and_market)]
    security_panel = panel[panel["ticker"] == primary].copy()

    with st.spinner("Running regressions …"):
        results = run_all_metrics(
            basket_df=basket_panel,
            security_df=security_panel,
            metrics=ALL_METRICS,
            rolling_window=rolling_win,
            min_obs=8,
        )

    if not results:
        st.warning("Not enough data for regression. Try a wider date range or smaller min_obs.")
        return

    reg_result   = results.get(metric, {}).get("reg")
    cross_sec_df = results.get(metric, {}).get("cross_sec", pd.DataFrame())
    preds_df     = results.get(metric, {}).get("predictions", security_panel)

    # Build a RegressionResult from the latest cross-sec snapshot for the scatter
    scatter_reg = reg_result  # fallback to pooled OLS
    if cross_sec_df is not None and not cross_sec_df.empty:
        cs_latest = cross_sec_df.iloc[-1]
        scatter_reg = RegressionResult(
            metric=metric, x_col="fwd_cagr_3y",
            alpha=float(cs_latest["c"]),
            beta=float(cs_latest["m"]),
            r2=float(cs_latest["r2"]),
            n_obs=int(cs_latest["n_obs"]),
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Scatter", "📉 Premium/Discount", "📋 Summary Table", "💾 Export"]
    )

    with tab1:
        st.subheader(f"{YIELD_METRIC_LABELS.get(metric, metric)} — Current Snapshot")
        fig_scatter = scatter_plot(
            panel, primary, basket_list + [market], market,
            metric, show_regression=show_reg, reg_result=scatter_reg,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        if scatter_reg:
            src = "latest cross-sec" if (cross_sec_df is not None and not cross_sec_df.empty) else "pooled OLS"
            cols = st.columns(5)
            cols[0].metric("R² (latest)", f"{scatter_reg.r2:.3f}")
            cols[1].metric("c (intercept)", f"{scatter_reg.alpha:.4f}")
            cols[2].metric("m (slope)",     f"{scatter_reg.beta:.4f}")
            cols[3].metric("n (tickers)",   str(scatter_reg.n_obs))
            if cross_sec_df is not None and not cross_sec_df.empty:
                cols[4].metric("Mean R² (all periods)", f"{cross_sec_df['r2'].mean():.3f}")
            st.caption(f"Regression source: {src}")

        st.subheader(f"{YIELD_METRIC_LABELS.get(metric, metric)} — History")
        fig_ts = basket_timeseries_plot(
            panel, primary, basket_list, market, metric, basket_agg=basket_agg,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Rolling R² chart
        rolling_df = results.get(metric, {}).get("rolling")
        if rolling_df is not None and not rolling_df.empty:
            with st.expander("📈 Rolling OLS over time"):
                fig_roll = go.Figure()
                fig_roll.add_trace(
                    go.Scatter(
                        x=rolling_df["date"], y=rolling_df["r2"],
                        mode="lines", name="Rolling R²",
                        line=dict(color=_COLORS["security"], width=2),
                    )
                )
                fig_roll.update_layout(
                    title="Rolling R²", height=300,
                    xaxis_title="Date", yaxis_title="R²",
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_roll, use_container_width=True)

    with tab2:
        st.subheader(f"{primary} — Fair vs Observed | {YIELD_METRIC_LABELS.get(metric, metric)}")
        if preds_df is not None and "premium_yield" in preds_df.columns:
            fig_prem = premium_timeseries_plot(
                preds_df, primary, metric,
                cross_sec_df=cross_sec_df if cross_sec_df is not None and not cross_sec_df.empty else None,
            )
            st.plotly_chart(fig_prem, use_container_width=True)

            # Latest premium table
            latest = preds_df.dropna(subset=["premium_yield"])
            if not latest.empty:
                cur = latest.iloc[-1]
                st.markdown(f"**Latest ({cur['date'].strftime('%Y-%m-%d')})**")
                _c = st.columns(5)
                _c[0].metric("Actual yield",    f"{cur[metric]*100:.2f}%"          if pd.notna(cur.get(metric)) else "–")
                _c[1].metric("Predicted yield", f"{cur.get(f'predicted_{metric}',np.nan)*100:.2f}%" if pd.notna(cur.get(f"predicted_{metric}")) else "–")
                _c[2].metric("Premium (yield)", f"{cur['premium_yield']*100:+.2f}%" if pd.notna(cur["premium_yield"]) else "–")
                _c[3].metric("Premium (%)",     f"{cur.get('premium_pct',np.nan)*100:+.1f}%"  if pd.notna(cur.get("premium_pct")) else "–")
                _c[4].metric("Z-score",         f"{cur.get('premium_zscore',np.nan):+.2f}"    if pd.notna(cur.get("premium_zscore")) else "–")
        else:
            st.info("No regression predictions available for this metric.")

        st.divider()
        st.subheader(f"Basket — Average Multiple Gap | {YIELD_METRIC_LABELS.get(metric, metric)}")
        fig_gap = basket_gap_timeseries(
            basket_panel=basket_panel,
            preds_df=preds_df,
            cross_sec_df=cross_sec_df if cross_sec_df is not None and not cross_sec_df.empty else pd.DataFrame(),
            metric=metric,
            primary=primary,
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        st.caption(
            "Gap = actual multiple − predicted multiple from the cross-sectional regression.  "
            "Positive = trading at a **premium** to fair value; "
            "negative = **discount**.  "
            "Shaded band shows 25th–75th percentile range across basket tickers."
        )

    with tab3:
        st.subheader("Regression Summary (all metrics)")
        summary_rows = [v["summary_row"] for v in results.values() if v.get("summary_row")]
        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            sdf["metric_label"] = sdf["metric"].map(YIELD_METRIC_LABELS)
            sdf = sdf.sort_values("r2", ascending=False)

            display_cols = {
                "metric_label":             "Metric",
                "r2":                       "Mean R²",
                "c_latest":                 "c (intercept)",
                "m_latest":                 "m (slope)",
                "x_latest":                 "Rev. CAGR p.a.",
                "n_obs":                    "n (tickers)",
                "latest_actual_yield":      "Observed yield",
                "latest_predicted_yield":   "Fair yield",
                "latest_premium_yield":     "Spread (obs−fair)",
                "latest_premium_pct":       "Spread (%)",
                "latest_premium_zscore":    "Z-score",
                "latest_actual_multiple":   "Obs multiple",
                "latest_predicted_multiple":"Fair multiple",
                "latest_premium_multiple":  "Mult spread (%)",
            }
            sdf = sdf.rename(columns=display_cols)
            show_cols = [v for v in display_cols.values() if v in sdf.columns]
            st.dataframe(
                sdf[show_cols].style.format(
                    {
                        "Mean R²":          "{:.3f}",
                        "c (intercept)":    "{:.5f}",
                        "m (slope)":        "{:.4f}",
                        "Rev. CAGR p.a.":   "{:.4f}",
                        "Observed yield":   "{:.4f}",
                        "Fair yield":       "{:.4f}",
                        "Spread (obs−fair)":"{:+.4f}",
                        "Spread (%)":       "{:+.2f}",
                        "Z-score":          "{:+.2f}",
                        "Obs multiple":     "{:.1f}",
                        "Fair multiple":    "{:.1f}",
                        "Mult spread (%)":  "{:+.1f}",
                    },
                    na_rep="–",
                ),
                use_container_width=True,
            )

        # Winsor bounds
        if winsor_on:
            with st.expander("Winsorisation bounds applied"):
                bounds = get_winsor_bounds(
                    panel, ALL_METRICS + ["fwd_cagr_3y"],
                    method=w_method, p_low=w_p_low, p_high=w_p_high, n_std=w_nstd,
                )
                bdf = pd.DataFrame(
                    [{"Column": k, "Lo": v[0], "Hi": v[1]} for k, v in bounds.items()]
                )
                st.dataframe(bdf.style.format({"Lo": "{:.4f}", "Hi": "{:.4f}"}, na_rep="–"))

    with tab4:
        st.subheader("Export to CSV")

        def _to_csv(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode()

        # Full panel
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "📥 Full panel (all tickers)",
            data=_to_csv(panel),
            file_name="panel_all.csv",
            mime="text/csv",
        )

        # Security predictions
        if preds_df is not None:
            c2.download_button(
                f"📥 {primary} predictions",
                data=_to_csv(preds_df),
                file_name=f"{primary}_predictions.csv",
                mime="text/csv",
            )

        # Summary table
        if summary_rows:
            sdf_export = pd.DataFrame(summary_rows)
            c3.download_button(
                "📥 Regression summary",
                data=_to_csv(sdf_export),
                file_name="regression_summary.csv",
                mime="text/csv",
            )

        # Also auto-save to output/
        if st.button("💾 Save all outputs to /output"):
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            panel.to_csv(OUTPUT_DIR / f"panel_{ts}.csv", index=False)
            if preds_df is not None:
                preds_df.to_csv(OUTPUT_DIR / f"{primary}_predictions_{ts}.csv", index=False)
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    OUTPUT_DIR / f"regression_summary_{ts}.csv", index=False
                )
            st.success(f"Saved to /output/ with timestamp {ts}")


if __name__ == "__main__":
    main()
