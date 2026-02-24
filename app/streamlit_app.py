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
import streamlit as st

# ── Path setup (allow running from repo root or app/ directory) ──────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.assembler import build_panel
from transforms.compute_ev import ensure_ev
from transforms.compute_forward_metrics import (
    ALL_METRICS,
    MULTIPLE_LABELS,
    YIELD_METRIC_LABELS,
    compute_valuation_yields,
)
from transforms.compute_cagr import compute_forward_cagr
from transforms.winsor import winsorise_panel, get_winsor_bounds
from models.regression import run_all_metrics

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
    freq: str,
) -> pd.DataFrame:
    """Cached panel build (re-runs only when inputs change)."""
    byod_path = None
    if byod_bytes is not None:
        tmp = OUTPUT_DIR / "_byod_upload.csv"
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
    basket_agg: str = "points",
    show_regression: bool = True,
    highlight_current: bool = True,
    reg_result=None,
) -> go.Figure:
    """
    Scatter: X = fwd_cagr_3y, Y = metric yield.
    """
    fig = go.Figure()
    x_col = "fwd_cagr_3y"

    def add_scatter(subset, name, color, symbol="circle", size=7, opacity=0.55):
        sub = subset[[x_col, metric, "date"]].dropna()
        if sub.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=sub[x_col] * 100,
                y=sub[metric] * 100,
                mode="markers",
                name=name,
                marker=dict(color=color, size=size, symbol=symbol, opacity=opacity),
                customdata=sub[["date"]].values,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Date: %{customdata[0]|%Y-%m-%d}<br>"
                    f"CAGR: %{{x:.2f}}%<br>{YIELD_METRIC_LABELS.get(metric, metric)}: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    # Basket
    basket_tickers = [t for t in basket if t != primary]
    basket_data = panel[panel["ticker"].isin(basket_tickers)]

    if basket_agg == "points":
        add_scatter(basket_data, "Basket", _COLORS["basket"], symbol="diamond", size=6)
    elif basket_agg in ("median", "mean"):
        agg_fn = basket_data.groupby("date")[[x_col, metric]].median if basket_agg == "median" \
            else basket_data.groupby("date")[[x_col, metric]].mean
        agg = agg_fn().reset_index()
        fig.add_trace(
            go.Scatter(
                x=agg[x_col] * 100, y=agg[metric] * 100,
                mode="markers+lines",
                name=f"Basket {basket_agg.capitalize()}",
                marker=dict(color=_COLORS["basket"], size=7, symbol="diamond"),
                line=dict(color=_COLORS["basket"], width=1, dash="dot"),
                hovertemplate=(
                    f"<b>Basket {basket_agg}</b><br>Date: %{{customdata}}<br>"
                    f"CAGR: %{{x:.2f}}%<br>{YIELD_METRIC_LABELS.get(metric, metric)}: %{{y:.2f}}%<extra></extra>"
                ),
                customdata=agg["date"].dt.strftime("%Y-%m-%d").values,
            )
        )

    # Market
    mkt_data = panel[panel["ticker"] == market]
    add_scatter(mkt_data, f"Market ({market})", _COLORS["market"], symbol="square", size=6)

    # Security history
    sec_data = panel[panel["ticker"] == primary]
    add_scatter(sec_data, f"{primary} (history)", _COLORS["security"], symbol="circle", size=8, opacity=0.7)

    # Highlight current (latest) point for primary
    if highlight_current:
        latest = sec_data.dropna(subset=[x_col, metric])
        if not latest.empty:
            cur = latest.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[cur[x_col] * 100], y=[cur[metric] * 100],
                    mode="markers",
                    name=f"{primary} (latest)",
                    marker=dict(color=_COLORS["current"], size=14, symbol="star"),
                    hovertemplate=(
                        f"<b>{primary} LATEST</b><br>"
                        f"Date: {cur['date'].strftime('%Y-%m-%d')}<br>"
                        f"CAGR: {cur[x_col]*100:.2f}%<br>"
                        f"{YIELD_METRIC_LABELS.get(metric, metric)}: {cur[metric]*100:.2f}%<extra></extra>"
                    ),
                )
            )

    # Regression line
    if show_regression and reg_result is not None:
        all_data = panel[panel["ticker"].isin(basket + [market])].dropna(subset=[x_col, metric])
        if not all_data.empty:
            x_range = np.linspace(all_data[x_col].min(), all_data[x_col].max(), 100)
            y_line  = reg_result.alpha + reg_result.beta * x_range
            fig.add_trace(
                go.Scatter(
                    x=x_range * 100, y=y_line * 100,
                    mode="lines",
                    name=f"OLS fit (R²={reg_result.r2:.2f})",
                    line=dict(color=_COLORS["regression"], width=2, dash="dash"),
                    hovertemplate="CAGR: %{x:.2f}%<br>Fitted yield: %{y:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"{YIELD_METRIC_LABELS.get(metric, metric)} vs 3Y Fwd Revenue CAGR",
            font=dict(size=16),
        ),
        xaxis=dict(title="3-Year Forward Revenue CAGR (%)", ticksuffix="%", gridcolor="#e5e7eb"),
        yaxis=dict(title=f"{YIELD_METRIC_LABELS.get(metric, metric)} (%)", ticksuffix="%", gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=-0.2),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
    )
    return fig


def premium_timeseries_plot(
    preds_df: pd.DataFrame,
    primary: str,
    metric: str,
) -> go.Figure:
    """Premium/discount time series for the primary security."""
    fig = go.Figure()

    data = preds_df.dropna(subset=["premium_yield", "date"])

    # Shade positive (expensive) vs negative (cheap) regions
    fig.add_hline(y=0, line=dict(color="black", width=1.5))

    # Premium area
    pos = data["premium_yield"].clip(lower=0) * 100
    neg = data["premium_yield"].clip(upper=0) * 100

    fig.add_trace(
        go.Scatter(
            x=data["date"], y=pos, fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)", line=dict(width=0),
            name="Rich vs regression", showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"], y=neg, fill="tozeroy",
            fillcolor="rgba(22,163,74,0.15)", line=dict(width=0),
            name="Cheap vs regression", showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"], y=data["premium_yield"] * 100,
            mode="lines",
            name="Premium (actual − predicted)",
            line=dict(color=_COLORS["security"], width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Premium: %{y:.2f}%<extra></extra>",
        )
    )

    # 1-sigma bands
    sig = data["premium_yield"].std() * 100
    mu  = data["premium_yield"].mean() * 100
    for k, style in [(1, "dot"), (2, "dash")]:
        for sign, lbl in [(1, f"+{k}σ"), (-1, f"-{k}σ")]:
            fig.add_hline(
                y=mu + sign * k * sig,
                line=dict(color="gray", width=1, dash=style),
                annotation_text=lbl,
                annotation_position="right",
            )

    fig.update_layout(
        title=dict(
            text=f"{primary} – Premium/Discount on {YIELD_METRIC_LABELS.get(metric, metric)}",
            font=dict(size=16),
        ),
        xaxis=dict(title="Date", gridcolor="#e5e7eb"),
        yaxis=dict(title="Yield Spread (actual − predicted, %)", ticksuffix="%", gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
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
        freq_label = st.selectbox("Frequency", ["Monthly", "Quarterly"])
        freq = "ME" if freq_label == "Monthly" else "QE"

        st.divider()
        st.header("BYOD Data")
        st.markdown(
            "_Upload a Bloomberg CSV export. See `config/byod_template.csv` for schema._"
        )
        byod_file = st.file_uploader("Upload BYOD CSV / Excel", type=["csv", "xlsx"])
        byod_bytes = byod_file.read() if byod_file else None

        st.divider()
        st.header("Settings")
        fcf_use_ev = st.toggle("FCF yield denominator = EV (else Mkt Cap)", value=True)

        metric = st.selectbox(
            "Valuation metric",
            options=ALL_METRICS,
            format_func=lambda m: YIELD_METRIC_LABELS.get(m, m),
        )

        basket_agg = st.selectbox(
            "Basket display", ["points", "median", "mean"],
            format_func=lambda v: {"points": "Individual points", "median": "Median per date", "mean": "Mean per date"}[v],
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
    tier_rows = "  |  ".join(f"{r.ticker}: {tier_label(r.data_tier)}" for _, r in tiers.iterrows())
    st.caption(f"Data tiers — {tier_rows}")

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

    reg_result  = results.get(metric, {}).get("reg")
    preds_df    = results.get(metric, {}).get("predictions", security_panel)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Scatter", "📉 Premium/Discount", "📋 Summary Table", "💾 Export"]
    )

    with tab1:
        st.subheader(f"{YIELD_METRIC_LABELS.get(metric, metric)} — Scatter")
        fig_scatter = scatter_plot(
            panel, primary, basket_list + [market], market,
            metric, basket_agg=basket_agg,
            show_regression=show_reg, reg_result=reg_result,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        if reg_result:
            cols = st.columns(4)
            cols[0].metric("R²",    f"{reg_result.r2:.3f}")
            cols[1].metric("α (intercept)", f"{reg_result.alpha:.4f}")
            cols[2].metric("β (slope)",     f"{reg_result.beta:.4f}")
            cols[3].metric("Obs",   str(reg_result.n_obs))

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
        st.subheader(f"{primary} — Premium / Discount on {YIELD_METRIC_LABELS.get(metric, metric)}")
        if preds_df is not None and "premium_yield" in preds_df.columns:
            fig_prem = premium_timeseries_plot(preds_df, primary, metric)
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

    with tab3:
        st.subheader("Regression Summary (all metrics)")
        summary_rows = [v["summary_row"] for v in results.values() if v.get("summary_row")]
        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            sdf["metric_label"] = sdf["metric"].map(YIELD_METRIC_LABELS)
            sdf = sdf.sort_values("r2", ascending=False)

            display_cols = {
                "metric_label":             "Metric",
                "r2":                       "R²",
                "alpha":                    "α",
                "beta":                     "β",
                "n_obs":                    "Obs",
                "latest_actual_yield":      "Actual yield",
                "latest_predicted_yield":   "Pred yield",
                "latest_premium_yield":     "Premium (yld)",
                "latest_premium_pct":       "Premium (%)",
                "latest_premium_zscore":    "Z-score",
                "latest_actual_multiple":   "Actual mult",
                "latest_predicted_multiple":"Pred mult",
                "latest_premium_multiple":  "Premium mult (%)",
            }
            sdf = sdf.rename(columns=display_cols)
            show_cols = [v for v in display_cols.values() if v in sdf.columns]
            st.dataframe(
                sdf[show_cols].style.format(
                    {
                        "R²":              "{:.3f}",
                        "α":               "{:.5f}",
                        "β":               "{:.4f}",
                        "Actual yield":    "{:.4f}",
                        "Pred yield":      "{:.4f}",
                        "Premium (yld)":   "{:+.4f}",
                        "Premium (%)":     "{:+.2f}",
                        "Z-score":         "{:+.2f}",
                        "Actual mult":     "{:.1f}",
                        "Pred mult":       "{:.1f}",
                        "Premium mult (%)":"{:+.1f}",
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
