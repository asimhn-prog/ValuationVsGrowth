"""
Valuation vs Growth – CLI batch runner.

Usage examples
--------------
# Minimal (Tier-3 trailing data from yfinance):
  python cli.py --primary AAPL --basket MSFT,GOOGL,META --start 2020-01-01 --end 2024-12-31

# With BYOD forward fundamentals:
  python cli.py --primary AAPL --basket MSFT,GOOGL,META \\
    --byod config/byod_template.csv --start 2020-01-01 --end 2024-12-31

# With winsorisation (percentile) and rolling window:
  python cli.py --primary AAPL --basket MSFT,GOOGL,META \\
    --winsor percentile --winsor-p 2.5 --rolling 36 \\
    --start 2020-01-01 --end 2024-12-31

# Load config from JSON:
  python cli.py --config config/example_basket.json

All outputs are saved to ./output/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.assembler import build_panel
from transforms.compute_ev import ensure_ev
from transforms.compute_forward_metrics import (
    ALL_METRICS,
    YIELD_METRIC_LABELS,
    compute_valuation_yields,
)
from transforms.compute_cagr import compute_forward_cagr
from transforms.winsor import winsorise_panel
from models.regression import run_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config",    default=None,            help="Path to JSON config file (overrides other flags).")
@click.option("--primary",   default="AAPL",          help="Primary ticker symbol.")
@click.option("--basket",    default="MSFT,GOOGL,META", help="Basket tickers, comma-separated.")
@click.option("--market",    default="SPY",            help="Market proxy ticker.")
@click.option("--start",     default="2019-01-01",     help="Start date YYYY-MM-DD.")
@click.option("--end",       default="2024-12-31",     help="End date YYYY-MM-DD.")
@click.option("--freq",      default="ME",             help="Frequency: ME (monthly) or QE (quarterly).")
@click.option("--byod",      default=None,             help="Path to BYOD CSV/Excel file.")
@click.option("--fcf-ev/--fcf-mktcap", default=True,  help="FCF yield denominator: EV or Market Cap.")
@click.option("--winsor",    default=None,             type=click.Choice(["percentile", "stddev"]),
              help="Winsorisation method.")
@click.option("--winsor-p",  default=2.5,              help="Percentile trim % each tail (e.g. 2.5).")
@click.option("--winsor-std",default=2.5,              help="Std-dev clip multiplier.")
@click.option("--rolling",   default=0,                help="Rolling OLS window in months (0 = full sample).")
@click.option("--metrics",   default=",".join(ALL_METRICS), help="Comma-separated metrics to run.")
@click.option("--min-obs",   default=10,               help="Minimum observations for OLS.")
@click.option("--verbose",   is_flag=True,             help="Enable DEBUG logging.")
def run(
    config, primary, basket, market, start, end, freq,
    byod, fcf_ev, winsor, winsor_p, winsor_std, rolling,
    metrics, min_obs, verbose,
):
    """Valuation vs Growth CLI – run batch analysis and save CSVs."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load JSON config if supplied ─────────────────────────────────────────
    if config:
        cfg_path = Path(config)
        if not cfg_path.exists():
            logger.error("Config file not found: %s", config)
            sys.exit(1)
        cfg = json.loads(cfg_path.read_text())
        primary = cfg.get("primary_ticker", primary)
        basket  = ",".join(cfg.get("basket_tickers", basket.split(",")))
        market  = cfg.get("market_proxy", market)
        start   = cfg.get("date_range", {}).get("start", start)
        end     = cfg.get("date_range", {}).get("end", end)
        freq    = cfg.get("frequency", freq)
        byod    = cfg.get("byod_path") or byod
        fcf_ev  = cfg.get("fcf_use_ev", fcf_ev)
        rolling = cfg.get("rolling_window", rolling) or 0
        w_cfg   = cfg.get("winsor", {})
        if w_cfg.get("enabled"):
            winsor     = w_cfg.get("method", "percentile")
            winsor_p   = w_cfg.get("p_low", 0.025) * 100
            winsor_std = w_cfg.get("n_std", 2.5)
        logger.info("Loaded config from %s", config)

    basket_list   = [t.strip().upper() for t in basket.split(",") if t.strip()]
    metrics_list  = [m.strip() for m in metrics.split(",") if m.strip() in ALL_METRICS]
    rolling_win   = int(rolling) if int(rolling) > 0 else None
    all_tickers   = list(dict.fromkeys([primary.upper()] + basket_list + [market.upper()]))

    click.echo(f"\n{'='*60}")
    click.echo(f"  Primary  : {primary.upper()}")
    click.echo(f"  Basket   : {', '.join(basket_list)}")
    click.echo(f"  Market   : {market.upper()}")
    click.echo(f"  Period   : {start}  →  {end}  ({freq})")
    click.echo(f"  BYOD     : {byod or 'None (Tier-3 trailing)'}")
    click.echo(f"  Metrics  : {', '.join(metrics_list)}")
    click.echo(f"  Winsor   : {winsor or 'off'}")
    click.echo(f"  Rolling  : {rolling_win or 'full-sample'}")
    click.echo(f"{'='*60}\n")

    # ── Build panel ──────────────────────────────────────────────────────────
    logger.info("Building panel for %d tickers …", len(all_tickers))
    panel = build_panel(all_tickers, start, end, byod_path=byod, freq=freq)

    if panel.empty:
        logger.error("Panel is empty. Check tickers and date range.")
        sys.exit(1)

    # ── Transform ────────────────────────────────────────────────────────────
    panel = ensure_ev(panel)
    panel = compute_valuation_yields(panel, fcf_use_ev=fcf_ev)
    panel = compute_forward_cagr(panel)

    if winsor:
        cols_to_winsor = metrics_list + ["fwd_cagr_3y"]
        cols_to_winsor = [c for c in cols_to_winsor if c in panel.columns]
        panel = winsorise_panel(
            panel,
            columns=cols_to_winsor,
            method=winsor,
            p_low=winsor_p / 100,
            p_high=winsor_p / 100,
            n_std=winsor_std,
        )
        logger.info("Winsorisation applied (%s).", winsor)

    # ── Regression ───────────────────────────────────────────────────────────
    basket_panel   = panel[panel["ticker"].isin(basket_list + [market.upper()])].copy()
    security_panel = panel[panel["ticker"] == primary.upper()].copy()

    logger.info("Running OLS regressions …")
    results = run_all_metrics(
        basket_df=basket_panel,
        security_df=security_panel,
        metrics=metrics_list,
        rolling_window=rolling_win,
        min_obs=int(min_obs),
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    click.echo("\n── Regression summary ────────────────────────────────────────")
    header = f"{'Metric':<18} {'R²':>6} {'α':>10} {'β':>8} {'n':>6}  " \
             f"{'Act.Yld':>9} {'Pred.Yld':>9} {'Prem(%)':>9} {'Z-score':>8}"
    click.echo(header)
    click.echo("─" * len(header))

    rows_sorted = sorted(results.items(), key=lambda x: x[1]["summary_row"].get("r2", 0), reverse=True)
    for metric_name, res in rows_sorted:
        sr = res["summary_row"]
        def _f(k, fmt=".4f"):
            v = sr.get(k)
            return f"{v:{fmt}}" if v is not None and str(v) != "nan" else "    –"

        click.echo(
            f"{YIELD_METRIC_LABELS.get(metric_name, metric_name):<18} "
            f"{_f('r2','.4f'):>6} {_f('alpha','.5f'):>10} {_f('beta','.4f'):>8} "
            f"{sr.get('n_obs',0):>6}  "
            f"{_f('latest_actual_yield','.4f'):>9} {_f('latest_predicted_yield','.4f'):>9} "
            f"{_f('latest_premium_pct','+.2f'):>9} {_f('latest_premium_zscore','+.2f'):>8}"
        )

    click.echo()

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    panel_path = OUTPUT_DIR / f"panel_{primary}_{ts}.csv"
    panel.to_csv(panel_path, index=False)
    logger.info("Panel saved: %s", panel_path)

    summary_rows = [v["summary_row"] for v in results.values()]
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"summary_{primary}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved: %s", summary_path)

    for metric_name, res in results.items():
        preds_df = res.get("predictions")
        if preds_df is not None and not preds_df.empty:
            p_path = OUTPUT_DIR / f"{primary}_{metric_name}_predictions_{ts}.csv"
            preds_df.to_csv(p_path, index=False)
            logger.info("Predictions saved: %s", p_path)

        rolling_df = res.get("rolling")
        if rolling_df is not None and not rolling_df.empty:
            r_path = OUTPUT_DIR / f"{primary}_{metric_name}_rolling_ols_{ts}.csv"
            rolling_df.to_csv(r_path, index=False)
            logger.info("Rolling OLS saved: %s", r_path)

    click.echo(f"\n✅ All outputs saved to  {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    run()
