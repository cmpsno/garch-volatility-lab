"""Decay + forecast-vs-realized charts for the README.

Usage: python src/plot_charts.py [data/sp500.csv]
Outputs (dark theme, matching src/plot_3d.py):
  assets/decay.png               -- TikTok shock decay, 30 days
  assets/forecast_vs_realized.png -- held-out 252 days, GJR-GARCH VaR band

Decay chart: video params omega=0.05, alpha=0.10, beta=0.85.
Day 1 is the 4% shock forecast itself (variance 2.50, vol 1.58%);
later days use forecast_decay() with no new shocks, so day 5 = 1.49%,
day 10 = 1.39%, day 30 = 1.16%, matching README / reproduce_tiktok.py.

Forecast chart: same split and rolling one-step-ahead logic as
src/fit_real_data.py -- fit on all but the last 252 returns, then roll
the fitted GJR-GARCH recursion forward through the held-out window.
Plotted: actual daily returns (grey), +/-1.65 * forecast sigma
(the 95% 1-day VaR band, normal approximation), and red dots where
the return breaches -VaR. The GARCH(1,1) forecast is shown faintly
for comparison. Violation counts are printed and must match README:
GARCH 16/252, GJR-GARCH 11/252 on this CSV.
"""

import csv
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from garch import (
    TIKTOK_ALPHA, TIKTOK_BETA, TIKTOK_OMEGA,
    fit_garch, forecast_decay, var_backtest,
)

BG = "#050508"
PANEL = "#0b0b14"
GRID = "#222233"
BLUE = "#4a90d9"
LIGHT_BLUE = "#7fb3e8"
GREY = "#aaaaaa"


def style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333344")
    ax.grid(True, color=GRID, linewidth=0.5)


def load_returns(path):
    dates, closes = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["SP500"]:
                dates.append(row["observation_date"])
                closes.append(float(row["SP500"]))
    closes = np.array(closes)
    return dates[1:], 100 * np.diff(np.log(closes))


def rolling_forecast(fit, train, test, gjr):
    """One-step-ahead variances on test, as in fit_real_data.py."""
    params = fit["params"]
    if gjr:
        p = [params["omega"], params["alpha"],
             params["gamma"], params["beta"]]
    else:
        p = [params["omega"], params["alpha"], params["beta"]]
    var = fit["variances"][-1]
    out = np.empty(len(test))
    prev_ret = train[-1]
    for i, r in enumerate(test):
        out[i] = var
        shock = prev_ret ** 2
        asym = p[2] * shock if (gjr and prev_ret < 0) else 0.0
        if gjr:
            var = p[0] + p[1] * shock + asym + p[3] * var
        else:
            var = p[0] + p[1] * shock + p[2] * var
        prev_ret = r
    return out


def decay_chart():
    # Day 1 = shock forecast; days 2..30 from forecast_decay().
    path = forecast_decay(TIKTOK_OMEGA, TIKTOK_ALPHA, TIKTOK_BETA,
                          start_variance=2.50, days=29)
    vols = np.concatenate([[np.sqrt(2.50)], np.sqrt(path)])
    days = np.arange(1, 31)
    for d, want in ((1, 1.58), (5, 1.49), (10, 1.39), (30, 1.16)):
        got = float(vols[d - 1])
        assert abs(got - want) < 0.015, f"day {d}: {got:.2f} != {want}"
        print(f"decay day {d}: {got:.2f}% (README: {want:.2f}%)")

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    style(ax)
    ax.plot(days, vols, color=BLUE, linewidth=2.5)
    ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1,
               label="baseline 1%/day")
    for d in (1, 5, 10, 30):
        ax.plot(d, vols[d - 1], "o", color="white", markersize=5)
        ax.annotate(f"day {d}: {vols[d - 1]:.2f}%",
                    (d, vols[d - 1]), color="white", fontsize=9,
                    xytext=(6, 8), textcoords="offset points")
    ax.set_xlabel("days after the 4% shock", color="white")
    ax.set_ylabel("forecast volatility (%/day)", color="white")
    ax.set_title("After a 4% shock, volatility decays back to 1%\n"
                 "GARCH(1,1) ω=0.05, α=0.10, β=0.85 — excess variance "
                 "half-life ~13.5 trading days",
                 color="white", fontsize=12)
    ax.legend(facecolor=PANEL, edgecolor="#333344", labelcolor="white")
    fig.tight_layout()
    fig.savefig("assets/decay.png", dpi=200, facecolor=BG,
                bbox_inches="tight")
    print("saved assets/decay.png")


def forecast_chart(path):
    dates, returns = load_returns(path)
    split = len(returns) - 252
    train, test = returns[:split], returns[split:]
    test_dates = dates[split:]
    print(f"data: {len(returns)} returns, {len(train)} in-sample, "
          f"{len(test)} held out ({test_dates[0]} to {test_dates[-1]})")

    fits = {}
    test_var = {}
    for gjr in (False, True):
        fit = fit_garch(train, gjr=gjr)
        fits[gjr] = fit
        test_var[gjr] = rolling_forecast(fit, train, test, gjr)
        bt = var_backtest(test, test_var[gjr])
        print(f"{fit['model']}: params={fit['params']} "
              f"VaR violations {bt['n_violations']}/{bt['n']} "
              f"({bt['observed_violation_rate']*100:.2f}%)")
    # README numbers for this CSV; fail loudly if the data changed.
    assert var_backtest(test, test_var[False])["n_violations"] == 16
    assert var_backtest(test, test_var[True])["n_violations"] == 11

    from scipy.stats import norm
    z = float(norm.ppf(0.95))
    sig_gjr = np.sqrt(test_var[True])
    sig_garch = np.sqrt(test_var[False])
    x = np.arange(1, len(test) + 1)
    violations = test < -z * sig_gjr

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
    style(ax)
    ax.fill_between(x, -z * sig_gjr, z * sig_gjr, color=BLUE,
                    alpha=0.18, label="GJR-GARCH 95% VaR band (±1.65σ)")
    ax.plot(x, test, color=GREY, linewidth=0.8, label="actual return")
    ax.plot(x, sig_gjr, color=BLUE, linewidth=1.5,
            label="GJR-GARCH forecast volatility")
    ax.plot(x, sig_garch, color=LIGHT_BLUE, linewidth=1, linestyle="--",
            label="GARCH(1,1) forecast volatility")
    ax.plot(x[violations], test[violations], "o", color="#e5484d",
            markersize=5, label=f"VaR violations (GJR: {violations.sum()})")
    ax.axhline(0, color="white", linewidth=0.8)
    ax.set_xlabel("held-out trading day (last 252 days)", color="white")
    ax.set_ylabel("return / volatility (%/day)", color="white")
    ax.set_title("Forecast volatility vs realized returns — held-out window\n"
                 "GJR-GARCH breaches VaR 11/252 days (4.37%), "
                 "GARCH(1,1) 16/252 (6.35%); expected 5%",
                 color="white", fontsize=12)
    ax.legend(facecolor=PANEL, edgecolor="#333344", labelcolor="white",
              fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("assets/forecast_vs_realized.png", dpi=200,
                facecolor=BG, bbox_inches="tight")
    print("saved assets/forecast_vs_realized.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sp500.csv"
    decay_chart()
    forecast_chart(path)
