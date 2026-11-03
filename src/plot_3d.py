"""3D volatility landscape from real S&P 500 data, in the style of the
@quantfinancetogo GARCH screenshot (dark background, blue 3D columns).

Screenshot: a grid of columns, height = daily move, grouped so clusters
are visible. Their grid was 20 simulated stocks sharing one GARCH vol.
Ours is real data, so the grid is different and labelled honestly:
  x = trading day within the year (1..~252)
  y = year (2017..2026)
  z = |daily log return| in percent

Tall bars bunching together = volatility clustering, exactly as in 2020
and 2022. Colour deepens with bar height.

Usage: python src/plot_3d.py [data/sp500.csv] [output.png]
"""

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_returns(path):
    dates, closes = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["SP500"]:
                dates.append(row["observation_date"])
                closes.append(float(row["SP500"]))
    closes = np.array(closes)
    rets = 100 * np.diff(np.log(closes))
    return dates[1:], rets


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sp500.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "garch_3d_my_data.png"
    dates, rets = load_returns(path)

    # Group by year, sequential trading-day index within each year.
    years = sorted({d[:4] for d in dates})
    xs, ys, zs = [], [], []
    for yi, year in enumerate(years):
        idx = [i for i, d in enumerate(dates) if d.startswith(year)]
        for di, i in enumerate(idx):
            xs.append(di)
            ys.append(yi)
            zs.append(abs(rets[i]))
    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)

    fig = plt.figure(figsize=(14, 8), facecolor="#050508")
    ax = fig.add_subplot(111, projection="3d", facecolor="#050508")
    ax.set_facecolor("#050508")

    # Colour: blue, brighter/taller = bigger move.
    norm = plt.Normalize(0, max(6.0, zs.max()))
    colors = cm.Blues(0.35 + 0.65 * norm(zs))
    # Cap a few extreme COVID bars for scale readability, note it.
    zs_plot = np.minimum(zs, 8.0)

    ax.bar3d(xs, ys, np.zeros_like(zs_plot), 0.55, 0.65, zs_plot,
             color=colors, shade=True, edgecolor="none", alpha=0.95)

    ax.set_xlabel("trading day in year →", color="white", labelpad=12)
    ax.set_ylabel("year →", color="white", labelpad=12)
    ax.set_zlabel("daily move (%)", color="white", labelpad=12)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, color="white", fontsize=9)
    ax.set_xticks([0, 63, 126, 189, 251])
    ax.set_xticklabels(["0", "63", "126", "189", "251"], color="white", fontsize=9)
    ax.tick_params(axis="z", colors="white", labelsize=9)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#0b0b14")
        pane.set_edgecolor("#333344")
    ax.grid(True, color="#222233", linewidth=0.5)
    ax.view_init(elev=24, azim=-48)

    fig.suptitle("GARCH  —  Forecasting tomorrow's volatility",
                 color="white", fontsize=20, fontweight="bold", y=0.96)
    ax.set_title("S&P 500 daily moves, 2016–2026  •  2,510 trading days  •  "
                 "tall bars bunch together = volatility clusters",
                 color="#aaaaaa", fontsize=11, pad=18)
    fig.text(0.5, 0.02,
             "Each column = |daily log return| for one trading day. "
             "Bars capped at 8% for scale. Source: FRED S&P 500 (real data, not simulated).",
             ha="center", color="#888888", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=200, facecolor="#050508", bbox_inches="tight")
    print(f"saved {out}  ({len(zs)} bars, max |return| {zs.max():.2f}%)")


if __name__ == "__main__":
    main()
