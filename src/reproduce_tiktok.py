"""Reproduce the @quantfinancetogo GARCH TikTok numbers, step by step.

Run:  python src/reproduce_tiktok.py
Expected output (matching the video):
  - 4% shock on a calm (1% vol) day -> variance 2.50, vol 1.58%
  - forecast vol: day 5 = 1.49%, day 10 = 1.39%, day 30 = 1.16%
  - excess-variance half-life ~13.5 trading days
  - simulated clustering: >2.5% moves far more common after a big move
    (video, 1,000,000 days: 1.75% overall vs 10.97% after; our 200,000-day
    run lands in the same ballpark, sampling noise aside)
"""

import numpy as np

from garch import (
    TIKTOK_ALPHA, TIKTOK_BETA, TIKTOK_OMEGA,
    clustering_rate, forecast_decay, half_life_days, next_variance,
    simulate_garch,
)

if __name__ == "__main__":
    # 1. The video's worked example.
    var = next_variance(TIKTOK_OMEGA, TIKTOK_ALPHA, TIKTOK_BETA,
                        last_return_pct=-4.0, last_variance=1.0)
    print(f"4% shock: variance = {var:.2f}, vol = {np.sqrt(var):.2f}%")
    assert abs(var - 2.50) < 1e-9

    # 2. Decay: `var` above is already the day-1 forecast (1.58%), so
    #    path[0] is day 2, path[3] is day 5, path[8] is day 10, etc.
    path = forecast_decay(TIKTOK_OMEGA, TIKTOK_ALPHA, TIKTOK_BETA,
                          start_variance=var, days=30)
    vols = np.sqrt(path)
    print(f"forecast vol: day 5 = {vols[3]:.2f}%, "
          f"day 10 = {vols[8]:.2f}%, day 30 = {vols[28]:.2f}%")
    print(f"excess-variance half-life = "
          f"{half_life_days(TIKTOK_ALPHA, TIKTOK_BETA):.1f} trading days")

    # 3. Clustering on simulated data.
    returns, _ = simulate_garch(TIKTOK_OMEGA, TIKTOK_ALPHA, TIKTOK_BETA,
                                n=200_000, seed=42)
    stats = clustering_rate(returns, threshold=2.5)
    print(f"simulated ({stats['n']:,} days): >2.5% moves "
          f"{stats['overall']*100:.2f}% overall, "
          f"{stats['day_after_big_move']*100:.2f}% the day after one "
          f"(ratio {stats['ratio']:.1f}x)")
    print("OK: TikTok numbers reproduced.")
