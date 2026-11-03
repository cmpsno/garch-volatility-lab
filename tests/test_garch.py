import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garch import (  # noqa: E402
    clustering_rate, fit_garch, forecast_decay, half_life_days,
    next_variance, simulate_garch,
)


def test_tiktok_shock():
    assert next_variance(0.05, 0.10, 0.85, -4.0, 1.0) == 2.50


def test_tiktok_decay():
    # 2.50 is already the day-1 forecast, so path[0] is day 2.
    vols = np.sqrt(forecast_decay(0.05, 0.10, 0.85, 2.50, 30))
    assert abs(vols[3] - 1.49) < 0.01
    assert abs(vols[8] - 1.39) < 0.01
    assert abs(vols[28] - 1.16) < 0.01
    assert abs(half_life_days(0.10, 0.85) - 13.5) < 0.2


def test_clustering_exists_in_simulation():
    returns, _ = simulate_garch(0.05, 0.10, 0.85, n=50_000, seed=1)
    stats = clustering_rate(returns)
    assert stats["day_after_big_move"] > stats["overall"]


def test_fit_recovers_known_params():
    returns, _ = simulate_garch(0.05, 0.10, 0.85, n=3000, seed=3)
    fit = fit_garch(returns)
    assert fit["success"]
    assert abs(fit["params"]["alpha"] - 0.10) < 0.08
    assert abs(fit["params"]["beta"] - 0.85) < 0.08
