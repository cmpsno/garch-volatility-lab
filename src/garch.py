"""GARCH(1,1) and GJR-GARCH(1,1) core.

Formula (GARCH(1,1), returns in percent):
    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

GJR-GARCH adds an asymmetry term so falls hit volatility harder than rises,
the gap the @quantfinancetogo TikTok flags in its own honest notes:
    sigma^2_t = omega + alpha * eps^2_{t-1}
                + gamma * eps^2_{t-1} * 1[eps_{t-1} < 0]
                + beta * sigma^2_{t-1}

Units: everything here is in percent per day, matching the TikTok
(omega=0.05, alpha=0.10, beta=0.85, baseline volatility 1%/day).
Variance is in percent-squared per day.
"""

from __future__ import annotations

import numpy as np


# --- TikTok reproduction -------------------------------------------------

TIKTOK_OMEGA = 0.05
TIKTOK_ALPHA = 0.10
TIKTOK_BETA = 0.85


def next_variance(omega: float, alpha: float, beta: float,
                  last_return_pct: float, last_variance: float) -> float:
    """One-step GARCH(1,1) variance forecast."""
    return omega + alpha * last_return_pct ** 2 + beta * last_variance


def forecast_decay(omega: float, alpha: float, beta: float,
                   start_variance: float, days: int) -> np.ndarray:
    """Expected variance path with no new shocks.

    E[eps^2] = sigma^2, so E[sigma^2_{t+1}] = omega + (alpha+beta) * sigma^2_t,
    which mean-reverts to omega / (1 - alpha - beta).
    Day 1 in the returned array is the first forecast day.
    """
    out = np.empty(days)
    var = start_variance
    for i in range(days):
        var = omega + (alpha + beta) * var
        out[i] = var
    return out


def half_life_days(alpha: float, beta: float) -> float:
    """Days for excess variance to halve: ln(0.5) / ln(alpha+beta)."""
    return float(np.log(0.5) / np.log(alpha + beta))


# --- Simulation ----------------------------------------------------------

def simulate_garch(omega: float, alpha: float, beta: float, n: int,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns (percent) and variances from a GARCH(1,1) model.

    Returns (returns, variances). The unconditional variance
    omega / (1 - alpha - beta) seeds the first variance.
    """
    rng = np.random.default_rng(seed)
    variances = np.empty(n)
    returns = np.empty(n)
    var = omega / (1 - alpha - beta)
    for i in range(n):
        variances[i] = var
        returns[i] = rng.normal(0.0, np.sqrt(var))
        var = omega + alpha * returns[i] ** 2 + beta * var
    return returns, variances


def clustering_rate(returns: np.ndarray, threshold: float = 2.5) -> dict:
    """Share of |return| > threshold overall vs. the day after such a move.

    This is the TikTok's headline statistic, reproduced on simulated data:
    in their 1,000,000 simulated days, >2.5% moves were 1.75% overall but
    10.97% the day after such a move (~6.3x).
    """
    big = np.abs(returns) > threshold
    overall = float(big.mean())
    after = float(big[1:][big[:-1]].mean()) if big[:-1].any() else float("nan")
    return {
        "overall": overall,
        "day_after_big_move": after,
        "ratio": float(after / overall) if overall else float("nan"),
        "n": int(len(returns)),
    }


# --- Fitting (MLE, no `arch` dependency) ---------------------------------

def _garch_variances(params, returns, gjr: bool = False) -> np.ndarray:
    if gjr:
        omega, alpha, gamma, beta = params
    else:
        omega, alpha, beta = params
        gamma = 0.0
    n = len(returns)
    var = np.empty(n)
    # Seed with the sample variance, a standard choice.
    var[0] = float(np.var(returns)) or 1.0
    for t in range(1, n):
        shock = returns[t - 1] ** 2
        asym = gamma * shock if returns[t - 1] < 0 else 0.0
        var[t] = omega + alpha * shock + asym + beta * var[t - 1]
        var[t] = max(var[t], 1e-8)
    return var


def _neg_log_likelihood(params, returns, gjr: bool = False) -> float:
    var = _garch_variances(params, returns, gjr=gjr)
    # Gaussian log-likelihood for returns ~ N(0, var).
    return float(0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var))


def fit_garch(returns: np.ndarray, gjr: bool = False) -> dict:
    """Fit GARCH(1,1) (or GJR-GARCH) by Gaussian MLE with scipy.

    Constraints keep the model stationary and variances positive:
    omega > 0, alpha >= 0, beta >= 0, gamma >= 0, alpha + beta (+ gamma/2) < 1.
    Returns the fitted params, log-likelihood and the fitted variance series.
    """
    from scipy.optimize import minimize

    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    sample_var = float(np.var(returns)) or 1.0
    if gjr:
        x0 = np.array([0.05 * sample_var, 0.05, 0.05, 0.85])
        bounds = [(1e-6, None), (0.0, 0.5), (0.0, 0.5), (0.0, 0.99)]
        cons = {"type": "ineq",
                "fun": lambda p: 0.999 - (p[1] + p[2] / 2 + p[3])}
    else:
        x0 = np.array([0.05 * sample_var, 0.10, 0.85])
        bounds = [(1e-6, None), (0.0, 0.5), (0.0, 0.99)]
        cons = {"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])}
    res = minimize(_neg_log_likelihood, x0, args=(returns, gjr),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500})
    params = res.x
    names = (["omega", "alpha", "gamma", "beta"] if gjr
             else ["omega", "alpha", "beta"])
    return {
        "model": "GJR-GARCH(1,1)" if gjr else "GARCH(1,1)",
        "params": {name: float(v) for name, v in zip(names, params)},
        "log_likelihood": float(-res.fun),
        "success": bool(res.success),
        "variances": _garch_variances(params, returns, gjr=gjr),
    }


# --- VaR backtest --------------------------------------------------------

def var_backtest(returns: np.ndarray, variances: np.ndarray,
                 level: float = 0.95) -> dict:
    """1-day VaR backtest.

    VaR = z * sigma with z = 1.65 for a normal 95% 1-day VaR, as in the
    TikTok's notes. A violation is a return below -VaR. Good coverage means
    the violation rate is close to 1 - level (5%).
    """
    from scipy.stats import norm

    z = float(norm.ppf(level))
    var_limit = z * np.sqrt(variances)
    violations = returns < -var_limit
    rate = float(violations.mean())
    return {
        "level": level,
        "z": z,
        "expected_violation_rate": float(1 - level),
        "observed_violation_rate": rate,
        "n_violations": int(violations.sum()),
        "n": int(len(returns)),
    }
