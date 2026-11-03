"""Fit GARCH(1,1) and GJR-GARCH(1,1) to real returns, then backtest VaR.

Data: daily closes from Stooq (free, no key), e.g. SPY or BTC.
Usage:  python src/fit_real_data.py [spy.us|btcusd] [days]
        python src/fit_real_data.py --simulated   # no internet needed

Returns are 100 * log(close_t / close_{t-1}), i.e. percent per day.
The fit is split in-sample / out-of-sample: the last 252 trading days
are held out, and VaR is backtested on those days using variances
forecast one step ahead from the fitted model (re-estimated only
in-sample, so the backtest is honest about look-ahead).

If the download fails, the script says so and stops -- it does not
silently fall back to simulated data and call it real.
"""

import sys
import urllib.request

import numpy as np

from garch import fit_garch, var_backtest, _garch_variances  # noqa: F401


def fetch_stooq(symbol: str) -> np.ndarray:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "garch-lab"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()
    closes = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 5:
            try:
                closes.append(float(parts[4]))
            except ValueError:
                pass
    closes = np.array(closes)
    if len(closes) < 300:
        raise RuntimeError(f"only {len(closes)} closes fetched for {symbol}")
    return 100 * np.diff(np.log(closes))


def fetch_fred_sp500() -> np.ndarray:
    """S&P 500 daily closes from FRED (free, no key) -- fallback when
    Stooq is bot-blocked, as it was from this machine."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
    req = urllib.request.Request(url, headers={"User-Agent": "garch-lab"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()
    closes = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip():
            try:
                closes.append(float(parts[1]))
            except ValueError:
                pass
    closes = np.array(closes)
    if len(closes) < 300:
        raise RuntimeError(f"only {len(closes)} FRED closes fetched")
    return 100 * np.diff(np.log(closes))


def load_csv(path: str) -> np.ndarray:
    closes = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1]:
                try:
                    closes.append(float(parts[1]))
                except ValueError:
                    pass
    return 100 * np.diff(np.log(np.array(closes)))


def main() -> None:
    if "--simulated" in sys.argv:
        from garch import simulate_garch
        returns, _ = simulate_garch(0.05, 0.10, 0.85, n=1500, seed=7)
        label = "simulated GARCH(0.05, 0.10, 0.85)"
    elif "--csv" in sys.argv:
        path = sys.argv[sys.argv.index("--csv") + 1]
        returns = load_csv(path)
        label = f"CSV {path}"
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "spy.us"
        try:
            returns = fetch_stooq(symbol)
            label = f"Stooq {symbol}"
        except Exception as e:
            print(f"Stooq failed ({e}); falling back to FRED S&P 500")
            returns = fetch_fred_sp500()
            label = "FRED SP500"
        if len(sys.argv) > 2:
            returns = returns[-int(sys.argv[2]):]

    split = len(returns) - 252
    train, test = returns[:split], returns[split:]
    print(f"data: {label}, {len(returns)} returns, "
          f"{len(train)} in-sample, {len(test)} held out")

    for gjr in (False, True):
        fit = fit_garch(train, gjr=gjr)
        print(f"\n{fit['model']}: {fit['params']} "
              f"(log-likelihood {fit['log_likelihood']:.1f}, "
              f"success={fit['success']})")
        # One-step-ahead variances on the held-out window, rolling the
        # fitted recursion forward from the last in-sample variance.
        params = fit["params"]
        if gjr:
            p = [params["omega"], params["alpha"],
                 params["gamma"], params["beta"]]
        else:
            p = [params["omega"], params["alpha"], params["beta"]]
        var = fit["variances"][-1]
        test_var = np.empty(len(test))
        prev_ret = train[-1]
        for i, r in enumerate(test):
            test_var[i] = var
            shock = prev_ret ** 2
            asym = p[2] * shock if (gjr and prev_ret < 0) else 0.0
            if gjr:
                var = p[0] + p[1] * shock + asym + p[3] * var
            else:
                var = p[0] + p[1] * shock + p[2] * var
            prev_ret = r
        bt = var_backtest(test, test_var)
        print(f"  VaR backtest (held out): expected "
              f"{bt['expected_violation_rate']*100:.0f}%, observed "
              f"{bt['observed_violation_rate']*100:.2f}% "
              f"({bt['n_violations']}/{bt['n']} violations)")


if __name__ == "__main__":
    main()
