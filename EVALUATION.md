# Evaluation

## Real run (2026-11-03, verified in this workspace)

Data: FRED S&P 500 daily closes, 2016-11-04 to 2026-11-02, 2,510 returns.
Split: 2,258 in-sample, last 252 held out for the VaR backtest.

| Model | Params | Log-likelihood | Held-out VaR violations (expected 5%) |
|---|---|---|---|
| GARCH(1,1) | omega 0.031, alpha 0.157, beta 0.820 | -2953.9 | 6.35% (16/252) |
| GJR-GARCH(1,1) | omega 0.034, alpha 0.042, gamma 0.224, beta 0.826 | -2912.1 | 4.37% (11/252) |

Asymmetry result: a negative shock raises next-day variance by
alpha + gamma = 0.266, vs 0.042 for an equal positive shock — about
6.3× stronger. The data did support asymmetry here, and GJR-GARCH was
both higher-likelihood and closer to 5% VaR coverage than plain GARCH.

This project ships when the numbers below are filled in from a real run,
not from the TikTok's simulated data.

## Gate 1 — Reproduction (must pass)
- `python src/reproduce_tiktok.py` prints 2.50 variance / 1.58% vol for the
  4% shock, 1.49% / 1.39% / 1.16% for days 5 / 10 / 30, ~13.5-day half-life.
- `pytest` green: shock, decay, clustering, and fit-recovery on known
  simulated parameters.

## Gate 2 — Real data (must pass)
- Fit on SPY (or BTC) closes from Stooq, at least 1,000 returns.
- Report fitted omega / alpha / beta (and gamma for GJR) with
  in-sample log-likelihood for both models.
- Held-out (last 252 days) VaR backtest for both models: expected 5%,
  observed rate, violation count. The better model is the one closer to
  5% coverage, not the one with the prettier chart.

## Gate 3 — Asymmetry check (must report, win or lose)
- If gamma is ~0, say so: the data did not support asymmetry here.
- If gamma > 0, quantify it: how much more does a negative shock raise
  next-day variance than an equal positive shock?

## Known failures to watch for
- MLE not converging on short samples: lengthen the window, check
  stationarity (alpha + beta < 1, alpha + gamma/2 + beta < 1 for GJR).
- VaR violations clustering in a crisis window: that is a model failure
  worth reporting, not a data point to drop.
