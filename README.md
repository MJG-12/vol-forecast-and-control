# Volatility Forecasting and Risk Control

Do more accurate daily variance forecasts produce better volatility-control
outcomes?

This project compares RiskMetrics EWMA, HAR, GARCH and GJR-GARCH in a common
walk-forward experiment on S&P 500 Total Return data. Forecasts are evaluated
statistically using QLIKE and Diebold–Mariano tests, then passed through the
same capped daily volatility-control policy with 0–5 bps one-way transaction
costs.

GJR is the strongest variance forecaster, but plain GARCH is the better
volatility-control strategy. Both keep realized volatility below the 10%
budget; relative to GJR, GARCH produces higher returns and Sharpe, lower
turnover and a shallower drawdown at every tested cost. Incremental forecast
accuracy therefore does not necessarily improve the overall control outcome
once the risk objective is already met.

The [results notebook](notebooks/01_results.ipynb) contains the complete
methodology, tables and interpretation.

## Results

All models are evaluated on the same 4,535-observation walk-forward sample.
Lower QLIKE and volatility RMSE are better; higher Spearman correlation is
better.

| Model | QLIKE | Difference from EWMA | Volatility RMSE | Spearman |
|---|---:|---:|---:|---:|
| GJR-GARCH | 1.573 | -0.0926 | 0.1368 | 0.425 |
| HAR | 1.585 | -0.0801 | 0.1496 | 0.411 |
| GARCH | 1.618 | -0.0472 | 0.1389 | 0.378 |
| EWMA | 1.665 | 0.0000 | 0.1416 | 0.362 |

GJR ranks first on all three forecast metrics. HAR ranks second by QLIKE and
regime ordering but has the weakest volatility scale RMSE. Every alternative
has lower average QLIKE than EWMA, and the direction of each loss difference
is unchanged across Newey–West lags of 0, 5 and 20.

The same forecasts then determine the risky weight in a daily-reset,
unlevered 10% volatility-control strategy. At 1 bp one-way:

| Model | Annual return | Excess Sharpe | Realized volatility | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|---:|
| GARCH | 8.35% | 0.731 | 9.70% | -19.01% | 8.29x |
| HAR | 8.00% | 0.717 | 9.40% | -20.39% | 14.28x |
| EWMA | 8.27% | 0.703 | 10.03% | -18.51% | 4.24x |
| GJR-GARCH | 7.96% | 0.697 | 9.65% | -19.45% | 9.24x |

GARCH has the highest excess Sharpe at 0, 1 and 5 bps despite ranking only
third on QLIKE. More importantly, it dominates the statistically more accurate
GJR on every reported strategy dimension except realized volatility, where
both are already below budget and differ by only 0.05 percentage points. HAR's
high turnover erodes its zero-cost advantage as costs rise. EWMA remains
competitive because it trades least and records the shallowest drawdown.

All four policies reduce realized volatility from 19.99% for buy-and-hold to
approximately the 10% risk budget. The relevant question is therefore not only
which model forecasts variance most accurately, but which one meets the risk
budget with the strongest return, turnover and drawdown trade-off.

## Experimental design

- **Data:** S&P 500 Total Return closes and S&P 500 OHLC data from January 2004
  through 5 February 2026.
- **Target:** next-day squared log return, annualized using 252 trading days,
  as a noisy proxy for latent daily variance.
- **Timing:** every forecast uses information available by the previous close.
- **Evaluation:** one common walk-forward sample after a 1,000-observation
  training window; HAR, GARCH and GJR are refitted every 60 observations.
- **Baseline:** RiskMetrics EWMA with fixed daily decay
  \(\lambda=0.94\).
- **Alternatives:** HAR using lagged 1/5/22-day range-variance components, plus
  GARCH and GJR-GARCH with Student-\(t\) innovations.
- **Primary loss:** QLIKE on the variance scale. Volatility RMSE and Spearman
  correlation are secondary diagnostics.
- **Uncertainty:** Diebold–Mariano comparisons against EWMA with Newey–West
  lags of 0, 5 and 20.

The strategy allocates between the equity index and a lagged overnight cash
proxy:

\[
w_t=\min\left(1,\frac{0.10}{\widehat{\sigma}_t}\right).
\]

The weight is reset daily and cannot exceed 100% equity. Transaction costs are
charged per one-way unit of drift-adjusted turnover. Strategy Sharpe ratios use
returns in excess of the cash proxy.

## Repository

```text
src/vol_forecast/
  data.py        market and cash data
  features.py    targets, EWMA and HAR components
  models/        walk-forward HAR and GARCH-family forecasts
  evaluation.py  QLIKE metrics and Diebold–Mariano tests
  strategy.py    capped volatility-control backtest
  experiment.py  shared experiment orchestration
  config.py      fixed experiment settings
notebooks/
  01_results.ipynb
tests/
```

The notebook is intentionally a thin presentation layer; the forecasting,
evaluation and strategy logic lives in the installable package.

## Reproduction

```bash
git clone https://github.com/MJG-12/vol-forecast-and-control.git
cd vol-forecast-and-control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
jupyter notebook notebooks/01_results.ipynb
```

On Windows, activate the environment with
`.venv\Scripts\Activate.ps1` in PowerShell.

Data are downloaded from public sources:

- S&P 500 Total Return index: Yahoo Finance `^SP500TR`.
- S&P 500 OHLC series used by HAR: Yahoo Finance `^GSPC`.
- Cash proxy: FRED effective federal funds rate `DFF`, converted using
  ACT/360 and aligned with a one-trading-day lag.

## Scope

The experiment concerns one equity index, a one-day forecast horizon and one
capped volatility-control policy. Squared daily returns are a noisy variance
proxy, and strategy rankings reflect both the forecast and the realized return
path. The conclusion is therefore specific: when several models already meet
the risk budget, additional forecast accuracy need not improve the strategy's
overall return, turnover and drawdown trade-off.

## AI assistance

OpenAI Codex assisted with research discussion, implementation and refactoring,
debugging, analysis, code review and writing. The author initiated and directed
the project, contributed substantial portions of the implementation, reviewed
the outputs and takes responsibility for the final code, results and
conclusions.
