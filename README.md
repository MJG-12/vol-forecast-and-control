# vol-forecast-and-control

In this project we implement a walk-forward experiment to forecast one day conditional variance for equity index returns (default here is S&P 500 Total Return). We evaluate forecasts using loss based diagnostics and then use them in a daily volatility control backtest with transaction-cost sensitivity.

The objective is to compare common conditional volatility models as both statistical forecasts and daily risk control signals after accounting for exposure, turnover, and transaction costs.

## What we test specifically 

- **Target**: next day squared return, used as a noisy proxy for latent daily variance and annualized using 252 trading days. Let $r_t = \log(P_t / P_{t-1})$ be the daily log return. The target is:

$$\mathrm{Var}_{\mathrm{realized,ann}}(t) = 252r_t^2$$

- **Timing convention**: at forecast origin $t$, predictors use information available by the close of $t-1$ (no look-ahead).
- **Experiment design**: leakage-safe walk-forward forecasts evaluated over one common sample.
- **Models**: RiskMetrics EWMA, [HAR](https://doi.org/10.1016/j.jempfin.2008.07.001), GARCH, and GJR-GARCH. HAR uses daily, weekly, and monthly components of an OHLC range-variance proxy as predictors; the other models retain their standard return-based inputs. Every model forecasts the same squared-return target.
- **Baseline**: RiskMetrics EWMA with the fixed daily decay factor $\lambda=0.94$.
- **Diagnostics**: QLIKE is the primary variance-forecast loss. RMSE on the volatility scale and Spearman correlation are secondary measures of forecast magnitude and regime ordering. Diebold–Mariano tests describe uncertainty around the QLIKE differences versus EWMA over a HAC lag sensitivity grid.
- **Backtest**: a capped volatility-control strategy rebalanced daily and evaluated at 0, 1, and 5 bps of one-way cost per traded notional. Risk-adjusted performance is measured using excess returns over the cash proxy.


## Results

The primary output of this repository is the results notebook, which contains the tables and narrative for the forecast evaluation and strategy sensitivity analysis.

* **Main results notebook:** `notebooks/01_results.ipynb`

In the saved run, GJR has the lowest QLIKE. GARCH also keeps realized volatility below the 10% risk budget and has the highest excess Sharpe across cost levels, while EWMA has the lowest turnover and the shallowest drawdown. More accurate forecasts help avoid exceeding the volatility ceiling but do not consistently improve every volatility control outcome.

The source code in `src/vol_forecast/` is structured to keep the notebook thin. The notebook reports the model rankings under each forecast and strategy metric rather than selecting one model independently of the evaluation criterion.


## Quickstart

```bash
git clone https://github.com/MJG-12/vol-forecast-and-control.git
cd vol-forecast-and-control
python3 -m venv .venv
```

**Activate the virtual environment**

```powershell
. .\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

**Install**

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## How to run

The results notebook `notebooks/01_results.ipynb` is the single entry point. It runs the workflow step by step:

- Define the fixed `ExperimentSpec`.
- Build the canonical experiment dataframe (data loading + return construction + feature/target engineering).
- Run `compute_experiment_report(df, spec)` to fit the models over the common evaluation sample and produce the forecast and strategy tables.

Path: `src/vol_forecast/experiment.py` (`build_experiment_df`, `compute_experiment_report`)

## Repository layout

- `src/vol_forecast/`: Installable package (src layout)

  - `data.py`: Data loading and return calculation helpers.

  - `features.py`: One day target, baseline forecasts, and HAR components.

  - `models/`: HAR and GARCH-family walk-forward forecasters.

  - `evaluation.py`: QLIKE, headline forecast metrics, and DM tests.

  - `experiment.py`: Experiment orchestration (dataframe build + report computation).

  - `strategy.py`: Daily reset volatility control backtest utilities.

  - `config.py`: Fixed experiment and estimation-window settings.

- `notebooks/`: Results notebook


## Data sources

Equity index series: Loaded from Yahoo via `yfinance` (default: `^SP500TR`).

S&P 500 OHLC series used by HAR: Yahoo `^GSPC`.

Cash proxy: FRED `DFF`, converted to per-period simple returns using ACT/360 conventions and aligned with a 1-trading-day lag.
