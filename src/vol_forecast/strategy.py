import math

import numpy as np
import pandas as pd

from vol_forecast.features import TRADING_DAYS


def _drift_weight(
    weight: float,
    risky_return: float,
    cash_return: float,
) -> float:
    """Updates the risky weight after one period without rebalancing."""
    risky_value = weight * (1.0 + risky_return)
    cash_value = (1.0 - weight) * (1.0 + cash_return)
    total = risky_value + cash_value
    return weight if total <= 0.0 else float(risky_value / total)


def simulate_daily_policy(
    log_returns: pd.Series,
    cash_returns: pd.Series,
    forecast_volatility: pd.Series,
    *,
    target_volatility: float,
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Simulates a daily-reset gross strategy path and its turnover.

    Transaction costs are intentionally applied later because, under the
    linear cost model, they do not alter the policy's weights or trades.
    """
    if target_volatility <= 0.0:
        raise ValueError("target_volatility must be positive")

    frame = pd.concat(
        {
            "log_return": log_returns,
            "cash_return": cash_returns,
            "forecast_volatility": forecast_volatility,
        },
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("strategy inputs have no common observations")

    risky = np.expm1(frame["log_return"].to_numpy(dtype=float))
    cash = frame["cash_return"].to_numpy(dtype=float)
    forecast = frame["forecast_volatility"].to_numpy(dtype=float)
    target = np.clip(
        target_volatility / np.clip(forecast, 1e-8, np.inf),
        0.0,
        1.0,
    )

    n = len(frame)
    weights = target.copy()
    turnover = np.zeros(n)
    for position in range(1, n):
        pretrade_weight = _drift_weight(
            float(weights[position - 1]),
            risky[position - 1],
            cash[position - 1],
        )
        turnover[position] = abs(
            float(target[position]) - pretrade_weight
        )

    gross_returns = weights * risky + (1.0 - weights) * cash
    capped_fraction = float(np.mean(target >= 1.0 - 1e-12))
    index = frame.index
    return (
        pd.Series(gross_returns, index=index, name="gross_return"),
        pd.Series(weights, index=index, name="equity_weight"),
        pd.Series(turnover, index=index, name="turnover"),
        capped_fraction,
    )


def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_bps: float,
) -> pd.Series:
    """Applies linear one-way costs to a previously simulated gross path."""
    net = gross_returns - (float(cost_bps) / 10_000.0) * turnover
    net = net.clip(lower=-0.999999)
    net.name = "net_return"
    return net


def strategy_statistics(
    simple_returns: pd.Series,
    cash_returns: pd.Series,
) -> dict[str, float]:
    """Computes return, excess-Sharpe, risk, and drawdown statistics."""
    frame = pd.concat(
        {"strategy": simple_returns, "cash": cash_returns},
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("strategy statistics require non-empty returns")

    strategy = frame["strategy"].to_numpy(dtype=float)
    cash = frame["cash"].to_numpy(dtype=float)
    annual_return = float(
        np.expm1(TRADING_DAYS * np.log1p(strategy).mean())
    )
    realized_volatility = float(
        math.sqrt(TRADING_DAYS) * np.std(strategy, ddof=1)
    )

    excess = strategy - cash
    excess_std = float(np.std(excess, ddof=1))
    excess_sharpe = (
        float(math.sqrt(TRADING_DAYS) * np.mean(excess) / excess_std)
        if excess_std > 0.0
        else float("nan")
    )

    wealth = np.cumprod(1.0 + strategy)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "annual_return": annual_return,
        "excess_sharpe": excess_sharpe,
        "realized_volatility": realized_volatility,
        "max_drawdown": float(drawdown.min()),
    }


def strategy_cost_table(
    panel: pd.DataFrame,
    *,
    return_col: str,
    cash_col: str,
    signal_cols: tuple[str, ...],
    baseline_col: str,
    target_volatility: float,
    costs_bps: tuple[float, ...],
) -> pd.DataFrame:
    """Evaluates daily-reset strategies over a transaction-cost grid."""
    rows: list[dict[str, float | str]] = []

    buy_and_hold = np.expm1(panel[return_col])
    buy_and_hold_stats = strategy_statistics(
        buy_and_hold,
        panel[cash_col],
    )
    rows.append(
        {
            "cost_bps": float("nan"),
            "model": "buy_and_hold",
            **buy_and_hold_stats,
            "average_equity_weight": 1.0,
            "annual_turnover": 0.0,
            "pct_capped": float("nan"),
        }
    )

    for signal_col in signal_cols:
        forecast_volatility = np.sqrt(panel[signal_col].clip(lower=0.0))
        gross, weights, turnover, capped_fraction = simulate_daily_policy(
            panel[return_col],
            panel[cash_col],
            forecast_volatility,
            target_volatility=target_volatility,
        )
        path_values = {
            "average_equity_weight": float(weights.mean()),
            "annual_turnover": float(TRADING_DAYS * turnover.mean()),
            "pct_capped": capped_fraction,
        }
        for cost_bps in costs_bps:
            net = apply_transaction_costs(
                gross,
                turnover,
                cost_bps=cost_bps,
            )
            rows.append(
                {
                    "cost_bps": float(cost_bps),
                    "model": signal_col,
                    **strategy_statistics(net, panel[cash_col]),
                    **path_values,
                }
            )

    output = pd.DataFrame(rows)
    baseline_sharpe = (
        output.loc[
            output["model"].eq(baseline_col),
            ["cost_bps", "excess_sharpe"],
        ]
        .set_index("cost_bps")["excess_sharpe"]
        .to_dict()
    )
    output["delta_sharpe_vs_ewma"] = [
        (
            row.excess_sharpe
            - baseline_sharpe[row.cost_bps]
            if row.model != "buy_and_hold"
            else float("nan")
        )
        for row in output.itertuples()
    ]

    columns = [
        "cost_bps",
        "model",
        "annual_return",
        "excess_sharpe",
        "delta_sharpe_vs_ewma",
        "realized_volatility",
        "max_drawdown",
        "average_equity_weight",
        "annual_turnover",
        "pct_capped",
    ]
    return output[columns].sort_values(
        ["cost_bps", "excess_sharpe"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
