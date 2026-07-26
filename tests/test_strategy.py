import numpy as np
import pandas as pd

from vol_forecast.strategy import (
    apply_transaction_costs,
    simulate_daily_policy,
    strategy_cost_table,
    strategy_statistics,
)


def test_strategy_weights_and_cost_monotonicity() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    log_returns = pd.Series(0.0, index=index)
    cash_returns = pd.Series(0.0, index=index)
    forecast_volatility = pd.Series(
        np.where(np.arange(len(index)) % 2 == 0, 0.10, 0.20),
        index=index,
    )

    gross, weights, turnover, _ = simulate_daily_policy(
        log_returns,
        cash_returns,
        forecast_volatility,
        target_volatility=0.10,
    )
    expected_weights = 0.10 / forecast_volatility
    np.testing.assert_allclose(weights, expected_weights)
    expected_gross = (
        weights * np.expm1(log_returns)
        + (1.0 - weights) * cash_returns
    )
    np.testing.assert_allclose(gross, expected_gross)
    assert turnover.iloc[1:].gt(0.0).all()

    net_zero = apply_transaction_costs(gross, turnover, cost_bps=0.0)
    net_one = apply_transaction_costs(gross, turnover, cost_bps=1.0)
    net_five = apply_transaction_costs(
        gross,
        turnover,
        cost_bps=5.0,
    )
    returns = [
        strategy_statistics(net, cash_returns)["annual_return"]
        for net in (net_zero, net_one, net_five)
    ]
    assert returns[0] >= returns[1] >= returns[2]

    panel = pd.DataFrame(
        {
            "log_return": np.tile([-0.01, 0.01], len(index) // 2),
            "cash_return": cash_returns,
            "ewma": forecast_volatility.pow(2),
            "model": forecast_volatility.pow(2),
        }
    )
    table = strategy_cost_table(
        panel,
        return_col="log_return",
        cash_col="cash_return",
        signal_cols=("ewma", "model"),
        baseline_col="ewma",
        target_volatility=0.10,
        costs_bps=(0.0, 1.0),
    )
    assert table["model"].eq("buy_and_hold").sum() == 1
    baseline_rows = table.loc[table["model"].eq("ewma")]
    np.testing.assert_allclose(
        baseline_rows["delta_sharpe_vs_ewma"],
        0.0,
    )
