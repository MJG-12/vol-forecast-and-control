import numpy as np
import pandas as pd
import pandas.testing as pdt

from vol_forecast.features import (
    CASH_COL,
    EWMA_FORECAST_COL,
    HAR_VAR_FEATURES,
    PRICE_CLOSE_COL,
    PRICE_HIGH_COL,
    PRICE_LOW_COL,
    PRICE_OPEN_COL,
    RETURN_COL,
    TARGET_VAR_COL,
    build_features,
)


def test_predictors_do_not_use_future_data() -> None:
    index = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.Series(
        np.linspace(-0.02, 0.02, len(index)),
        index=index,
    )
    data = pd.DataFrame(
        {
            RETURN_COL: returns,
            CASH_COL: 0.0,
        }
    )
    price_close = 100.0 * np.exp(returns.cumsum())
    data[PRICE_CLOSE_COL] = price_close
    data[PRICE_OPEN_COL] = price_close.shift(1).fillna(price_close.iloc[0])
    data[PRICE_HIGH_COL] = data[
        [PRICE_OPEN_COL, PRICE_CLOSE_COL]
    ].max(axis=1) * 1.005
    data[PRICE_LOW_COL] = data[
        [PRICE_OPEN_COL, PRICE_CLOSE_COL]
    ].min(axis=1) * 0.995
    original = build_features(data)
    expected_third_forecast = 252 * (
        0.94 * returns.iloc[0] ** 2
        + 0.06 * returns.iloc[1] ** 2
    )
    np.testing.assert_allclose(
        original.loc[index[2], EWMA_FORECAST_COL],
        expected_third_forecast,
    )

    cutoff = index[50]
    changed_data = data.copy()
    changed_data.loc[cutoff:, RETURN_COL] *= 10.0
    changed_data.loc[cutoff:, PRICE_HIGH_COL] *= 1.10
    changed_data.loc[cutoff:, PRICE_LOW_COL] *= 0.90
    changed = build_features(changed_data)

    pdt.assert_series_equal(
        original.loc[:cutoff, EWMA_FORECAST_COL],
        changed.loc[:cutoff, EWMA_FORECAST_COL],
    )
    for column in HAR_VAR_FEATURES:
        pdt.assert_series_equal(
            original.loc[:cutoff, column],
            changed.loc[:cutoff, column],
        )
    assert original.loc[cutoff, TARGET_VAR_COL] != changed.loc[
        cutoff,
        TARGET_VAR_COL,
    ]
