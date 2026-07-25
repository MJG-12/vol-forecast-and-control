import numpy as np
import pandas as pd
import pandas.testing as pdt

from vol_forecast.features import (
    CASH_COL,
    HAR_LOG_FEATURES,
    RETURN_COL,
    ROLLING_FORECAST_COL,
    TARGET_VAR_COL,
    build_features,
)


def test_predictors_do_not_use_future_returns() -> None:
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
    original = build_features(data, horizon=5)

    cutoff = index[50]
    changed_data = data.copy()
    changed_data.loc[index[51]:, RETURN_COL] *= 10.0
    changed = build_features(changed_data, horizon=5)

    predictor_cols = [ROLLING_FORECAST_COL, *HAR_LOG_FEATURES]
    pdt.assert_frame_equal(
        original.loc[:cutoff, predictor_cols],
        changed.loc[:cutoff, predictor_cols],
    )
    assert original.loc[cutoff, TARGET_VAR_COL] != changed.loc[
        cutoff,
        TARGET_VAR_COL,
    ]
