import numpy as np
import pandas as pd


TRADING_DAYS = 252
EWMA_DECAY = 0.94

RETURN_COL = "log_return"
CASH_COL = "cash_return"
PRICE_OPEN_COL = "price_open"
PRICE_HIGH_COL = "price_high"
PRICE_LOW_COL = "price_low"
PRICE_CLOSE_COL = "price_close"
DAILY_VAR_COL = "daily_variance"
RANGE_VAR_COL = "range_variance"
TARGET_VAR_COL = "realized_variance"
EWMA_FORECAST_COL = "ewma_forecast"
HAR_VAR_FEATURES = (
    "range_variance_1d",
    "range_variance_5d",
    "range_variance_22d",
)


def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the one-day target, EWMA baseline, and lagged HAR components.

    Predictors and baseline forecasts at origin t use information through t-1.
    The target is the annualized squared return at t and is used only as a
    label.
    """
    if RETURN_COL not in data:
        raise ValueError(f"Missing required column: {RETURN_COL}")

    out = data.copy()
    daily_variance = out[RETURN_COL].astype(float).pow(2)
    lagged_variance = daily_variance.shift(1)
    out[DAILY_VAR_COL] = daily_variance
    out[TARGET_VAR_COL] = TRADING_DAYS * daily_variance
    out[EWMA_FORECAST_COL] = (
        TRADING_DAYS
        * lagged_variance.ewm(
            alpha=1.0 - EWMA_DECAY,
            adjust=False,
        ).mean()
    )

    overnight_variance = np.log(
        out[PRICE_OPEN_COL] / out[PRICE_CLOSE_COL].shift(1)
    ).pow(2)
    range_variance = np.log(
        out[PRICE_HIGH_COL] / out[PRICE_LOW_COL]
    ).pow(2) / (4.0 * np.log(2.0))
    out[RANGE_VAR_COL] = TRADING_DAYS * (
        overnight_variance + range_variance
    )
    lagged_range_variance = out[RANGE_VAR_COL].shift(1)
    out[HAR_VAR_FEATURES[0]] = lagged_range_variance
    out[HAR_VAR_FEATURES[1]] = lagged_range_variance.rolling(5).mean()
    out[HAR_VAR_FEATURES[2]] = lagged_range_variance.rolling(22).mean()
    return out
