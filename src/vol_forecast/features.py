import numpy as np
import pandas as pd


TRADING_DAYS = 252

RETURN_COL = "log_return"
CASH_COL = "cash_return"
DAILY_VAR_COL = "daily_variance"
TRAILING_VAR_COL = "trailing_variance"
TARGET_VAR_COL = "forward_variance"
LOG_TARGET_COL = "log_forward_variance"
ROLLING_FORECAST_COL = "rolling_forecast"
HAR_LOG_FEATURES = (
    "log_variance_1d",
    "log_variance_5d",
    "log_variance_22d",
)


def forward_annualized_variance(
    daily_variance: pd.Series,
    horizon: int,
) -> pd.Series:
    """Computes annualized mean variance over t through t+h-1, aligned at t."""
    return (
        TRADING_DAYS
        * daily_variance.rolling(horizon).mean().shift(-(horizon - 1))
    )


def build_features(
    data: pd.DataFrame,
    *,
    horizon: int,
    log_floor: float = 1e-18,
) -> pd.DataFrame:
    """
    Adds the variance target, rolling baseline, and t-1 HAR predictors.

    Predictors at origin t use returns through t-1. The forward target uses
    returns from t through t+h-1 and is used only as a label.
    """
    if RETURN_COL not in data:
        raise ValueError(f"Missing required column: {RETURN_COL}")

    out = data.copy()
    daily_variance = out[RETURN_COL].astype(float).pow(2)
    out[DAILY_VAR_COL] = daily_variance

    trailing = TRADING_DAYS * daily_variance.rolling(horizon).mean()
    out[TRAILING_VAR_COL] = trailing
    out[TARGET_VAR_COL] = forward_annualized_variance(
        daily_variance,
        horizon,
    )
    out[ROLLING_FORECAST_COL] = trailing.shift(1)

    lagged_variance = daily_variance.shift(1)
    har_levels = (
        TRADING_DAYS * lagged_variance,
        TRADING_DAYS * lagged_variance.rolling(5).mean(),
        TRADING_DAYS * lagged_variance.rolling(22).mean(),
    )
    for column, values in zip(HAR_LOG_FEATURES, har_levels, strict=True):
        out[column] = np.log(values.clip(lower=log_floor))

    out[LOG_TARGET_COL] = np.log(
        out[TARGET_VAR_COL].clip(lower=log_floor)
    )
    return out
