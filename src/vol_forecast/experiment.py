import warnings

import pandas as pd

from vol_forecast.config import ExperimentSpec
from vol_forecast.data import (
    compute_log_returns,
    load_cash_returns,
    load_sp500_price_ohlc,
    load_sp500_total_return_close,
)
from vol_forecast.evaluation import dm_tests, forecast_metrics
from vol_forecast.features import (
    CASH_COL,
    EWMA_FORECAST_COL,
    HAR_VAR_FEATURES,
    RETURN_COL,
    TARGET_VAR_COL,
    build_features,
)
from vol_forecast.models.garch import walk_forward_garch
from vol_forecast.models.har import walk_forward_har
from vol_forecast.strategy import strategy_cost_table


HAR_FORECAST_COL = "har_forecast"
GARCH_FORECAST_COL = "garch_forecast"
GJR_FORECAST_COL = "gjr_forecast"
FORECAST_COLS = (
    EWMA_FORECAST_COL,
    HAR_FORECAST_COL,
    GARCH_FORECAST_COL,
    GJR_FORECAST_COL,
)


def build_experiment_df(
    spec: ExperimentSpec,
) -> pd.DataFrame:
    """Builds the canonical dataset from equity-index and cash-rate data."""
    prices = load_sp500_total_return_close(spec.data_start, spec.data_end)
    returns = compute_log_returns(prices, name=RETURN_COL)
    data = returns.to_frame().join(
        load_sp500_price_ohlc(spec.data_start, spec.data_end),
        how="left",
    )
    data[CASH_COL] = load_cash_returns(
        spec.data_start,
        spec.data_end,
        data.index,
    )
    data = build_features(data)
    return data


def fit_forecasts(
    data: pd.DataFrame,
    spec: ExperimentSpec,
) -> pd.DataFrame:
    """Fits HAR, GARCH, and GJR and warns about any refit failures."""
    har, har_diagnostics = walk_forward_har(
        data,
        feature_cols=HAR_VAR_FEATURES,
        target_col=TARGET_VAR_COL,
        rolling_window=spec.rolling_window,
        refit_every=spec.refit_every,
        output_name=HAR_FORECAST_COL,
    )
    garch, garch_diagnostics = walk_forward_garch(
        data,
        return_col=RETURN_COL,
        kind="garch",
        rolling_window=spec.rolling_window,
        refit_every=spec.refit_every,
        output_name=GARCH_FORECAST_COL,
    )
    gjr, gjr_diagnostics = walk_forward_garch(
        data,
        return_col=RETURN_COL,
        kind="gjr",
        rolling_window=spec.rolling_window,
        refit_every=spec.refit_every,
        output_name=GJR_FORECAST_COL,
    )

    failure_counts = {
        model: diagnostics["refit_failures"]
        for model, diagnostics in (
            (HAR_FORECAST_COL, har_diagnostics),
            (GARCH_FORECAST_COL, garch_diagnostics),
            (GJR_FORECAST_COL, gjr_diagnostics),
        )
        if diagnostics["refit_failures"] > 0
    }
    if failure_counts:
        warnings.warn(
            f"model refit failures occurred: {failure_counts}",
            RuntimeWarning,
            stacklevel=2,
        )
    return pd.concat([har, garch, gjr], axis=1)


def build_evaluation_panel(
    data: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Builds one complete common sample for forecasting and strategy results."""
    combined = data.join(forecasts)
    required = [
        RETURN_COL,
        CASH_COL,
        TARGET_VAR_COL,
        *FORECAST_COLS,
    ]
    eligible = combined.loc[combined[TARGET_VAR_COL].notna(), required]
    complete = eligible.notna().all(axis=1)
    if not complete.any():
        raise ValueError("evaluation sample is empty")
    first_complete = complete.loc[complete].index[0]
    candidates = eligible.loc[first_complete:]
    missing = candidates[required].isna().sum()
    missing = missing.loc[missing > 0]
    if not missing.empty:
        raise ValueError(
            "evaluation forecasts do not share a complete sample: "
            f"{missing.to_dict()}"
        )
    return candidates[required].copy()


def compute_experiment_report(
    data: pd.DataFrame,
    spec: ExperimentSpec,
) -> dict[str, object]:
    """Fits the fixed models and computes forecast and strategy results."""
    forecasts = fit_forecasts(data, spec)
    panel = build_evaluation_panel(data, forecasts)

    metrics = forecast_metrics(
        panel,
        target_col=TARGET_VAR_COL,
        baseline_col=EWMA_FORECAST_COL,
        forecast_cols=FORECAST_COLS,
    )
    dm = dm_tests(
        panel,
        target_col=TARGET_VAR_COL,
        baseline_col=EWMA_FORECAST_COL,
        model_cols=FORECAST_COLS[1:],
        hac_lags=spec.hac_lags,
    )
    strategy = strategy_cost_table(
        panel,
        return_col=RETURN_COL,
        cash_col=CASH_COL,
        signal_cols=FORECAST_COLS,
        baseline_col=EWMA_FORECAST_COL,
        target_volatility=spec.target_volatility,
        costs_bps=spec.costs_bps,
    )

    return {
        "forecast_metrics": metrics,
        "dm": dm,
        "strategy": strategy,
    }
