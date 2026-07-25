import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_log_har(
    features: np.ndarray,
    target: np.ndarray,
):
    """Fits OLS and returns the model plus residual variance."""
    design = sm.add_constant(features, has_constant="add")
    model = sm.OLS(target, design).fit()
    return model, float(model.mse_resid)


def _predict_log_har(
    model,
    sigma2: float,
    features: np.ndarray,
) -> np.ndarray:
    """Produces variance forecasts for a block of HAR predictors."""
    design = sm.add_constant(features, has_constant="add")
    log_forecast = np.asarray(model.predict(design), dtype=float)
    return np.exp(log_forecast + 0.5 * sigma2)


def walk_forward_har(
    data: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    log_target_col: str,
    horizon: int,
    rolling_window: int,
    refit_every: int,
    output_name: str,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Produces leakage-safe rolling HAR forecasts on the variance scale.

    Forecasts are aligned at origin t. Predictors use information through
    t-1, while the label is forward variance over t through t+h-1.

    The model is refitted every `refit_every` origins using a fixed rolling
    window and a purged training cutoff. If a refit fails, the last valid
    fit is retained and forecasting continues. Forecasting begins at the
    first origin that leaves one complete purged rolling window.

    Log forecasts are mapped to variance using the lognormal mean
    exp(mu + 0.5 * sigma2), where sigma2 is the OLS residual variance.

    Returns the forecast series and refit diagnostics.
    """
    required = [*feature_cols, log_target_col]
    valid = data[required].dropna()
    start = rolling_window + horizon - 1
    feature_values = valid[list(feature_cols)].to_numpy(dtype=float)
    target_values = valid[log_target_col].to_numpy(dtype=float)
    forecasts = np.full(len(valid), np.nan)

    model = None
    sigma2 = float("nan")
    attempts = failures = 0

    for block_start in range(start, len(valid), refit_every):
        # A label at origin j uses j..j+h-1, so the last admissible
        # training origin is block_start-horizon.
        train_end = block_start - horizon + 1
        train_slice = slice(train_end - rolling_window, train_end)

        attempts += 1
        try:
            new_model, new_sigma2 = _fit_log_har(
                feature_values[train_slice],
                target_values[train_slice],
            )
            if not np.isfinite(new_sigma2) or new_sigma2 < 0.0:
                raise ValueError("invalid HAR residual variance")
            model, sigma2 = new_model, new_sigma2
        except (ValueError, np.linalg.LinAlgError):
            failures += 1

        if model is None:
            continue

        block_end = min(block_start + refit_every, len(valid))
        forecasts[block_start:block_end] = _predict_log_har(
            model,
            sigma2,
            feature_values[block_start:block_end],
        )

    output = pd.Series(forecasts, index=valid.index, name=output_name)
    return output.reindex(data.index), {
        "refit_attempts": attempts,
        "refit_failures": failures,
    }
