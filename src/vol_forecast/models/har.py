import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_har(
    features: np.ndarray,
    target: np.ndarray,
):
    """Fits a HAR regression by ordinary least squares."""
    design = sm.add_constant(features, has_constant="add")
    return sm.OLS(target, design).fit()


def _predict_har(
    model,
    features: np.ndarray,
) -> np.ndarray:
    """Produces positive variance forecasts for a block of HAR predictors."""
    design = sm.add_constant(features, has_constant="add")
    forecast_variance = np.asarray(model.predict(design), dtype=float)
    return np.clip(forecast_variance, 1e-8, np.inf)


def walk_forward_har(
    data: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    target_col: str,
    rolling_window: int,
    refit_every: int,
    output_name: str,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Produces rolling one-day HAR variance forecasts.

    The regression uses daily, weekly, and monthly range-variance
    components available through t-1 to forecast squared return at t. It
    is refitted every `refit_every` origins on a fixed rolling window.

    Returns the forecast series and refit diagnostics.
    """
    required = [*feature_cols, target_col]
    valid = data[required].dropna()
    features = valid[list(feature_cols)].to_numpy(dtype=float)
    target = valid[target_col].to_numpy(dtype=float)
    forecasts = np.full(len(valid), np.nan)

    attempts = failures = 0
    for block_start in range(rolling_window, len(valid), refit_every):
        train_slice = slice(block_start - rolling_window, block_start)
        attempts += 1
        try:
            model = _fit_har(
                features[train_slice],
                target[train_slice],
            )
            block_end = min(block_start + refit_every, len(valid))
            forecasts[block_start:block_end] = _predict_har(
                model,
                features[block_start:block_end],
            )
        except (ValueError, np.linalg.LinAlgError):
            failures += 1

    output = pd.Series(forecasts, index=valid.index, name=output_name)
    return output.reindex(data.index), {
        "refit_attempts": attempts,
        "refit_failures": failures,
    }
