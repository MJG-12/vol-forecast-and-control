import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def qlike_series(
    realized_variance,
    forecast_variance,
    *,
    floor: float = 1e-12,
) -> np.ndarray:
    """Computes observation-level QLIKE for variance forecasts."""
    realized = np.asarray(realized_variance, dtype=float)
    forecast = np.asarray(forecast_variance, dtype=float)
    if realized.shape != forecast.shape:
        raise ValueError("realized and forecast variance must have equal shapes")

    realized = np.clip(realized, floor, np.inf)
    forecast = np.clip(forecast, floor, np.inf)
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


def forecast_metrics(
    panel: pd.DataFrame,
    *,
    target_col: str,
    baseline_col: str,
    forecast_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Computes QLIKE, RMSE, and Spearman correlation on one common sample."""
    realized = panel[target_col].to_numpy(dtype=float)
    realized_volatility = np.sqrt(np.clip(realized, 0.0, None))
    baseline_qlike = float(
        qlike_series(realized, panel[baseline_col].to_numpy(dtype=float)).mean()
    )

    rows = []
    for column in forecast_cols:
        forecast = panel[column].to_numpy(dtype=float)
        forecast_volatility = np.sqrt(np.clip(forecast, 0.0, None))
        qlike = float(qlike_series(realized, forecast).mean())
        rmse_vol = float(
            np.sqrt(
                np.mean(
                    (realized_volatility - forecast_volatility) ** 2
                )
            )
        )
        spearman_vol = float(
            pd.Series(forecast_volatility).corr(
                pd.Series(realized_volatility),
                method="spearman",
            )
        )
        rows.append(
            {
                "model": column,
                "n": len(panel),
                "qlike": qlike,
                "delta_qlike_vs_rolling": qlike - baseline_qlike,
                "rmse_vol": rmse_vol,
                "spearman_vol": spearman_vol,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("qlike")
        .reset_index(drop=True)
    )


def _newey_west_long_run_variance(values: np.ndarray, lag: int) -> float:
    """Computes Newey-West long-run variance with Bartlett weights."""
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    n = len(values)
    variance = float(values @ values / n)
    for offset in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(values[offset:] @ values[:-offset] / n)
        variance += 2.0 * weight * covariance
    return variance


def dm_tests(
    panel: pd.DataFrame,
    *,
    target_col: str,
    baseline_col: str,
    model_cols: tuple[str, ...],
    hac_lags: tuple[int, ...],
) -> pd.DataFrame:
    """
    Computes QLIKE Diebold-Mariano tests versus the rolling baseline.

    Tests are reported over a HAC lag grid. The loss differential is model
    QLIKE minus baseline QLIKE, so a negative mean indicates lower loss.
    """
    realized = panel[target_col].to_numpy(dtype=float)
    baseline_loss = qlike_series(
        realized,
        panel[baseline_col].to_numpy(dtype=float),
    )

    rows = []
    for hac_lag in hac_lags:
        for column in model_cols:
            model_loss = qlike_series(
                realized,
                panel[column].to_numpy(dtype=float),
            )
            differential = model_loss - baseline_loss
            mean_d = float(differential.mean())
            long_run_variance = _newey_west_long_run_variance(
                differential,
                hac_lag,
            )
            if (
                long_run_variance <= 0.0
                or not np.isfinite(long_run_variance)
            ):
                statistic = p_value = float("nan")
            else:
                statistic = mean_d / math.sqrt(
                    long_run_variance / len(panel)
                )
                p_value = float(2.0 * norm.sf(abs(statistic)))

            rows.append(
                {
                    "model": column,
                    "hac_lag": hac_lag,
                    "n": len(panel),
                    "mean_d": mean_d,
                    "dm_stat": statistic,
                    "p_value": p_value,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "hac_lag"])
        .reset_index(drop=True)
    )
