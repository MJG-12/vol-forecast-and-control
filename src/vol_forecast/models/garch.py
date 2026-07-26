from typing import Literal

from arch import arch_model
import numpy as np
import pandas as pd

from vol_forecast.features import TRADING_DAYS


GarchKind = Literal["garch", "gjr"]
RefitState = tuple[float, float, float, float, float, float]
PERSISTENCE_TOLERANCE = 1e-8


def _validated_refit_state(
    result,
    train: pd.Series,
    kind: GarchKind,
) -> RefitState | None:
    """Extracts a finite, non-explosive state from a successful `arch` fit."""
    omega = float(result.params.get("omega", np.nan))
    alpha = float(result.params.get("alpha[1]", np.nan))
    beta = float(result.params.get("beta[1]", np.nan))
    gamma = (
        float(result.params.get("gamma[1]", np.nan))
        if kind == "gjr"
        else 0.0
    )
    values = np.array([omega, alpha, beta, gamma], dtype=float)
    if not np.isfinite(values).all():
        return None

    h_previous = float(result.conditional_volatility.iloc[-1] ** 2)
    shock_previous = float(train.iloc[-1])
    persistence = alpha + beta + (0.5 * gamma if kind == "gjr" else 0.0)
    if (
        not np.isfinite(h_previous)
        or h_previous <= 0.0
        or not np.isfinite(shock_previous)
        or not 0.0 <= persistence <= 1.0 + PERSISTENCE_TOLERANCE
    ):
        return None

    return (
        omega,
        alpha,
        beta,
        gamma,
        h_previous,
        shock_previous,
    )


def walk_forward_garch(
    data: pd.DataFrame,
    *,
    return_col: str,
    kind: GarchKind,
    rolling_window: int,
    refit_every: int,
    output_name: str,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Produces rolling GARCH(1,1) or GJR-GARCH(1,1) variance forecasts.

    Forecasts are aligned at origin t using returns observed through t-1.
    Models use Student-t innovations and are refitted every `refit_every`
    origins on a fixed rolling window. If a refit fails, the last valid
    fitted state is retained and forecasting continues. Forecasting begins
    immediately after the first complete rolling window.

    Converged estimates at the unit-persistence boundary are accepted.
    Although these estimates do not have a finite unconditional variance,
    their one-step conditional variance forecasts remain well-defined.
    Persistence above one beyond numerical tolerance is rejected.

    Returns the forecast series and refit diagnostics.
    """
    if kind not in ("garch", "gjr"):
        raise ValueError("kind must be 'garch' or 'gjr'")

    returns = data[return_col].dropna().astype(float) * 100.0
    start = rolling_window
    forecasts = np.full(len(returns), np.nan)

    state: RefitState | None = None
    attempts = failures = 0

    for position in range(start, len(returns)):
        should_refit = (
            state is None
            or (position - start) % refit_every == 0
        )
        if should_refit:
            train = returns.iloc[position - rolling_window : position]
            attempts += 1
            model = arch_model(
                train,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1 if kind == "gjr" else 0,
                q=1,
                dist="t",
                rescale=False,
            )
            try:
                result = model.fit(disp="off")
            except (
                ValueError,
                FloatingPointError,
                RuntimeError,
                np.linalg.LinAlgError,
            ):
                failures += 1
            else:
                candidate = (
                    None
                    if getattr(result, "convergence_flag", 0) != 0
                    else _validated_refit_state(result, train, kind)
                )
                if candidate is None:
                    failures += 1
                else:
                    state = candidate

        if state is None:
            continue

        (
            omega,
            alpha,
            beta,
            gamma,
            h_previous,
            shock_previous,
        ) = state
        asymmetry = gamma if kind == "gjr" and shock_previous < 0.0 else 0.0
        h_today = (
            omega
            + (alpha + asymmetry) * shock_previous**2
            + beta * h_previous
        )
        if not np.isfinite(h_today) or h_today <= 0.0:
            continue

        forecasts[position] = (
            TRADING_DAYS * h_today / 100.0**2
        )

        shock_today = float(returns.iloc[position])
        state = (
            omega,
            alpha,
            beta,
            gamma,
            h_today,
            shock_today,
        )

    output = pd.Series(forecasts, index=returns.index, name=output_name)
    return output.reindex(data.index), {
        "refit_attempts": attempts,
        "refit_failures": failures,
    }
