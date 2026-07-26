from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """Material choices for the fixed volatility experiment."""

    data_start: str = "2004-01-01"
    data_end: str | None = "2026-02-06"
    rolling_window: int = 1000
    refit_every: int = 60
    hac_lags: tuple[int, ...] = (0, 5, 20)
    target_volatility: float = 0.10
    costs_bps: tuple[float, ...] = (0.0, 1.0, 5.0)

    def __post_init__(self) -> None:
        if self.target_volatility <= 0.0:
            raise ValueError("target_volatility must be positive")
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")
        if self.refit_every <= 0:
            raise ValueError("refit_every must be positive")
        if not self.hac_lags or any(lag < 0 for lag in self.hac_lags):
            raise ValueError("hac_lags must contain non-negative values")
        if not self.costs_bps or any(cost < 0.0 for cost in self.costs_bps):
            raise ValueError("costs_bps must contain non-negative values")
