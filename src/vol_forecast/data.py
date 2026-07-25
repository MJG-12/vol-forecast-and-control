import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf


def _standardize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a sorted, duplicate-free DataFrame with a DatetimeIndex."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def load_sp500_total_return_close(
    start_date: str,
    end_date: str | None,
) -> pd.Series:
    """Loads the S&P 500 total-return index close from Yahoo."""
    data = yf.download(
        "^SP500TR",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="column",
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]
    data = _standardize_time_index(data)
    if data.empty or "Close" not in data:
        raise ValueError(
            f"Yahoo returned no S&P 500 total-return closes in "
            f"[{start_date}, {end_date}]"
        )

    close = pd.to_numeric(data["Close"], errors="coerce").astype(float)
    close = close.where(close > 0.0).dropna()
    close.name = "sp500_total_return_close"
    return close


def compute_log_returns(
    prices: pd.Series,
    *,
    name: str = "log_return",
) -> pd.Series:
    """Computes finite close-to-close log returns from positive prices."""
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = name
    return returns


def load_fred_series(
    series_id: str,
    start_date: str,
    end_date: str | None,
) -> pd.Series:
    """Loads one FRED series as a sorted float Series."""
    end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    data = _standardize_time_index(
        pdr.DataReader(series_id, "fred", start_date, end)
    )
    if series_id not in data:
        raise ValueError(
            f"FRED returned unexpected columns for {series_id!r}: "
            f"{list(data.columns)}"
        )
    series = pd.to_numeric(data[series_id], errors="coerce").astype(float)
    series = series.dropna()
    series.name = series_id
    return series


def cash_rate_to_daily_returns(
    rate_percent: pd.Series,
    trading_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Converts an annualized percentage rate to lagged ACT/360 simple returns.

    Return at t covers the calendar-day gap from the prior trading date to t
    using the rate available at the prior trading date.
    """
    lagged_rate = rate_percent.reindex(trading_index).ffill().shift(1)
    gap_days = trading_index.to_series().diff().dt.days.astype(float)
    returns = (lagged_rate / 100.0) * (gap_days / 360.0)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns.name = "cash_return"
    return returns


def load_cash_returns(
    start_date: str,
    end_date: str | None,
    trading_index: pd.DatetimeIndex,
) -> pd.Series:
    """Loads the FRED DFF cash proxy and aligns it to trading dates."""
    rate = load_fred_series("DFF", start_date, end_date)
    return cash_rate_to_daily_returns(rate, trading_index)
