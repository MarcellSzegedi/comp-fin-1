"""Parameter validation functions."""

import datetime
from typing import Union

import yfinance as yf


def check_volatility_class_parameter(
    ticker: str,
    startdate: datetime.datetime,
    enddate: Union[datetime.datetime, str],
) -> None:
    """Checks the validity of various inputs of the volatility calculation class.

    Args:
        ticker: Ticker of the stock.
        startdate: Start date of the analysis.
        enddate: End date of the analysis.

    Raises:
        ValueError: in case any of the inputs are invalid.

    Returns:
        None
    """
    # Check if ticker is valid
    ticker_info = yf.Ticker(ticker).history(period="7d", interval="1d")
    if len(ticker_info) == 0:
        raise ValueError(f"No historical data found for {ticker}.")

    # Check if startdate is a datetime object
    if not isinstance(startdate, datetime.datetime):
        raise ValueError("'startdate' must be datetime object.")

    # Check if enddate is a datetime object
    if not isinstance(enddate, datetime.datetime):
        if enddate != "now":
            raise ValueError("'enddate' must be datetime object.")
        # Converting 'now' to datetime format
        enddate = datetime.datetime.now()

    # Check if startdate is before enddate
    if enddate.date() <= startdate.date():
        raise ValueError("End date must be larger than start date.")


def check_rolling_window_length(window_length: int, ts_len: int) -> None:
    """Checks if the window length is valid.

    Args:
        window_length: Length of the rolling window.
        ts_len: Length of the time series.

    Raises:
        ValueError: in case the window length is invalid.

    Returns:
        None
    """
    if not isinstance(window_length, int):
        raise ValueError("'window_length' must be an integer.")

    if window_length > ts_len:
        raise ValueError("'window_length' must be less than or equal to the market data length.")
