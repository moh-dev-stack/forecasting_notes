# time_series_foundations.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
Foundations of time series analysis: key concepts, why/how/when, examples and analogies.

This module covers three core foundational topics:
1. Time series vs cross-sectional data: definitions, why it matters, example and analogy.
2. Stationarity, trend, seasonality, noise: what each component means, when to expect them,
   how to detect or simulate them, short examples and analogies.
3. Basic plotting and quick checks: line plots, seasonal plots, rolling mean/std, and a simple
   stationarity check (ADF) when statsmodels is available.

Each conceptual section:
- gives a concise "why / how / when" explanation,
- provides a runnable example that generates a synthetic series,
- gives a short, teaching-friendly analogy.

Design notes:
- Target audience: intermediate-to-expert developers who want concise, actionable foundations
  to teach or to use in notebooks.
- Code is Python 3.11+ and uses type hints.
- Functions are small and unit-testable; plotting functions return the matplotlib Axes for testing.

Requires:
    pandas, matplotlib, statsmodels (statsmodels optional for ADF)
Install:
    pip install pandas matplotlib statsmodels
"""

from __future__ import annotations

# High-level overview:
# Keep functions focused: data generation, plotting, simple checks, and concise docstrings.
# Examples are self-contained and use synthetic series so they are safe to run in gists/notebooks.

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Attempt to import the ADF test. If unavailable, stationarity_check will advise installation.
try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore
    _HAS_ADF = True
except Exception:
    _HAS_ADF = False

# Public API
__all__ = [
    "SeriesComponents",
    "make_synthetic_series",
    "time_series_vs_cross_sectional",
    "explain_components",
    "plot_line",
    "plot_seasonal",
    "rolling_checks",
    "stationarity_check",
]


# -----------------------------------------------------------------------------
# Data classes and small utilities
# -----------------------------------------------------------------------------

# One-sentence comment: Encapsulate components so examples return structured information.
@dataclass(frozen=True)
class SeriesComponents:
    """Container for components returned by make_synthetic_series.

    Attributes:
        timestamps: DatetimeIndex used for the series.
        series: Combined series (trend + seasonal + noise).
        trend: Trend component series.
        seasonal: Seasonal component series.
        noise: Noise component series.
    """
    timestamps: pd.DatetimeIndex
    series: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    noise: pd.Series


# -----------------------------------------------------------------------------
# 1) Time series vs cross-sectional data
# -----------------------------------------------------------------------------

# One-sentence comment: Explain difference, why it matters, give an example and analogy.
def time_series_vs_cross_sectional() -> str:
    """Return a short explanation, example and analogy comparing time series and cross-sectional data.

    Why:
        Understanding the data structure is the first step. Forecasting requires ordered
        observations over time; cross-sectional methods typically assume i.i.d. samples.

    How:
        - Time series: one or more values observed at successive times (timestamps matter).
        - Cross-sectional: measurements at one point in time across many entities.

    When:
        - Use time series methods when order/temporal dependence matters; otherwise use
          cross-sectional tools (regressions, classification) that assume independence.

    Example (text):
        Time series: daily sales for a store over 3 years.
        Cross-sectional: a single snapshot of sales across 100 stores on the same day.

    Analogy:
        - Time series is a movie (frames ordered, story evolves).
        - Cross-sectional data is a photograph (one instant, many subjects).
    """
    return (
        "Time series: ordered observations where the timestamp and sequence matter. "
        "Cross-sectional: unordered observations at a single time point. "
        "Analogy: time series is a movie; cross-sectional is a photograph."
    )


# -----------------------------------------------------------------------------
# 2) Stationarity, trend, seasonality, noise - explanation + examples + analogies
# -----------------------------------------------------------------------------

# One-sentence comment: Explain each component with why/how/when plus code examples.
def explain_components() -> str:
    """Concise 'why / how / when' summary for stationarity, trend, seasonality, and noise.

    Why:
        Decomposing a series helps choose modeling approaches: differencing to remove trend,
        seasonal models for repeating patterns, and special treatment for heteroskedastic
        or autocorrelated noise.

    How:
        - Trend: persistent change in level over time. Detect by fitting a simple line or
          observing long-run movement.
        - Seasonality: repeating pattern with fixed period (daily, weekly, yearly).
          Detect by grouping by period and checking consistent cycles.
        - Noise: residual variation after removing trend and seasonality; often treated as
          white noise if uncorrelated.
        - Stationarity: statistical properties (mean, variance, autocorrelation) stable over time.
          Stationarity is required for many classical models like ARIMA.
    When:
        - If series shows drift, include trend/ differencing.
        - If repeating cycles exist, include seasonal terms or decomposition.
        - If residuals show structure, model errors (e.g., AR, GARCH).

    Example (text):
        See functions make_synthetic_series() and decomposition via plotting utilities.

    Analogy:
        - Trend is the river's slope; seasonality is the tide rising and falling; noise is the pebbles.
    """
    return (
        "Trend: long-term movement. Seasonality: repeating pattern at fixed period. "
        "Noise: random remainder. Stationarity: stable statistical properties. "
        "Analogy: river slope (trend), tide (seasonality), pebbles (noise)."
    )


def make_synthetic_series(
    periods: int = 365,
    freq: str = "D",
    trend_slope: float = 0.01,
    seasonal_amplitude: float = 10.0,
    noise_std: float = 2.0,
    seasonal_period: int = 7,
    seed: Optional[int] = 0,
) -> SeriesComponents:
    """Generate a synthetic time series with trend, seasonal pattern and Gaussian noise.

    Why:
        Synthetic data lets you demonstrate detection and visualization methods with
        known ground truth.

    How:
        - trend: linear increase over time.
        - seasonal: sinusoidal pattern repeated every seasonal_period.
        - noise: Gaussian random noise.

    When:
        - Use for teaching, unit tests and quick experiments before using real data.

    Args:
        periods: number of timestamps to generate.
        freq: frequency string used by pandas.date_range.
        trend_slope: slope of the linear trend per period.
        seasonal_amplitude: amplitude of the seasonal sine wave.
        noise_std: standard deviation of Gaussian noise.
        seasonal_period: number of periods in a season (e.g., 7 for weekly seasonality on daily data).
        seed: RNG seed for reproducibility.

    Returns:
        SeriesComponents containing timestamps and component series.

    Example:
        >>> comp = make_synthetic_series(periods=14, freq="D", seasonal_period=7, seed=1)
        >>> isinstance(comp.series, pd.Series)
        True
    """
    rng = np.random.default_rng(seed)
    # Build timestamps at the requested frequency.
    timestamps = pd.date_range(start="2020-01-01", periods=periods, freq=freq)

    t = np.arange(periods, dtype=float)

    # Trend: simple linear component.
    trend = trend_slope * t

    # Seasonal: periodic sine wave with the provided period; use phase to align peaks.
    seasonal = seasonal_amplitude * np.sin(2 * math.pi * t / seasonal_period)

    # Noise: Gaussian noise scaled by noise_std.
    noise = rng.normal(loc=0.0, scale=noise_std, size=periods)

    # Combine components into a pandas Series with datetime index.
    combined = pd.Series(data=trend + seasonal + noise, index=timestamps, name="value")
    trend_s = pd.Series(data=trend, index=timestamps, name="trend")
    seasonal_s = pd.Series(data=seasonal, index=timestamps, name="seasonal")
    noise_s = pd.Series(data=noise, index=timestamps, name="noise")

    return SeriesComponents(
        timestamps=timestamps, series=combined, trend=trend_s, seasonal=seasonal_s, noise=noise_s
    )


# -----------------------------------------------------------------------------
# 3) Plotting and quick checks: line plots, seasonal plots, rolling stats, ADF
# -----------------------------------------------------------------------------

# One-sentence comment: Basic line plot for quick visual inspection of trend/seasonality.
def plot_line(series: pd.Series, ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> plt.Axes:
    """Plot a time series line plot.

    Why:
        The first step in any time series analysis is visual inspection: see trend,
        breakpoints, missing data, and gross seasonal effects.

    How:
        Uses pandas plotting which integrates with Matplotlib and preserves datetime axis.

    Args:
        series: pandas Series with DatetimeIndex.
        ax: optional Matplotlib Axes to plot on.
        title: optional title text.

    Returns:
        The Axes containing the plot.

    Example:
        >>> comp = make_synthetic_series(periods=30, seasonal_period=7, seed=2)
        >>> ax = plot_line(comp.series)
        >>> hasattr(ax, "plot")
        True
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    # Plot the series with a thin line for clarity on long series.
    series.plot(ax=ax, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "value")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Time series (line plot)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax


# One-sentence comment: Seasonal plot that overlays period-aligned cycles (e.g., by month or weekday).
def plot_seasonal(series: pd.Series, period: str = "W", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot seasonal cycles by aggregating the series by a chosen period.

    Why:
        Seasonal plots reveal repeating structure by aligning comparable calendar positions.
        This is a quick alternative to STL decomposition for exploratory analysis.

    How:
        Group by the period label derived from the DatetimeIndex (month, weekday, hour)
        and plot each cycle on the same axis.

    Args:
        series: pandas Series with DatetimeIndex.
        period: aggregation period. Supported: "W" (weekday), "M" (month), "H" (hour of day), "D" (day of month).
                For daily data and weekly seasonality use "W" to group by weekday.
        ax: optional Matplotlib Axes.

    Returns:
        The Axes containing the seasonal overlay plot.

    Example:
        >>> comp = make_synthetic_series(periods=28, seasonal_period=7, seed=3)
        >>> ax = plot_seasonal(comp.series, period="W")
        >>> hasattr(ax, "lines")
        True
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    # Map the requested period to a grouping key function.
    if period == "W":
        # Weekday: 0=Monday ... 6=Sunday
        groups = series.groupby(series.index.weekday)
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    elif period == "M":
        groups = series.groupby(series.index.month)
        labels = [str(m) for m in range(1, 13)]
    elif period == "D":
        groups = series.groupby(series.index.day)
        labels = None
    elif period == "H":
        groups = series.groupby(series.index.hour)
        labels = [f"{h:02d}" for h in range(24)]
    else:
        raise ValueError("unsupported period; use one of 'W', 'M', 'D', 'H'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Plot each group's values normalized by the group's mean to compare shapes.
    for key, grp in groups:
        # Align by position within the cycle by using the group index converted to period-relative position.
        ax.plot(grp.index, grp.values, alpha=0.6, linewidth=1)
    ax.set_title("Seasonal overlay plot")
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "value")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax


# One-sentence comment: Rolling mean and rolling std to visually check for stationarity.
def rolling_checks(series: pd.Series, window: int = 12, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot series with rolling mean and rolling standard deviation.

    Why:
        If mean and variance change over time the series is likely non-stationary.
        Rolling statistics help reveal heteroskedasticity or level shifts.

    How:
        Compute rolling mean/std and overlay on the original series.

    Args:
        series: pandas Series with DatetimeIndex.
        window: rolling window size in periods.
        ax: optional Matplotlib Axes.

    Returns:
        The Axes containing the plot.

    Example:
        >>> comp = make_synthetic_series(periods=100, seasonal_period=7, seed=4)
        >>> ax = rolling_checks(comp.series, window=7)
        >>> hasattr(ax, "lines")
        True
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax, color="C0", linewidth=0.8, label="series")
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    rolling_mean.plot(ax=ax, color="C1", linewidth=1, label=f"rolling_mean({window})")
    ax.fill_between(series.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color="C1", alpha=0.1, label="rolling_std")
    ax.legend()
    ax.set_title("Rolling mean and standard deviation")
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "value")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax


# One-sentence comment: Simple stationarity check using the Augmented Dickey-Fuller test if available.
def stationarity_check(series: pd.Series) -> Tuple[bool, Optional[float], Optional[float]]:
    """Perform a quick stationarity check using the ADF test if statsmodels is installed.

    Why:
        Many classical forecasting models assume stationarity. The ADF test tests for
        a unit root under a null hypothesis of non-stationarity.

    How:
        Calls statsmodels.tsa.stattools.adfuller and interprets the p-value.

    Returns:
        (is_stationary, test_statistic, p_value)
        - is_stationary: True when p_value < 0.05 (reject null of unit root).
        - test_statistic, p_value: values returned by adfuller or None when unavailable.

    When:
        - Use as a guide not an absolute decision. Combine with visual checks and domain knowledge.

    Example:
        >>> comp = make_synthetic_series(periods=200, seasonal_period=7, seed=5)
        >>> stationarity_check(comp.series)  # type: ignore
        (False, -2.5, 0.11)  # Example output; actual numbers vary by RNG
    """
    if not _HAS_ADF:
        # Inform the caller that ADF is not available in this environment.
        raise RuntimeError("statsmodels is required for stationarity_check. Install via 'pip install statsmodels'")

    # adfuller returns (test_statistic, pvalue, usedlag, nobs, crit_values, icbest)
    adf_result = adfuller(series.dropna(), autolag="AIC")
    test_statistic = float(adf_result[0])
    p_value = float(adf_result[1])
    is_stationary = p_value < 0.05
    return is_stationary, test_statistic, p_value


# -----------------------------------------------------------------------------
# Small teaching examples collected as helpers (no top-level execution)
# -----------------------------------------------------------------------------

# One-sentence comment: Example snippets to embed in a notebook or gist to demonstrate concepts quickly.
def example_snippets() -> dict:
    """Return short examples (pandas Series) demonstrating common patterns.

    Why:
        Provide ready data for copy-paste into a notebook to illustrate plotting and checks.

    Contents:
        - short_weekly: 28 days with weekly seasonality and upward trend
        - long_series: 3 years daily series for rolling/stationarity demos

    Returns:
        dict mapping names to SeriesComponents
    """
    short_weekly = make_synthetic_series(periods=28, freq="D", trend_slope=0.02, seasonal_amplitude=5.0, noise_std=1.0, seasonal_period=7, seed=10)
    long_series = make_synthetic_series(periods=365 * 3, freq="D", trend_slope=0.001, seasonal_amplitude=20.0, noise_std=3.0, seasonal_period=365 // 12, seed=11)
    return {"short_weekly": short_weekly, "long_series": long_series}


# -----------------------------------------------------------------------------
# Compact teaching-style analogies (string helpers)
# -----------------------------------------------------------------------------

# One-sentence comment: Return short analogies to help teach the components verbally.
def analogies() -> dict:
    """Return easy teaching analogies for core concepts.

    Entries:
        - time_vs_cross: movie vs photograph
        - trend: river slope
        - seasonality: tide or clockwork
        - noise: pebbles on the shore
    """
    return {
        "time_vs_cross": "Time series is a movie; cross-sectional data is a photograph.",
        "trend": "Trend is the river's slope pushing water steadily in one direction.",
        "seasonality": "Seasonality is the tide rising and falling with a fixed rhythm.",
        "noise": "Noise is the pebbles and splashes that vary unpredictably on the shore.",
    }


# -----------------------------------------------------------------------------
# End of module
# -----------------------------------------------------------------------------

# Notes:
# - No top-level execution; this file is meant to be saved as a gist and imported into
#   a notebook. Use the example_snippets() helper to quickly obtain SeriesComponents for demos.
#
# - For production EDA, replace prints/plots with structured logging and figures saved to files.
#
# - Recommended next steps after this module:
#   * perform STL decomposition (statsmodels.tsa.seasonal.STL) for robust seasonal extraction,
#   * compute ACF/PACF plots for ARIMA identification,
#   * apply rolling-origin cross-validation for forecasting experiments.
