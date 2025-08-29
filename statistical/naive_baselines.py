# naive_baselines.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
Naïve and baseline forecasting utilities (Google-style docstring).

Focused update: `rolling_origin_evaluation` is simplified and better explained.
This file provides:
- three simple baseline forecasters: `naive_forecast`, `seasonal_naive_forecast`, `mean_forecast`
- basic error metrics: `rmse`, `mae`, `mape`
- a simplified, easy-to-understand `rolling_origin_evaluation` for comparing baselines
  across multiple forecast horizons using a rolling-origin protocol
- clear explanations (why/how/pros/cons) and a compact set of key takeaways at the end.

Why this file:
- Baselines are the minimum bar for forecasting. Always compute and archive them before
  experimenting with more complex models.

Notes:
- MAPE is unstable when true values are zero; prefer sMAPE or MASE for series with zeros.
- This module is intentionally dependency-light: it requires only numpy and pandas.

Example:
    >>> from naive_baselines import demo_series, rolling_origin_evaluation
    >>> s = demo_series(periods=120, seasonal_period=7, seed=0)
    >>> table = rolling_origin_evaluation(s, max_horizon=7, seasonal_period=7)
    >>> print(table)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Public API
__all__ = [
    "naive_forecast",
    "seasonal_naive_forecast",
    "mean_forecast",
    "rmse",
    "mae",
    "mape",
    "rolling_origin_evaluation",
    "demo_series",
    "KEY_TAKEAWAYS",
]


# -----------------------------------------------------------------------------
# Forecast generators (stateless and tiny)
# -----------------------------------------------------------------------------

# One-sentence comment: Return a horizon-length array repeating the last seen value.
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Naive forecast: repeat last observation.

    Args:
        train: training series (most recent observation is last).
        horizon: forecast horizon (>=1).

    Returns:
        numpy array of shape (horizon,) with repeated last value.

    Why:
        Extremely simple and often a surprisingly strong baseline for short horizons.
    """
    if len(train) == 0:
        raise ValueError("train series is empty")
    return np.repeat(float(train.iloc[-1]), horizon)


# One-sentence comment: For seasonal data, repeat the value from the same seasonal position.
def seasonal_naive_forecast(train: pd.Series, horizon: int, m: int) -> np.ndarray:
    """Seasonal naive: use value from last season position.

    Args:
        train: training series.
        horizon: forecast horizon.
        m: seasonal period (e.g., 7 for weekly seasonality on daily data).

    Returns:
        numpy array of seasonal-naive forecasts.

    Notes:
        - Requires len(train) >= m. Caller should ensure sufficient history.
    """
    n = len(train)
    if n < m:
        raise ValueError(f"need at least m={m} observations for seasonal_naive_forecast")
    # pick entries at positions n-m + (h-1) for h=1..horizon
    idxs = [n - m + (h - 1) for h in range(1, horizon + 1)]
    if any(i < 0 for i in idxs):
        raise ValueError("insufficient seasonal history for requested horizon")
    return train.iloc[idxs].astype(float).to_numpy()


# One-sentence comment: Constant forecast equal to the training mean.
def mean_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Mean forecast: constant forecast equal to the training mean."""
    if len(train) == 0:
        raise ValueError("train series is empty")
    return np.repeat(float(train.mean()), horizon)


# -----------------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------------

# One-sentence comment: Simple vectorized RMSE.
def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# One-sentence comment: Simple vectorized MAE.
def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# One-sentence comment: MAPE with basic zero-handling (skips zero true values).
def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error as percentage (not fraction).

    Behavior:
        - If any y_true == 0 those positions are ignored in the calculation.
        - If all y_true are zero returns NaN.

    Warning:
        - MAPE is not recommended when zeros or near-zeros are common.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    if not np.any(nonzero):
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))


# -----------------------------------------------------------------------------
# Simplified rolling-origin evaluation
# -----------------------------------------------------------------------------

# One-sentence comment: Simple, heavily-commented rolling-origin evaluation for clarity.
def rolling_origin_evaluation(
    series: pd.Series,
    max_horizon: int = 12,
    seasonal_period: int = 7,
    min_train_size: Optional[int] = None,
    methods: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Evaluate baseline forecasts using a simplified rolling-origin protocol.

    Why:
        Rolling-origin evaluation (sliding-origin) gives a robust estimate of out-of-sample
        performance by repeatedly forecasting from growing training windows. It approximates
        how a model would behave over many real forecasting occasions.

    How (step-by-step, implemented plainly and commented):
        - Determine a set of valid origins t where we can train on series[:t] and evaluate
          forecasts on the next `max_horizon` points series[t : t + max_horizon].
        - For each origin:
            * build simple forecasts (naive, mean, optionally seasonal-naive if history allows)
            * collect (actual, predicted) pairs for each horizon 1..max_horizon
        - After iterating origins, compute aggregated metrics (RMSE/MAE/MAPE) per horizon
          by averaging errors across all origins that produced a forecast for that method/horizon.

    Pros:
        - Simplicity and interpretability.
        - Produces horizon-wise metrics that reflect realistic forecasting deployments.

    Cons:
        - Requires enough historical data to support multiple origins and the chosen max_horizon.
        - Seasonal-naive will be undefined for origins without at least one seasonal cycle.
        - This implementation focuses on clarity; high-performance experiments should use
          vectorized or batched implementations.

    Args:
        series: full univariate series (values numeric). Index can be anything.
        max_horizon: maximum horizon H to evaluate (>=1). Evaluates horizons 1..H.
        seasonal_period: seasonal period m used by seasonal-naive.
        min_train_size: minimum training size to start origins. If None, defaults to max(2*m, 10).
        methods: subset of methods to evaluate; supported names: "naive", "seasonal", "mean".
                 Default: ("naive", "seasonal", "mean").

    Returns:
        pandas.DataFrame indexed by horizon (1..max_horizon) with columns:
            "{method}_rmse", "{method}_mae", "{method}_mape"
        Each cell is the mean metric across all valid origins for that (method, horizon).
    """
    # Default methods and validation -------------------------------------------------
    if methods is None:
        methods = ("naive", "seasonal", "mean")
    methods = tuple(methods)

    supported = {"naive", "seasonal", "mean"}
    for m in methods:
        if m not in supported:
            raise ValueError(f"unsupported method '{m}'. choose from {supported}")

    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")

    n = len(series)
    if n < 2:
        raise ValueError("series must contain at least 2 observations")

    # Default minimum training size ensures we have reasonable history for origins.
    if min_train_size is None:
        min_train_size = max(2 * seasonal_period, 10)

    # Determine the inclusive range of origins:
    # origin t indicates training = series[:t] and evaluation window = series[t : t+max_horizon]
    origin_start = int(min_train_size)
    origin_end = n - max_horizon  # inclusive: last origin must have full max_horizon after it

    # If origin_start > origin_end, we cannot run even a single origin with full horizon.
    if origin_start > origin_end:
        raise ValueError("not enough data for the requested min_train_size and max_horizon")

    # Prepare container: for each method we keep a list-of-lists for horizons 1..max_horizon.
    # pairs[method][h-1] will be a list of (true, pred) tuples collected across origins.
    pairs: Dict[str, List[List[Tuple[float, float]]]] = {
        method: [[] for _ in range(max_horizon)] for method in methods
    }

    # Loop over origins (clear, stepwise logic with comments)
    for t in range(origin_start, origin_end + 1):
        # TRAIN: historical data available up to but not including t
        train = series.iloc[:t].reset_index(drop=True)  # reset index so we can use integer positions safely
        # FUTURE: the next max_horizon points we want to predict and compare against
        future = series.iloc[t : t + max_horizon].reset_index(drop=True)  # length == max_horizon

        # Generate forecasts for each method if requested (None indicates unavailable)
        # Naive: always available (requires at least one training point)
        f_naive = naive_forecast(train, max_horizon) if "naive" in methods else None

        # Mean: always available (requires at least one training point)
        f_mean = mean_forecast(train, max_horizon) if "mean" in methods else None

        # Seasonal: only available when the training window includes at least one seasonal cycle
        f_seasonal = None
        if "seasonal" in methods:
            if len(train) >= seasonal_period:
                try:
                    f_seasonal = seasonal_naive_forecast(train, max_horizon, seasonal_period)
                except ValueError:
                    # defensive: if seasonal_naive raises for some reason, treat as unavailable
                    f_seasonal = None
            else:
                # Not enough history for seasonal naive at this origin; skip seasonal for this origin.
                f_seasonal = None

        # Collect (actual, predicted) pairs per horizon
        for h in range(1, max_horizon + 1):
            # Actual observed value at horizon h (1-indexed)
            true = float(future.iloc[h - 1])

            # Append pairs for each method if the forecast was produced for this origin
            if f_naive is not None:
                pairs["naive"][h - 1].append((true, float(f_naive[h - 1])))
            if f_mean is not None:
                pairs["mean"][h - 1].append((true, float(f_mean[h - 1])))
            if "seasonal" in methods and f_seasonal is not None:
                pairs["seasonal"][h - 1].append((true, float(f_seasonal[h - 1])))

    # Aggregate collected pairs into metrics per horizon
    horizons = list(range(1, max_horizon + 1))
    rows: List[List[float]] = []
    col_names: List[str] = []
    metrics = ("rmse", "mae", "mape")
    for method in methods:
        for metric in metrics:
            col_names.append(f"{method}_{metric}")

    for h_idx in range(max_horizon):
        row: List[float] = []
        for method in methods:
            data_pairs = pairs[method][h_idx]  # list of (true, pred) for this horizon across origins
            if len(data_pairs) == 0:
                # No valid origins produced a forecast for this (method, horizon).
                # This can happen for seasonal method when insufficient history exists.
                row.extend([float("nan")] * len(metrics))
                continue
            trues = np.array([p[0] for p in data_pairs], dtype=float)
            preds = np.array([p[1] for p in data_pairs], dtype=float)
            # Compute and append metrics in the chosen order
            row.append(rmse(trues, preds))
            row.append(mae(trues, preds))
            row.append(mape(trues, preds))
        rows.append(row)

    df = pd.DataFrame(rows, index=horizons, columns=col_names)
    df.index.name = "horizon"
    return df


# -----------------------------------------------------------------------------
# Small demo data generator (kept tiny)
# -----------------------------------------------------------------------------

# One-sentence comment: Produce a small synthetic series with trend+seasonality+noise for examples.
def demo_series(periods: int = 120, seasonal_period: int = 7, seed: int = 0) -> pd.Series:
    """Generate a demo series for quickly exercising baseline functions."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods, dtype=float)
    trend = 0.02 * t
    seasonal = 5.0 * np.sin(2 * math.pi * t / seasonal_period)
    noise = rng.normal(0, 1.0, size=periods)
    s = pd.Series(50.0 + trend + seasonal + noise)
    s.index = pd.RangeIndex(start=0, stop=periods, step=1)
    s.name = "value"
    return s


# -----------------------------------------------------------------------------
# Key takeaways (compact & copy-paste friendly)
# -----------------------------------------------------------------------------

# One-sentence comment: Bulleted summary of the most important practical points.
KEY_TAKEAWAYS = """
Key takeaways - Naïve baselines & rolling-origin evaluation:

- Always compute simple baselines (naive, seasonal-naive, mean) before building complex models.
  They are cheap, interpretable, and often surprisingly strong for short horizons.

- Use rolling-origin evaluation to measure out-of-sample performance across many origins.
  It estimates stability and variability of errors better than a single train/test split.

- Seasonal-naive requires at least one full seasonal cycle of history; skip or note results
  when history is insufficient.

- MAPE can be misleading when true values are zero or near-zero. Prefer sMAPE or MASE
  in those situations, or report multiple metrics (RMSE/MAE + a scale-free metric).

- Keep evaluation protocols identical when comparing models: same origins, same horizons,
  same metrics. Differences in CV protocol can easily dominate model improvements.

- Simplicity matters: the implementation here favors readability and correctness over
  micro-optimisations. For large scale experiments vectorised or streaming implementations
  may be preferable.

- Store baseline tables (the output DataFrame) alongside model artifacts so you can
  always compare new models to the original benchmark.

"""
# End of module.
