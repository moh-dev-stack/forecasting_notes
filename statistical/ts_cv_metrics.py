```python
# ts_cv_metrics.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
Time-series error metrics and cross-validation utilities (Google-style docstring).

This module collects:
- Core error metrics for forecasting: RMSE, MAE, MAPE, sMAPE, MASE (with clear behavior).
- Simple baseline forecasters (naive, seasonal-naive, mean) for quick use.
- Two CV strategies: expanding-window and rolling-window (sliding) rolling-origin evaluation.
- A clear, easy-to-use `cross_val_forecast` orchestrator that runs CV, collects per-fold
  per-horizon metric values, and returns a tidy DataFrame suitable for analysis.

Why:
- Proper CV for time series must respect temporal order. This module implements simple,
  auditable CV routines and returns metric *distributions* (not only averages) so you can
  assess stability and horizon-dependent behavior.

How:
- `cross_val_forecast(...)` generates origins according to the chosen method, asks a
  forecaster callable to produce a horizon-length forecast from each training window,
  and computes the metrics per origin and horizon. Use the included baseline forecasters
  or pass any callable with signature `f(train_series, horizon) -> ndarray`.

When:
- Use these utilities whenever you need honest out-of-sample performance estimates for
  forecasting methods (baselines, classical models, or ML models).

Notes / limitations:
- This implementation favours clarity and auditability over micro-optimizations.
- For very large datasets or high-throughput experiments vectorized/batched implementations
  will be faster.
- MAPE and sMAPE have well-known caveats around zeros; see function docstrings.

Requires:
    pandas, numpy

Install:
    pip install pandas numpy
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Public API
__all__ = [
    "rmse",
    "mae",
    "mape",
    "smape",
    "mase",
    "naive_forecaster",
    "seasonal_naive_forecaster",
    "mean_forecaster",
    "cross_val_forecast",
    "aggregate_metric_distribution",
    "demo_series",
    "KEY_TAKEAWAYS",
]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error (RMSE).

    Why:
        Penalizes large errors (square) — useful when large misses are particularly bad.

    How:
        sqrt(mean((y_true - y_pred)^2))

    When:
        Use when you care about large deviations; report alongside MAE for perspective.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error (MAE).

    Why:
        Simple, interpretable average absolute error. Less sensitive to outliers than RMSE.

    How:
        mean(|y_true - y_pred|)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Percentage Error (MAPE), returned as percentage (e.g. 12.5 means 12.5%).

    Why:
        Scale-free and easy to communicate (percentage). Caveat: unstable when y_true near zero.

    How:
        mean(|(y_true - y_pred) / y_true|) * 100

    Behavior:
        - Entries where y_true == 0 are ignored. If all y_true are zero returns NaN.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    nonzero = yt != 0
    if not np.any(nonzero):
        return float("nan")
    return float(100.0 * np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE) in percent.

    Why:
        Attempts to be more symmetric than MAPE and handle small denominators more gracefully.

    How:
        200 * mean(|y_true - y_pred| / (|y_true| + |y_pred|))

    Behavior:
        - When both y_true and y_pred are zero for a point, that point contributes 0 to the mean.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp)
    # handle zeros: where denom == 0, define contribution as 0
    nonzero = denom != 0
    if not np.any(nonzero):
        return 0.0
    return float(200.0 * np.mean(np.abs(yt[nonzero] - yp[nonzero]) / denom[nonzero]))


def mase(y_true: Sequence[float], y_pred: Sequence[float], insample: Sequence[float], m: int = 1) -> float:
    """Mean Absolute Scaled Error (MASE).

    Why:
        Scale-free and interpretable relative to a naive (seasonal) in-sample benchmark.
        MASE = MAE_model / MAE_naive_insample. Values < 1 mean model beats naive.

    How:
        - Numerator: mean absolute error on the evaluation set.
        - Denominator: mean absolute in-sample naive one-step differences:
            denom = mean(|insample[t] - insample[t-m]|) for t = m..n-1
        - If denom == 0 returns NaN.

    When:
        Prefer MASE for a robust scale-free comparison to naive performance.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ins = np.asarray(insample, dtype=float)

    n = len(ins)
    if n <= m:
        # cannot compute denominator with insufficient insample length
        return float("nan")
    denom = np.mean(np.abs(ins[m:] - ins[:-m]))
    if denom == 0:
        return float("nan")
    num = np.mean(np.abs(yt - yp))
    return float(num / denom)


# -----------------------------------------------------------------------------
# Small baseline forecasters (callables for CV)
# -----------------------------------------------------------------------------

Forecaster = Callable[[pd.Series, int], np.ndarray]
# simple forecasters return a numpy array length = horizon


def naive_forecaster(train: pd.Series, horizon: int) -> np.ndarray:
    """Callable forecaster: repeat last training value."""
    if len(train) == 0:
        raise ValueError("train is empty")
    return np.repeat(float(train.iloc[-1]), horizon)


def seasonal_naive_forecaster(train: pd.Series, horizon: int, m: int = 1) -> np.ndarray:
    """Callable seasonal-naive forecaster. m=seasonal_period (default 1 -> equivalent to naive)."""
    n = len(train)
    if n < m:
        raise ValueError("insufficient history for seasonal naive")
    idxs = [n - m + (h - 1) for h in range(1, horizon + 1)]
    if any(i < 0 for i in idxs):
        raise ValueError("insufficient seasonal history for requested horizon")
    return train.iloc[idxs].astype(float).to_numpy()


def mean_forecaster(train: pd.Series, horizon: int) -> np.ndarray:
    """Callable forecaster that returns the training mean for every horizon."""
    if len(train) == 0:
        raise ValueError("train is empty")
    return np.repeat(float(train.mean()), horizon)


# -----------------------------------------------------------------------------
# Cross-validation orchestration
# -----------------------------------------------------------------------------

def _generate_origins(
    n_obs: int,
    initial_train_size: int,
    horizon: int,
    step: int = 1,
    method: str = "expanding",
    train_window: Optional[int] = None,
) -> List[int]:
    """Internal helper: returns list of training end indices (t) to act as origins.

    Interpretation:
        For an origin t, training data is series[:t] (Python slice end-exclusive).
        The test window for that origin is series[t : t + horizon].

    Parameters:
        n_obs: total length of series.
        initial_train_size: minimum length of the first training window (>=1).
        horizon: forecast horizon (>=1).
        step: step size between origins.
        method: 'expanding' (train grows) or 'rolling' (train is a sliding window).
        train_window: for 'rolling' method, the fixed training window length; if None uses initial_train_size.

    Returns:
        List of integers t (origin end positions) where t + horizon <= n_obs and t >= initial_train_size.
    """
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if method not in {"expanding", "rolling"}:
        raise ValueError("method must be 'expanding' or 'rolling'")

    if method == "rolling":
        window = train_window or initial_train_size
        if window < 1:
            raise ValueError("train_window must be >=1 for rolling")
        # origins are t such that train is [t-window, t) with t-window >= 0 and t + horizon <= n_obs
        t_min = window
    else:
        # expanding: train starts at 0 and grows; first origin has length initial_train_size
        t_min = initial_train_size

    t_max = n_obs - horizon  # inclusive maximum t
    if t_min > t_max:
        return []

    origins = list(range(t_min, t_max + 1, step))
    return origins


def cross_val_forecast(
    series: pd.Series,
    forecasters: Dict[str, Forecaster],
    horizon: int = 1,
    initial_train_size: int = 30,
    step: int = 1,
    method: str = "expanding",
    train_window: Optional[int] = None,
    seasonal_period: int = 1,
) -> pd.DataFrame:
    """Run rolling-origin cross-validation and return per-origin, per-horizon metric values.

    Why:
        Produces metric *distributions* (not only averages), enabling stability checks
        and horizon-dependent analysis.

    How (simplified):
        - Generate origins using `_generate_origins`.
        - For each origin t:
            * define train = series[:t], test = series[t : t + horizon]
            * for each named forecaster in `forecasters` call f(train, horizon)
            * compute metrics: RMSE, MAE, MAPE, sMAPE, MASE (MASE uses insample=train and seasonal_period)
            * append a row per (origin, horizon_step, forecaster_name, metric_name, value)
        - Return tidy DataFrame with columns:
            ['origin', 'horizon', 'forecaster', 'rmse', 'mae', 'mape', 'smape', 'mase']

    Args:
        series: pandas Series of numeric values (index may be datetime or RangeIndex).
        forecasters: mapping name -> callable(train_series, horizon) -> ndarray.
                     For seasonal_naive you can use a small wrapper to pass m.
        horizon: forecast horizon H (>=1) to evaluate (this function treats horizon as one-step window
                 with H steps; we compute metrics for each h in 1..H and return separate rows).
        initial_train_size: first training size in observations (>=1).
        step: shift between successive origins (>=1).
        method: 'expanding' or 'rolling'.
        train_window: integer for rolling window size (ignored for expanding).
        seasonal_period: seasonal period m used for MASE denominator and for seasonal_naive wrappers.

    Returns:
        pandas.DataFrame with one row per origin * per horizon * per forecaster containing the metrics.
        Columns: ['origin', 'horizon', 'forecaster', 'rmse', 'mae', 'mape', 'smape', 'mase']
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(series)
    if n < initial_train_size + horizon:
        # Not enough data for even a single origin
        raise ValueError("not enough data for the requested initial_train_size and horizon")

    origins = _generate_origins(n_obs=n, initial_train_size=initial_train_size, horizon=horizon, step=step, method=method, train_window=train_window)
    if not origins:
        raise ValueError("no valid origins generated; adjust initial_train_size / horizon / train_window")

    rows: List[Dict[str, object]] = []

    # iterate each origin
    for t in origins:
        train = series.iloc[:t]  # training window
        future = series.iloc[t : t + horizon].to_numpy(dtype=float)  # length == horizon

        # precompute denominator for MASE using training window and seasonal_period
        # denom = mean(|ins[m:] - ins[:-m]|)
        insarray = train.to_numpy(dtype=float)
        if len(insarray) > seasonal_period:
            denom = float(np.mean(np.abs(insarray[seasonal_period:] - insarray[:-seasonal_period])))
        else:
            denom = float("nan")  # will result in mase = NaN

        # for each forecaster produce forecast and compute metrics per horizon step
        for name, fn in forecasters.items():
            try:
                preds = fn(train, horizon)
            except Exception as e:
                # If a forecaster cannot produce forecast for this origin (e.g., seasonal naive), skip
                # but record NaNs to make downstream analysis simpler
                preds = np.array([np.nan] * horizon, dtype=float)

            preds = np.asarray(preds, dtype=float)
            # Ensure preds length matches horizon; if shorter/longer, truncate/pad with nan
            if preds.shape[0] < horizon:
                preds = np.concatenate([preds, np.full(horizon - preds.shape[0], np.nan)])
            elif preds.shape[0] > horizon:
                preds = preds[:horizon]

            # compute metrics per horizon step (1..H)
            for h_idx in range(horizon):
                true_val = float(future[h_idx])
                pred_val = float(preds[h_idx])
                # If pred is nan, metrics will be nan (we accept that)
                rm = rmse([true_val], [pred_val]) if not np.isnan(pred_val) else float("nan")
                ma = mae([true_val], [pred_val]) if not np.isnan(pred_val) else float("nan")
                mp = mape([true_val], [pred_val]) if not np.isnan(pred_val) else float("nan")
                sp = smape([true_val], [pred_val]) if not np.isnan(pred_val) else float("nan")
                # For MASE we need the insample denominator; mase can be computed on single-point evals but usually more meaningful aggregated.
                mase_val = float("nan")
                if not np.isnan(denom):
                    mase_val = mase([true_val], [pred_val], insample=insarray, m=seasonal_period)
                rows.append(
                    {
                        "origin": int(t),
                        "horizon": int(h_idx + 1),
                        "forecaster": name,
                        "rmse": rm,
                        "mae": ma,
                        "mape": mp,
                        "smape": sp,
                        "mase": mase_val,
                    }
                )

    df = pd.DataFrame(rows)
    # Order columns for readability
    df = df[["origin", "horizon", "forecaster", "rmse", "mae", "mape", "smape", "mase"]]
    return df


# -----------------------------------------------------------------------------
# Aggregation helpers and summaries
# -----------------------------------------------------------------------------

def aggregate_metric_distribution(cv_df: pd.DataFrame, agg: Iterable[str] = ("mean", "std")) -> pd.DataFrame:
    """Aggregate CV metric distributions by (forecaster, horizon).

    Returns a DataFrame indexed by (forecaster, horizon) with aggregated metric values
    for each metric and aggregation function supplied.

    Example:
        >>> dist = cross_val_forecast(...)
        >>> aggregate_metric_distribution(dist)
    """
    if cv_df.empty:
        return pd.DataFrame()

    # group by forecaster and horizon
    grouped = cv_df.groupby(["forecaster", "horizon"])

    # compute requested aggregations for each metric
    aggs = {}
    for metric in ("rmse", "mae", "mape", "smape", "mase"):
        if "mean" in agg:
            aggs[f"{metric}_mean"] = grouped[metric].mean()
        if "std" in agg:
            aggs[f"{metric}_std"] = grouped[metric].std()

    # combine into a single DataFrame
    result = pd.concat(aggs, axis=1)
    # flatten column names if necessary (already constructed as series)
    result.columns = list(result.columns)
    return result


# -----------------------------------------------------------------------------
# Demo data helper
# -----------------------------------------------------------------------------

def demo_series(periods: int = 240, seasonal_period: int = 7, seed: int = 0) -> pd.Series:
    """Small synthetic series (trend + seasonality + noise) for demoing CV functions."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods, dtype=float)
    trend = 0.01 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / seasonal_period)
    noise = rng.normal(0, 1.0, size=periods)
    s = pd.Series(20.0 + trend + seasonal + noise)
    s.index = pd.RangeIndex(start=0, stop=periods, step=1)
    s.name = "value"
    return s


# -----------------------------------------------------------------------------
# Key takeaways (concise)
# -----------------------------------------------------------------------------

KEY_TAKEAWAYS = """
Key takeaways — Error metrics & time-series CV:

- Choose metrics to match business priorities:
  * RMSE penalizes large misses; use when big errors are costly.
  * MAE is robust and interpretable (same units as data).
  * MAPE is intuitive (percent) but unstable near zero; report with caution.
  * sMAPE handles small denominators better than MAPE but has its own quirks.
  * MASE is a robust scale-free metric referenced to in-sample naive performance.

- Always respect temporal order when cross-validating. Use expanding-window or rolling-window
  (sliding) origins rather than random shuffles.

- Roll-forward CV returns metric distributions — inspect mean AND variability (std, quantiles)
  across origins and horizons to understand model stability.

- For seasonal series choose the seasonal_period carefully; MASE uses it to scale errors.

- Comparing models requires identical CV protocols (same origins, horizons, metrics).
  Differences in CV are a common source of misleading comparisons.

- Start with baselines (naive / seasonal-naive / mean) and use the same CV routine so improvements
  from complex models are meaningful.

"""

# End of module.
```
