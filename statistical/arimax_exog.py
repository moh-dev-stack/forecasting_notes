# arimax_exog.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
ARIMAX (ARIMA with exogenous regressors) utilities — fit, validate, compare, and forecast.

This module provides clear, production-ready helpers to:
- prepare lagged exogenous features,
- validate exogenous availability at forecast time (avoid leakage),
- fit ARIMAX models using statsmodels' SARIMAX interface,
- run a simple rolling-origin comparison between ARIMA (no exog) and ARIMAX,
- produce forecasts given future exogenous covariates.

Why:
- Including exogenous variables (promotions, price, weather) often improves forecasts,
  but only when those regressors are available (or plausibly forecastable) for the
  forecast horizon. This module emphasizes safety (leakage checks) and auditability.

How:
- Use `fit_arimax(...)` to fit a SARIMAX model with exogenous regressors.
- Use `forecast_arimax(...)` to generate forecasts; you must provide exog values
  that line up with the forecast horizon.
- Use `rolling_origin_compare(...)` to get a horizon-wise comparison of ARIMA vs ARIMAX
  using a simple rolling-origin evaluation (small, readable implementation).

Notes:
- This file favors clarity and testability. For high-throughput experiments vectorized
  implementations are faster.
- Model-fitting requires `statsmodels`. Install:
    # Requires: statsmodels, pandas, numpy
    pip install statsmodels pandas numpy
- Target Python: 3.11+

Example (quick):
    >>> import pandas as pd
    >>> from arimax_exog import demo_series_with_exog, fit_arimax
    >>> y, ex = demo_series_with_exog(periods=100, seed=1)
    >>> res = fit_arimax(y.iloc[:80], ex.iloc[:80], order=(1,0,0))
    >>> isinstance(res, object)
    True
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels imports (deferred check)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_SM = True
except Exception:
    SARIMAX = None  # type: ignore
    SARIMAXResults = object  # type: ignore
    _HAS_SM = False

# Public API
__all__ = [
    "prepare_lagged_exog",
    "check_exog_forecastable",
    "fit_arimax",
    "forecast_arimax",
    "rolling_origin_compare",
    "rmse",
    "mae",
    "mape",
    "demo_series_with_exog",
    "KEY_TAKEAWAYS",
]


# --------------------------
# Utility: require statsmodels
# --------------------------
def _ensure_statsmodels() -> None:
    if not _HAS_SM:
        raise RuntimeError("statsmodels is required. Install via `pip install statsmodels`")


# --------------------------
# Simple error metrics
# --------------------------
def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_a = np.asarray(list(y_true), dtype=float)
    y_pred_a = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_a = np.asarray(list(y_true), dtype=float)
    y_pred_a = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true_a - y_pred_a)))


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_a = np.asarray(list(y_true), dtype=float)
    y_pred_a = np.asarray(list(y_pred), dtype=float)
    nonzero = y_true_a != 0
    if not np.any(nonzero):
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true_a[nonzero] - y_pred_a[nonzero]) / y_true_a[nonzero])))


# --------------------------
# Exogenous preparation helpers
# --------------------------
def prepare_lagged_exog(
    exog: pd.DataFrame,
    lags: Iterable[int],
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Create lagged versions of exogenous DataFrame.

    Why:
        Often regressors influence the target with a delay (promotions take effect after a few days).
        This helper creates lag columns like `feature_lag1`, `feature_lag2`, etc.

    Args:
        exog: DataFrame of exogenous variables, indexed like the target series.
        lags: iterable of positive integers representing lag amounts (1 = previous period).
        dropna: whether to drop rows with NaN created by lagging (default True).

    Returns:
        DataFrame with original columns and additional lagged columns. Column names:
        "<orig>_lag<k>".

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'promo':[0,1,0,0]}, index=pd.RangeIndex(0,4))
        >>> prepare_lagged_exog(df, [1]).columns.tolist()
        ['promo', 'promo_lag1']
    """
    if not isinstance(exog, pd.DataFrame):
        exog = pd.DataFrame(exog)
    df = exog.copy()
    for k in sorted(set(int(x) for x in lags)):
        if k <= 0:
            raise ValueError("lags must be positive integers")
        for col in exog.columns:
            df[f"{col}_lag{k}"] = exog[col].shift(k)
    if dropna:
        df = df.dropna()
    return df


def check_exog_forecastable(exog_future: Optional[pd.DataFrame], horizon: int) -> None:
    """
    Ensure exogenous covariates are provided for the forecast horizon.

    Why:
        A common leakage mistake is fitting ARIMAX with regressors that are
        not available at forecast time. This function enforces that exog_future
        is present for each forecasted period.

    Raises:
        ValueError if exog_future is missing or has insufficient rows.
    """
    if exog_future is None:
        raise ValueError("exog_future must be provided for ARIMAX forecasting to avoid leakage")
    if len(exog_future) < horizon:
        raise ValueError(f"exog_future must contain at least {horizon} rows (has {len(exog_future)})")


# --------------------------
# Fit / forecast wrappers
# --------------------------
def fit_arimax(
    endog: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    disp: bool = False,
    maxiter: int = 100,
) -> SARIMAXResults:
    """
    Fit ARIMAX (SARIMAX with exog) and return fitted results.

    Why:
        This wrapper centralises common options and provides a consistent interface.

    Args:
        endog: target series (pandas Series).
        exog: optional exogenous DataFrame aligned with endog (same index).
        order: non-seasonal (p,d,q).
        seasonal_order: seasonal (P,D,Q,s).
        enforce_stationarity/invertibility: controls estimation constraints.

    Returns:
        fitted SARIMAXResults object.

    Notes:
        - If statsmodels is not installed this raises a RuntimeError.
        - The user should still run diagnostics (see check_residuals below).
    """
    _ensure_statsmodels()
    if exog is not None and len(exog) != len(endog):
        raise ValueError("exog and endog must have the same length")
    try:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility)
        res = model.fit(disp=disp, maxiter=maxiter)
    except Exception as e:  # pragma: no cover - bubble up meaningful error
        raise RuntimeError(f"SARIMAX fit failed: {e}") from e
    return res


def forecast_arimax(
    res: SARIMAXResults,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Forecast with a fitted ARIMAX result.

    Must provide exog_future if the model was fitted with exog.

    Args:
        res: fitted SARIMAXResults (from fit_arimax).
        steps: number of steps to forecast.
        exog_future: exogenous DataFrame for the forecast horizon (shape: [steps, n_exog_cols]).

    Returns:
        pandas Series of point forecasts (RangeIndex by default).

    Raises:
        ValueError if exog was used at fit time but exog_future missing/incorrect.
    """
    _ensure_statsmodels()
    # Determine if exogenous were used when fitting
    used_exog = getattr(res.model, "k_exog", 0) > 0
    if used_exog:
        if exog_future is None:
            raise ValueError("This model was fitted with exogenous variables; exog_future must be provided for forecasting")
        if len(exog_future) < steps:
            raise ValueError("exog_future must have at least `steps` rows")
    pred = res.get_forecast(steps=steps, exog=exog_future)
    mean = pred.predicted_mean
    if isinstance(mean, (pd.Series, pd.DataFrame)):
        return pd.Series(mean).squeeze()
    return pd.Series(mean)


# --------------------------
# Rolling-origin comparison (simple, readable)
# --------------------------
def rolling_origin_compare(
    series: pd.Series,
    exog: Optional[pd.DataFrame],
    horizon: int = 1,
    initial_train_size: int = 50,
    step: int = 1,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pd.DataFrame:
    """
    Simple rolling-origin comparison: ARIMA (no exog) vs ARIMAX (with exog).

    Behavior (kept simple for readability):
      - Origins t vary from initial_train_size .. len(series) - horizon (inclusive).
      - For each origin:
          * train ARIMA on series[:t]
          * train ARIMAX on series[:t] with exog[:t] (if exog provided)
          * forecast horizon steps and compare to actuals
      - Returns a DataFrame with columns: ['origin', 'horizon', 'model', 'rmse', 'mae', 'mape']

    Important:
      - This routine is intentionally simple and not optimized for speed or for complex exog pipelines.
      - Exogenous data rows aligned with series index are required if exog is not None.
    """
    _ensure_statsmodels()
    n = len(series)
    if initial_train_size < 2:
        raise ValueError("initial_train_size must be >= 2")
    last_origin = n - horizon
    if initial_train_size > last_origin:
        raise ValueError("Not enough data for the requested initial_train_size and horizon")

    rows: List[Dict[str, object]] = []
    for t in range(initial_train_size, last_origin + 1, step):
        train_y = series.iloc[:t]
        test_y = series.iloc[t : t + horizon].reset_index(drop=True)
        # Fit ARIMA (no exog)
        arima_res = fit_arimax(train_y, exog=None, order=order, seasonal_order=seasonal_order, disp=False)
        arima_fc = forecast_arimax(arima_res, steps=horizon, exog_future=None).to_numpy()
        # Evaluate per-horizon
        for h_idx in range(horizon):
            rows.append(
                {
                    "origin": int(t),
                    "horizon": int(h_idx + 1),
                    "model": "ARIMA",
                    "rmse": rmse([test_y.iloc[h_idx]], [arima_fc[h_idx]]),
                    "mae": mae([test_y.iloc[h_idx]], [arima_fc[h_idx]]),
                    "mape": mape([test_y.iloc[h_idx]], [arima_fc[h_idx]]),
                }
            )
        # Fit ARIMAX if exog provided
        if exog is not None:
            train_ex = exog.iloc[:t]
            test_ex = exog.iloc[t : t + horizon]
            # If not enough exog rows to forecast, skip ARIMAX for this origin (safety)
            if len(test_ex) < horizon:
                # We record NaNs for clarity
                for h_idx in range(horizon):
                    rows.append(
                        {
                            "origin": int(t),
                            "horizon": int(h_idx + 1),
                            "model": "ARIMAX",
                            "rmse": float("nan"),
                            "mae": float("nan"),
                            "mape": float("nan"),
                        }
                    )
            else:
                arimax_res = fit_arimax(train_y, exog=train_ex, order=order, seasonal_order=seasonal_order, disp=False)
                arimax_fc = forecast_arimax(arimax_res, steps=horizon, exog_future=test_ex).to_numpy()
                for h_idx in range(horizon):
                    rows.append(
                        {
                            "origin": int(t),
                            "horizon": int(h_idx + 1),
                            "model": "ARIMAX",
                            "rmse": rmse([test_y.iloc[h_idx]], [arimax_fc[h_idx]]),
                            "mae": mae([test_y.iloc[h_idx]], [arimax_fc[h_idx]]),
                            "mape": mape([test_y.iloc[h_idx]], [arimax_fc[h_idx]]),
                        }
                    )
    df = pd.DataFrame(rows)
    return df


# --------------------------
# Small synthetic demo generator
# --------------------------
def demo_series_with_exog(periods: int = 200, seasonal_period: int = 12, seed: int = 0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Generate a synthetic target series and a single exogenous covariate (e.g., promotion intensity).

    Characteristics:
      - baseline trend + seasonal pattern + noise
      - exog is a binary promo indicator that increases the target with a lagged effect

    Returns:
        (y_series, exog_df)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    level = 50 + 0.05 * t
    seasonal = 5.0 * np.sin(2 * np.pi * (t % seasonal_period) / seasonal_period)
    # Generate a binary promotion schedule (random sparse promotions)
    promo = (rng.random(size=periods) < 0.05).astype(float)
    # Create a lagged effect on the target: effect shows up one period after promo
    y = level + seasonal + np.roll(promo * 8.0, 1) + rng.normal(scale=1.5, size=periods)
    # First element roll artifact set to 0
    y[0] -= promo[0] * 8.0
    y_series = pd.Series(y)
    exog_df = pd.DataFrame({"promo": promo}, index=y_series.index)
    return y_series, exog_df


# --------------------------
# Key takeaways (compact)
# --------------------------
KEY_TAKEAWAYS = """
Key takeaways — ARIMAX (regression with ARIMA errors) and exogenous variables:

- ARIMAX mixes time series dynamics (ARIMA) with the explanatory power of external regressors.
  Use when regressors are known at forecast time or can be reliably forecasted.

- Avoid leakage: regressors must be available (or forecasted independently) for the horizon
  you want to predict. Always validate the availability of exog_future before forecasting.

- Lagged exogenous features are common. Create them explicitly (prepare_lagged_exog) and ensure
  alignment between the target and features.

- Compare ARIMAX to a plain ARIMA baseline using the same CV protocol (rolling-origin) so you
  measure real added value from regressors.

- Check residuals (Ljung-Box) and plot ACF/PACF of residuals to ensure remaining autocorrelation
  is small. Significant residual autocorrelation suggests model misspecification.

- If exogenous covariates are many and high-dimensional, consider regularised regression (with
  time-series cross-validation) or a two-stage approach: feature engineering + ARIMA residual modelling.
"""

# --------------------------
# CLI / demo
# --------------------------
def _run_demo(plot: bool = True) -> None:
    print("Generating synthetic series and exogenous covariate...")
    y, ex = demo_series_with_exog(periods=240, seasonal_period=12, seed=2)
    # Keep example small for speed
    initial_train = 120
    horizon = 4
    print(f"Series length {len(y)}; initial_train={initial_train}, horizon={horizon}")

    # Prepare a simple lagged exog: promotion effect appears with lag=1 in synthetic data
    ex_lagged = prepare_lagged_exog(ex, lags=[1], dropna=False)
    # Align series with lagged exog (drop NaNs introduced by shift)
    combined = pd.concat([y, ex_lagged], axis=1)
    combined = combined.dropna().reset_index(drop=True)
    y_aligned = combined.iloc[:, 0]
    ex_aligned = combined.iloc[:, 1:]

    print("Running rolling-origin comparison (ARIMA vs ARIMAX)...")
    df = rolling_origin_compare(
        y_aligned,
        ex_aligned,
        horizon=horizon,
        initial_train_size=initial_train - 1,  # account for one-row shift maybe
        step=4,
        order=(1, 0, 0),
    )
    # Aggregate results by model and horizon
    if not df.empty:
        summary = df.groupby(["model", "horizon"])[["rmse", "mae", "mape"]].mean().unstack(level=0)
        print("Aggregated CV results (mean metrics):\n", summary)
    else:
        print("No results produced by rolling_origin_compare (check data sizes).")

    # Fit a final ARIMAX on full training and show short forecast
    train_end = len(y_aligned) - horizon
    y_train = y_aligned.iloc[:train_end]
    ex_train = ex_aligned.iloc[:train_end]
    ex_future = ex_aligned.iloc[train_end : train_end + horizon]
    print("Fitting final ARIMAX on training data...")
    try:
        res = fit_arimax(y_train, ex=ex_train, order=(1, 0, 0))
        fc = forecast_arimax(res, steps=horizon, exog_future=ex_future)
        print("Forecast (ARIMAX):\n", fc.to_string(index=False))
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 3))
            plt.plot(np.arange(len(y_aligned)), y_aligned, label="observed")
            plt.plot(np.arange(train_end, train_end + horizon), fc, linestyle="--", marker="o", label="forecast ARIMAX")
            plt.axvline(train_end, color="k", linestyle=":", label="forecast start")
            plt.legend()
            plt.title("ARIMAX forecast (demo)")
            plt.show()
    except Exception as e:
        print("Final ARIMAX fit/forecast failed:", e)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ARIMAX exog demo")
    parser.add_argument("--demo", action="store_true", help="Run demo (synthetic data).")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting in demo.")
    args = parser.parse_args()
    if args.demo:
        _run_demo(plot=not args.no_plot)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
