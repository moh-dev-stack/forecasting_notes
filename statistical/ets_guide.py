```python
# ets_guide.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
Step-by-step guide to Exponential Smoothing and ETS models (Google style).

This module explains the intuition, math at a high level, practical pitfalls, and
how to fit ETS models using statsmodels' ExponentialSmoothing. It includes:
- clear explanations for Simple / Holt (double) / Holt-Winters (triple) exponential smoothing
- ETS notation and the state-space perspective in plain words
- guidance on choosing additive vs multiplicative components
- robust, minimal functions to fit a single ETS model and to run a small grid search
  over reasonable ETS variants
- diagnostics: residual checks and simple metrics
- a very simple, beginner-friendly plain-language explanation with a tiny numeric example

Why:
- ETS models are a powerful, light-weight family for level, trend, and seasonal structure.
  They are often competitive for many business time series and provide interpretable
  components and prediction intervals (approximate).

Notes:
- Requires statsmodels. Install with:
    pip install statsmodels
- Target Python: 3.11+
- The functions prefer a pandas Series with a DatetimeIndex or RangeIndex and a known seasonal_period.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    # statsmodels Holt-Winters implementation
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults
    _HAS_SM = True
except Exception:  # pragma: no cover - informative fallback if statsmodels not installed
    _HAS_SM = False

# Public API
__all__ = [
    "very_simple_ets_explanation",
    "explain_ets",
    "make_synthetic_seasonal",
    "fit_ets",
    "grid_search_ets",
    "forecast_from_result",
    "plot_ets_diagnostics",
    "KEY_TAKEAWAYS",
]


# -----------------------------------------------------------------------------
# Beginner-friendly explanation
# -----------------------------------------------------------------------------

def very_simple_ets_explanation() -> str:
    """Very simple, plain-language explanation of ETS for beginners.

    This function returns a short, easy-to-read explanation you can paste into a
    slide or read quickly. It uses a tiny numeric example to make the mechanics
    concrete without heavy math.

    Contents:
      - Intuition for SES, Holt, Holt-Winters
      - Why smoothing helps
      - Tiny numeric example for simple exponential smoothing update
      - Short analogies to remember each idea

    Example:
        >>> txt = very_simple_ets_explanation()
        >>> "Simple Exponential Smoothing" in txt
        True
    """
    return (
        "Very simple ETS explanation\n\n"
        "What ETS does in one sentence:\n"
        "  ETS models keep a running summary of the series (level), optionally track a "
        "trend (is it going up or down?), and optionally account for repeating patterns "
        "(seasonality). Predictions are made by propagating these components forward.\n\n"
        "Simple Exponential Smoothing (SES):\n"
        "  - Tracks only the level. Each new observation nudges the level a bit.\n"
        "  - Intuition: imagine your best guess for next value is your current 'level'. "
        "When a new observation arrives, you update the level by blending the old level "
        "and the new observation. The blending factor alpha (0 < alpha <= 1) controls how\n"
        "    quickly you forget the past. Large alpha -> adapt quickly; small alpha -> smooth.\n\n"
        "Holt (double) smoothing:\n"
        "  - Tracks level and trend. Trend is the estimated slope. Useful when the series "
        "is steadily rising or falling.\n\n"
        "Holt-Winters (triple) smoothing:\n"
        "  - Tracks level, trend, and seasonality. Use when series repeats patterns (daily, weekly, yearly).\n"
        "  - Seasonality can be additive (same size bumps) or multiplicative (bumps grow with level).\n\n"
        "Tiny numeric SES example (alpha = 0.5):\n"
        "  Suppose your initial level estimate is 10 and you observe 14 next.\n"
        "  New level = alpha * observed + (1-alpha) * old_level = 0.5*14 + 0.5*10 = 12.\n"
        "  Forecast for next step (one-step) = new level = 12.\n\n"
        "Why smoothing helps:\n"
        "  - It gives a stable short-term forecast that balances recent data and past history.\n"
        "  - It is simple, fast, interpretable, and often surprisingly effective.\n\n"
        "Analogies to remember:\n"
        "  - SES: a running average that pays more attention to recent frames of a movie.\n"
        "  - Holt: a running average plus a slope (like tracking both current speed and acceleration).\n"
        "  - Holt-Winters: adds the repeating soundtrack: level + slope + repeating beat.\n\n"
        "When to use ETS:\n"
        "  - Use SES if no trend or seasonality is visible.\n"
        "  - Use Holt if there is a clear linear trend.\n"
        "  - Use Holt-Winters if there is reproducible seasonality with known period.\n\n"
        "Caution:\n"
        "  - Multiplicative seasonality should only be used when the series is strictly positive.\n"
        "  - Always check residuals and test out-of-sample. Simple models are often a strong baseline.\n"
    )


# -----------------------------------------------------------------------------
# Explanations: what, why, how (concise, stepwise)
# -----------------------------------------------------------------------------

def explain_ets() -> str:
    """Return a compact, step-by-step explanation of ETS models and exponential smoothing.

    Contents (high level):
      1) Simple Exponential Smoothing (SES) - models level only.
      2) Holt (double) - level plus linear trend.
      3) Holt-Winters (triple) - level, trend, and seasonal component.
      4) ETS notation: ETS(Error, Trend, Seasonality) where each can be additive (A) or multiplicative (M) or none (N).
      5) Additive vs multiplicative: when amplitude changes with level prefer multiplicative.
      6) Practical fitting notes: seasonal_periods, initialization, bounded data, diagnostics.
    """
    return (
        "ETS step-by-step:\n\n"
        "1) Simple Exponential Smoothing (SES):\n"
        "   - Models only the level. New forecast is weighted average of last level and new observation.\n"
        "   - Use when series has no clear trend or seasonality.\n\n"
        "2) Holt (double) exponential smoothing:\n"
        "   - Adds a trend component (level and slope). Can be damped to prevent explosive forecasts.\n"
        "   - Use when there is a consistent linear trend.\n\n"
        "3) Holt-Winters (triple) exponential smoothing:\n"
        "   - Adds a seasonal component (additive or multiplicative) on top of level and optionally trend.\n"
        "   - Use when seasonality repeats with known period (weekly, monthly, quarterly).\n\n"
        "4) ETS notation: ETS(E, T, S) where E/T/S in {A, M, N} meaning additive, multiplicative, none.\n"
        "   Examples: ETS(A,A,A) = additive error, additive trend, additive seasonality.\n\n"
        "5) Additive vs multiplicative seasonality:\n"
        "   - Additive: seasonal amplitude roughly constant over time.\n"
        "   - Multiplicative: amplitude grows/shrinks with level. Do NOT use multiplicative if values can be zero or negative.\n\n"
        "6) Practical fitting:\n"
        "   - Provide seasonal_periods (m) based on frequency or domain knowledge.\n"
        "   - Use initialization_method='estimated' to let the optimizer pick start values.\n"
        "   - Check residuals for whiteness and unbiasedness; compare models with AIC and out-of-sample CV.\n"
    )


# -----------------------------------------------------------------------------
# Small synthetic data helper for examples
# -----------------------------------------------------------------------------

def make_synthetic_seasonal(
    periods: int = 4 * 12,
    seasonal_period: int = 12,
    trend_slope: float = 0.5,
    seasonal_amplitude: float = 10.0,
    noise_std: float = 2.0,
    multiplicative_seasonal: bool = False,
    seed: Optional[int] = 0,
) -> pd.Series:
    """Create a synthetic time series with level, trend, seasonal and noise components.

    Why:
        Synthetic examples let you try additive vs multiplicative fits with known ground truth.

    Args:
        periods: total number of time points.
        seasonal_period: period of seasonality (e.g., 12 for monthly data with yearly seasonality).
        trend_slope: linear trend slope per time step.
        seasonal_amplitude: amplitude of seasonal component (applies additively or multiplicatively).
        noise_std: standard deviation of additive Gaussian noise.
        multiplicative_seasonal: if True apply seasonality multiplicatively to level.
        seed: RNG seed for reproducibility.

    Returns:
        pandas Series of length `periods` with RangeIndex starting at 0.

    Example:
        >>> s = make_synthetic_seasonal(periods=24, seasonal_period=12, seed=1)
        >>> isinstance(s, pd.Series)
        True
    """
    rng = np.random.default_rng(seed)
    t = np.arange(periods, dtype=float)
    level = 50.0 + trend_slope * t  # baseline level + trend
    seasonal_base = np.sin(2 * np.pi * (t % seasonal_period) / seasonal_period)
    if multiplicative_seasonal:
        series = level * (1.0 + seasonal_amplitude / 100.0 * seasonal_base) + rng.normal(0, noise_std, size=periods)
    else:
        series = level + seasonal_amplitude * seasonal_base + rng.normal(0, noise_std, size=periods)
    s = pd.Series(series)
    s.name = "y"
    return s


# -----------------------------------------------------------------------------
# Fit a single ETS model cleanly with sensible defaults
# -----------------------------------------------------------------------------

def fit_ets(
    series: pd.Series,
    seasonal_periods: Optional[int] = None,
    trend: Optional[str] = None,
    seasonal: Optional[str] = None,
    damped_trend: bool = False,
    use_boxcox: Optional[Any] = None,
    remove_bias: bool = False,
    initialization_method: str = "estimated",
    optimize_kwargs: Optional[Dict[str, Any]] = None,
) -> HoltWintersResults:
    """Fit an ETS model using statsmodels' ExponentialSmoothing with recommended defaults.

    Why:
        Provide a small, well-documented wrapper to reduce common errors when calling the low-level API.

    How / choices explained:
      - seasonal_periods must be provided for seasonal models. It defines the cycle length.
      - trend: "add" or "mul" or None. Use "add" for linear additive trend.
      - seasonal: "add" or "mul" or None. Choose "mul" only when series > 0 and amplitude grows with level.
      - damped_trend: whether to use damped trend to prevent explosive long-term forecasts.
      - use_boxcox: False or True or a float parameter; Box-Cox can stabilise variance (careful with zeros).
      - initialization_method: using "estimated" usually works well. Alternatives: "heuristic", "legacy-heuristic".
      - optimize_kwargs: forwarded to .fit() optimizer for advanced control.

    Returns:
        A fitted statsmodels HoltWintersResults object.

    Raises:
        RuntimeError if statsmodels is not installed.
    """
    if not _HAS_SM:
        raise RuntimeError("statsmodels is required for fit_ets. Install via `pip install statsmodels`")

    if seasonal is not None and seasonal_periods is None:
        raise ValueError("seasonal_periods must be provided when seasonal is not None")

    if trend not in {None, "add", "mul"}:
        raise ValueError("trend must be one of {None, 'add', 'mul'}")
    if seasonal not in {None, "add", "mul"}:
        raise ValueError("seasonal must be one of {None, 'add', 'mul'}")

    # Default optimize kwargs: keep simple and allow user-supplied overrides
    opt_kwargs = {"method": "lbfgs"}
    if optimize_kwargs:
        opt_kwargs.update(optimize_kwargs)

    try:
        model = ExponentialSmoothing(
            series,
            trend=trend,
            damped_trend=damped_trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=initialization_method,
        )
        res = model.fit(optimized=True, use_boxcox=use_boxcox, remove_bias=remove_bias, **opt_kwargs)
    except Exception as e:
        raise RuntimeError(f"ETS fit failed: {e}") from e

    return res


# -----------------------------------------------------------------------------
# Small grid search over sensible ETS variants
# -----------------------------------------------------------------------------

def grid_search_ets(
    series: pd.Series,
    seasonal_periods: Optional[int],
    candidate_trends: Iterable[Optional[str]] = (None, "add"),
    candidate_seasonals: Iterable[Optional[str]] = (None, "add", "mul"),
    damped_options: Iterable[bool] = (False, True),
    max_models: int = 20,
) -> List[Tuple[str, HoltWintersResults]]:
    """Try a set of ETS variants and return fitted models sorted by AIC.

    Why:
        Model selection in ETS often starts by trying a small grid of trend/seasonal/damped options.
        This function automates that and returns models ranked by AIC.

    How:
        - Build combinations of trend x seasonal x damped.
        - Skip obviously invalid combos (e.g., multiplicative seasonal with nonpositive data).
        - Fit models and collect AIC. Return sorted results.

    Notes:
        - AIC is a useful model comparison metric but check residual diagnostics too.
    """
    if not _HAS_SM:
        raise RuntimeError("statsmodels is required for grid_search_ets. Install via `pip install statsmodels`")
    combos = list(itertools.product(candidate_trends, candidate_seasonals, damped_options))
    results: List[Tuple[str, HoltWintersResults]] = []
    tried = 0

    data_positive = (series > 0).all()

    for trend, seasonal, damped in combos:
        if tried >= max_models:
            break
        if seasonal == "mul" and not data_positive:
            continue
        label = f"trend={trend or 'N'}_seasonal={seasonal or 'N'}_damped={damped}"
        try:
            res = fit_ets(series, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, damped_trend=damped)
            results.append((label, res))
            tried += 1
        except Exception:
            continue

    def score(item: Tuple[str, HoltWintersResults]) -> float:
        _, r = item
        return getattr(r, "aic", float("inf"))

    results.sort(key=score)
    return results


# -----------------------------------------------------------------------------
# Forecast helper and simple diagnostics
# -----------------------------------------------------------------------------

def forecast_from_result(res: HoltWintersResults, steps: int = 12) -> pd.Series:
    """Produce point forecasts for `steps` future periods from a fitted result.

    Note:
        This function focuses on point forecasts. For intervals use result.get_prediction
        where supported or bootstrap residuals for empirical intervals.
    """
    preds = res.forecast(steps)
    if isinstance(res.data.endog, np.ndarray):
        start = len(res.data.endog)
        idx = pd.RangeIndex(start=start, stop=start + steps)
        return pd.Series(preds, index=idx, name="forecast")
    return pd.Series(preds, name="forecast")


def plot_ets_diagnostics(res: HoltWintersResults, plot_forecast_steps: int = 12) -> None:
    """Plot fitted values, residuals, and a short forecast to help with diagnostics.

    Why:
        Visual checks are essential: inspect fitted components, residuals, and forecast behavior.
    """
    series = res.data.endog if hasattr(res.data, "endog") else None
    fitted = res.fittedvalues if hasattr(res, "fittedvalues") else None
    resid = res.resid if hasattr(res, "resid") else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    if series is not None and fitted is not None:
        axes[0].plot(series, label="observed")
        axes[0].plot(fitted, label="fitted", alpha=0.8)
        axes[0].legend()
        axes[0].set_title("Observed vs fitted (ETS)")

    if resid is not None:
        axes[1].plot(resid)
        axes[1].set_title("Residuals (time series)")
        axes[2].hist(resid, bins=30)
        axes[2].set_title("Residuals histogram")

    try:
        fc = res.forecast(plot_forecast_steps)
        start = len(series) if series is not None else 0
        idx = np.arange(start, start + plot_forecast_steps)
        axes[0].plot(idx, fc, linestyle="--", label="forecast")
        axes[0].legend()
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Key takeaways - practical checklist and warnings
# -----------------------------------------------------------------------------

KEY_TAKEAWAYS = """
Key takeaways - ETS and exponential smoothing:

- Start simple: try SES, then add trend if a trend is visible, then add seasonality if clear cycles exist.
- ETS notation ETS(Error, Trend, Seasonality) with A=additive, M=multiplicative, N=none.
- Additive seasonality: use when seasonal amplitude is roughly constant in absolute terms.
  Multiplicative seasonality: use when amplitude scales with level. Do not use multiplicative
  if series can be zero or negative.
- Provide seasonal_periods (m) based on the data frequency and domain knowledge. For monthly data with yearly seasonality use m=12.
- Use initialization_method='estimated' so the optimizer finds starting values automatically.
- Compare candidate models using AIC and, more importantly, residual diagnostics and out-of-sample CV.
- Check residuals for autocorrelation (ACF) and bias. If residuals show structure consider ARIMA or hybrid methods.
- Consider Box-Cox transforms to stabilise variance before fitting ETS when variance depends on level.
- For probabilistic intervals either use prediction methods provided by your library or bootstrap residuals for empirical intervals.
- ETS is fast and interpretable. If accuracy is lacking consider ARIMA, regression with ARIMA errors, or machine learning approaches.
"""

# End of module.
```
