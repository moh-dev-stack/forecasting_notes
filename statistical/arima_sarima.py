# arima_sarima.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
ARIMA / SARIMA (Box-Jenkins) toolkit — identification, fitting, diagnostics, forecasting.

This single-file gist gives a compact, production-quality set of utilities and
explanations for working with ARIMA and SARIMA models (Box-Jenkins approach). It is
designed to be saved as a gist and used directly.

What it provides (concise):
- stationarity tests: ADF and KPSS wrappers with plain-language interpretation
- differencing helpers (regular and seasonal) and a small stationarize pipeline
- lightweight ACF/PACF interpretation helper (heuristic p/P and q/Q suggestions)
- fit wrapper around statsmodels SARIMAX (ARIMA/SARIMA) with sensible defaults
- residual diagnostics including Ljung-Box, ACF-of-residuals plotting helper
- forecasting convenience that returns pandas Series
- CLI `--demo` / `--example` to run a small synthetic example end-to-end
- doctest snippets and unit-testable functions

Why:
- ARIMA family is the classic statistical forecasting toolkit. This file helps
  you go from raw series to a fitted SARIMAX with diagnostics and forecasts.

How:
- Use `stationarize()` to decide differencing (d, D), verify with `adf_test` and `kpss_test`,
  inspect ACF/PACF (function provided), choose candidate (p,d,q)(P,D,Q,s), fit with `fit_sarimax`,
  then validate residuals with `check_residuals`.

Notes / requirements:
- This module uses statsmodels for modeling and diagnostics. Install with:
    pip install statsmodels pandas numpy matplotlib
- pmdarima (auto_arima) is optionally supported if installed; the code will detect it and
  offer a wrapper but will not require it.
- Target Python: 3.11+

Example (CLI):
    python arima_sarima.py --demo

Doctest example:
    >>> import pandas as pd
    >>> s = pd.Series([1,2,3,4,5])
    >>> difference_series(s, 1).tolist()
    [1, 1, 1, 1]
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try required statsmodels imports, handle gracefully if missing.
try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_SM = True
except Exception:  # pragma: no cover - we will raise user-friendly errors where needed
    _HAS_SM = False

# Optional: pmdarima auto_arima convenience (not required)
try:
    import pmdarima as pm  # type: ignore
    _HAS_PMD = True
except Exception:
    _HAS_PMD = False

# Exports
__all__ = [
    "difference_series",
    "seasonal_difference",
    "stationarize",
    "adf_test",
    "kpss_test",
    "suggest_orders_from_acf_pacf",
    "fit_sarimax",
    "check_residuals",
    "forecast_sarimax",
    "demo_series",
    "KEY_TAKEAWAYS",
]


# -----------------------------------------------------------------------------
# Small typed helpers for clarity
# -----------------------------------------------------------------------------

def _ensure_statsmodels():
    """Raise a helpful error if statsmodels is not available."""
    if not _HAS_SM:
        raise RuntimeError(
            "This function requires statsmodels. Install with: pip install statsmodels"
        )


# -----------------------------------------------------------------------------
# 1) Differencing helpers (stationarity toolkit)
# -----------------------------------------------------------------------------

def difference_series(series: pd.Series, d: int = 1) -> pd.Series:
    """Return the d-th difference of a series.

    Why:
        Differencing is the simplest way to remove deterministic trend and help stationarity.

    Args:
        series: input 1-D series.
        d: number of differences (non-negative integer).

    Returns:
        differenced series of length len(series) - d with same index shifted.

    Doctest:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> difference_series(s, 1).tolist()
        [1, 1, 1, 1]
    """
    if d < 0:
        raise ValueError("d must be >= 0")
    if d == 0:
        return series.copy()
    res = series.copy()
    for _ in range(d):
        res = res.diff().dropna()
    return res


def seasonal_difference(series: pd.Series, D: int = 1, m: int = 1) -> pd.Series:
    """Apply seasonal differencing D times with period m.

    Args:
        series: input series.
        D: number of seasonal differences.
        m: seasonal period (e.g., 12 for monthly yearly seasonality).

    Returns:
        seasonally differenced series.

    Notes:
        Seasonal differencing removes repeating seasonal level offsets.
    """
    if D < 0:
        raise ValueError("D must be >= 0")
    if m < 1:
        raise ValueError("m must be >= 1")
    res = series.copy()
    for _ in range(D):
        res = (res - res.shift(m)).dropna()
    return res


def stationarize(series: pd.Series, max_d: int = 2, max_D: int = 1, m: int = 1) -> Tuple[int, int, pd.Series]:
    """Heuristic stationarization: try up to max_d regular differences and up to max_D seasonal diffs.

    Why:
        Helps propose (d, D) choices before deeper analysis.

    How:
        - Try combinations of D in 0..max_D and d in 0..max_d.
        - For each transformed series run ADF (null: non-stationary) and KPSS (null: stationary).
        - Prefer transforms where ADF rejects non-stationarity (p < 0.05) and KPSS does not reject (p > 0.05).
        - If multiple succeed prefer smallest d+D (parsimony).

    Returns:
        Tuple (d, D, transformed_series) where transformed_series is the differenced result.
    """
    _ensure_statsmodels()
    best = None
    candidates = []
    for D in range(max_D + 1):
        s_seas = seasonal_difference(series, D, m) if D > 0 else series.copy()
        for d in range(max_d + 1):
            try:
                s_reg = difference_series(s_seas, d) if d > 0 else s_seas.copy()
            except Exception:
                continue
            # Need enough points to test; skip too-short transforms
            if len(s_reg) < max(10, 3 * m):
                continue
            # Run tests
            adf_p = adfuller(s_reg.dropna(), autolag="AIC")[1]
            kpss_stat, kpss_p, _, _ = kpss(s_reg.dropna(), nlags="auto")
            # Accept if ADF p < 0.05 and KPSS p > 0.05 (stationary by both)
            accept = (adf_p < 0.05) and (kpss_p > 0.05)
            candidates.append((d, D, adf_p, kpss_p, accept, s_reg))
            if accept:
                # Prefer small d+D; choose the first acceptable (parsimony) 
                if best is None or (d + D) < (best[0] + best[1]):
                    best = (d, D, s_reg)
    if best is not None:
        return best
    # fallback: choose candidate with smallest adf_p (most evidence against unit root)
    if candidates:
        candidates.sort(key=lambda x: (x[2], -x[3]))  # prefer small adf_p and large kpss_p
        d, D, _, _, _, s_reg = candidates[0]
        return d, D, s_reg
    # If no candidate, return zeros and original series
    return 0, 0, series.copy()


# -----------------------------------------------------------------------------
# 2) Stationarity tests (wraps with friendly interpretation)
# -----------------------------------------------------------------------------

def adf_test(series: pd.Series, autolag: str = "AIC") -> Tuple[float, float, dict]:
    """Run Augmented Dickey-Fuller test and return (statistic, pvalue, critical_values).

    Interpretation:
        - Null hypothesis: the series has a unit root (non-stationary).
        - Small p-value (< 0.05) suggests stationarity (reject null).
    """
    _ensure_statsmodels()
    res = adfuller(series.dropna(), autolag=autolag)
    stat, pvalue, usedlag, nobs, crit_values, icbest = res
    return float(stat), float(pvalue), dict(crit_values)


def kpss_test(series: pd.Series, regression: str = "c") -> Tuple[float, float, dict]:
    """Run KPSS test and return (statistic, pvalue, critical_values).

    Interpretation:
        - Null hypothesis: the series is stationary.
        - Small p-value (< 0.05) suggests non-stationarity (reject null).
    """
    _ensure_statsmodels()
    stat, pvalue, lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    return float(stat), float(pvalue), dict(crit)


# -----------------------------------------------------------------------------
# 3) ACF / PACF-based order heuristics
# -----------------------------------------------------------------------------

def suggest_orders_from_acf_pacf(series: pd.Series, nlags: int = 40) -> Tuple[int, int]:
    """Heuristic suggestion for (p, q) based on PACF/ACF cutoff rules (non-automatic).

    Very simple heuristic:
      - p (AR order) ~ first PACF lag outside significance bounds
      - q (MA order) ~ first ACF lag outside significance bounds

    Returns:
        (p_suggest, q_suggest)

    Notes:
        - This is a crude rule-of-thumb. Use with plots and further testing.
        - For seasonal patterns look at seasonal lags (multiples of s) for seasonal P/Q.
    """
    _ensure_statsmodels()
    # compute ACF and PACF with confidence bounds (approx)
    series_nonan = series.dropna()
    acf_vals = acf(series_nonan, nlags=nlags, fft=True)
    pacf_vals = pacf(series_nonan, nlags=nlags, method="yw")  # Yule-Walker for stability

    # approximate 95% conf bound = 1.96/sqrt(N)
    N = len(series_nonan)
    bound = 1.96 / np.sqrt(N)
    p = 0
    q = 0
    # find first significant PACF lag > bound (starting at lag=1)
    for lag in range(1, len(pacf_vals)):
        if abs(pacf_vals[lag]) > bound:
            p = lag
            break
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) > bound:
            q = lag
            break
    return int(p), int(q)


# -----------------------------------------------------------------------------
# 4) Fit SARIMAX (ARIMA / SARIMA) wrapper
# -----------------------------------------------------------------------------

def fit_sarimax(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    exog: Optional[pd.DataFrame] = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    disp: bool = False,
    maxiter: int = 50,
) -> SARIMAXResults:
    """Fit a SARIMAX model with helpful defaults and return the fitted results.

    Args:
        series: observed series (pandas Series).
        order: (p, d, q) non-seasonal orders.
        seasonal_order: (P, D, Q, s) seasonal orders including seasonal period s.
        exog: optional exogenous regressors aligned with series index.
        enforce_stationarity / enforce_invertibility: flags to control estimation constraints.
        disp: whether the optimizer shows output.
        maxiter: optimizer max iterations.

    Returns:
        statsmodels SARIMAXResults object.

    Notes:
        - Use try/except around this call; optimizer may fail on some datasets.
        - Check res.mle_retvals for optimizer diagnostics if available.
    """
    _ensure_statsmodels()
    # Build and fit - keep verbose error messaging
    try:
        model = SARIMAX(
            endog=series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        res = model.fit(disp=disp, maxiter=maxiter)
    except Exception as e:
        raise RuntimeError(f"SARIMAX fit failed for order={order} seasonal_order={seasonal_order}: {e}") from e
    return res


# -----------------------------------------------------------------------------
# 5) Diagnostics: residual checks and plots
# -----------------------------------------------------------------------------

def check_residuals(res: SARIMAXResults, lags: int = 24, alpha: float = 0.05) -> dict:
    """Compute residual diagnostics: Ljung-Box p-values, residual ACF, and simple summaries.

    Returns a dict with:
      - 'lb_pvalue': p-value from Ljung-Box at specified lags (array)
      - 'resid_mean', 'resid_std'
      - 'acf': residual acf values (first `lags` lags)
      - 'white_noise': boolean: True if Ljung-Box p-values > alpha for all reported lags

    Interpretation:
      - Ljung-Box null: residuals are independently distributed (white noise).
      - We want p-values > alpha (fail to reject) to accept whiteness.
    """
    _ensure_statsmodels()
    resid = res.resid
    # Ljung-Box test across multiple lags; returns dataframe if boxpierce False
    lb_res = acorr_ljungbox(resid.dropna(), lags=[lags], return_df=True)
    lb_p = float(lb_res["lb_pvalue"].iloc[0])
    resid_mean = float(np.mean(resid))
    resid_std = float(np.std(resid, ddof=1))
    # compute acf
    acf_vals = acf(resid.dropna(), nlags=lags, fft=True)
    white_noise = lb_p > alpha
    return {
        "lb_pvalue": lb_p,
        "resid_mean": resid_mean,
        "resid_std": resid_std,
        "acf": acf_vals,
        "white_noise": white_noise,
    }


def plot_diagnostics(res: SARIMAXResults, lags: int = 24) -> None:
    """Plot observed vs fitted, residuals, and ACF of residuals for visual checks."""
    _ensure_statsmodels()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    # observed vs fitted
    try:
        axes[0].plot(res.data.endog, label="observed")
        axes[0].plot(res.fittedvalues, label="fitted", alpha=0.8)
        axes[0].legend()
        axes[0].set_title("Observed vs Fitted")
    except Exception:
        pass
    # residuals
    axes[1].plot(res.resid)
    axes[1].set_title("Residuals")
    # acf plot
    plot_acf(res.resid.dropna(), ax=axes[2], lags=lags)
    axes[2].set_title("ACF of residuals")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 6) Forecasting wrapper
# -----------------------------------------------------------------------------

def forecast_sarimax(res: SARIMAXResults, steps: int = 12, exog_future: Optional[pd.DataFrame] = None) -> pd.Series:
    """Produce point forecasts for `steps` ahead; returns pandas Series aligned to integer index.

    Notes:
        - If the model was fitted with a datetime index you may want to reindex the outputs
          to a datetime index before plotting. This helper returns a RangeIndex-based series
          for generality.
    """
    preds = res.get_forecast(steps=steps, exog=exog_future).predicted_mean
    # Convert to pandas Series and return
    if isinstance(preds, (pd.Series, pd.DataFrame)):
        return pd.Series(preds).squeeze()
    return pd.Series(preds)


# -----------------------------------------------------------------------------
# 7) A small synthetic demo generator and CLI example
# -----------------------------------------------------------------------------

def demo_series(periods: int = 200, seasonal_period: int = 12, seed: int = 0) -> pd.Series:
    """Generate synthetic series with level, drift and seasonal component for demos."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    level = 50 + 0.1 * t  # small upward drift
    seasonal = 5.0 * np.sin(2 * np.pi * (t % seasonal_period) / seasonal_period)
    noise = rng.normal(scale=1.5, size=periods)
    s = pd.Series(level + seasonal + noise)
    s.index = pd.RangeIndex(start=0, stop=periods)
    s.name = "y"
    return s


# -----------------------------------------------------------------------------
# 8) Optional convenience: pmdarima auto_arima wrapper (if available)
# -----------------------------------------------------------------------------

def auto_arima_suggestion(series: pd.Series, seasonal: bool = True, m: int = 1):
    """If pmdarima is installed, run auto_arima to suggest an order. Returns the model object.

    Notes:
        - This is a convenience only; we do not require pmdarima.
    """
    if not _HAS_PMD:
        raise RuntimeError("pmdarima is not installed. Install via `pip install pmdarima` to use auto_arima.")
    return pm.auto_arima(series, seasonal=seasonal, m=m, error_action="ignore", suppress_warnings=True)


# -----------------------------------------------------------------------------
# 9) Compact key takeaways for ARIMA modeling (copy-paste friendly)
# -----------------------------------------------------------------------------

KEY_TAKEAWAYS = """
Key takeaways — ARIMA / SARIMA (Box-Jenkins):

- ARIMA models are specified as (p, d, q)(P, D, Q, s) where:
    * p, q: non-seasonal AR and MA orders
    * d: non-seasonal differences required for stationarity
    * P, Q: seasonal AR and MA orders
    * D: seasonal differences
    * s: seasonal period (e.g., 12 for monthly, 7 for daily-weekly)

- Identification workflow (Box-Jenkins):
    1. Visualise series and assess need for differencing and seasonal differencing.
    2. Use ADF / KPSS tests to gather evidence about stationarity.
    3. Use ACF (for q) and PACF (for p) heuristics to suggest candidate orders.
    4. Fit candidate ARIMA/SARIMA models and compare AIC/BIC and out-of-sample CV.
    5. Validate residuals (white noise) with Ljung-Box and ACF; check forecast performance.

- Common pitfalls:
    * Over-differencing removes structure and increases variance.
    * Choosing multiplicative seasonal terms incorrectly (only for positive data).
    * Relying solely on automatic order selection; always inspect diagnostics and CV.

- Tools:
    statsmodels.tsa.statespace.SARIMAX for estimation and diagnostics; pmdarima for
    optional automated order selection.

"""

# -----------------------------------------------------------------------------
# 10) CLI: small demo runner (--demo)
# -----------------------------------------------------------------------------

def _run_demo(plot: bool = True) -> None:
    """Run a compact end-to-end demo: generate series, stationarize, suggest orders, fit, diagnose, forecast."""
    print("Generating demo series...")
    s = demo_series(periods=240, seasonal_period=12, seed=1)
    print(f"Series length: {len(s)}. Head:\n{s.head()}\n")

    print("Stationarizing heuristically (trying differences)...")
    d, D, s_trans = stationarize(s, max_d=2, max_D=1, m=12)
    print(f"Suggested differences: d={d}, D={D}. Transformed length: {len(s_trans)}")

    print("Running ACF/PACF heuristic for (p,q)...")
    p_sugg, q_sugg = suggest_orders_from_acf_pacf(s_trans, nlags=36)
    print(f"Suggested non-seasonal orders p={p_sugg}, q={q_sugg}")

    # choose seasonal orders heuristically: inspect seasonal lags (this demo chooses P=1,Q=1)
    P_sugg, Q_sugg, s_period = 1, 1, 12

    order = (p_sugg, d, q_sugg)
    seasonal_order = (P_sugg, D, Q_sugg, s_period)

    print(f"Fitting SARIMAX with order={order} seasonal_order={seasonal_order} ...")
    try:
        res = fit_sarimax(s, order=order, seasonal_order=seasonal_order, maxiter=200, disp=False)
        print("Fit completed. Summary (AIC,BIC):", getattr(res, "aic", None), getattr(res, "bic", None))
        diag = check_residuals(res, lags=24)
        print("Residual diagnostics:", diag)
        if plot:
            plot_diagnostics(res, lags=36)
            fc = forecast_sarimax(res, steps=24)
            print("Forecast (first 6):\n", fc.head(6))
            plt.figure(figsize=(8, 3))
            plt.plot(s.index, s, label="observed")
            plt.plot(np.arange(len(s), len(s) + len(fc)), fc, linestyle="--", label="forecast")
            plt.legend()
            plt.title("Observed and forecast")
            plt.show()
    except Exception as e:
        print("Model fit failed in demo:", e)
        print("Try adjusting orders or using auto_arima (pmdarima) for suggestions.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ARIMA/SARIMA quick toolkit demo")
    parser.add_argument("--demo", action="store_true", help="Run compact demo (synthetic series).")
    parser.add_argument("--no-plot", action="store_true", help="When running demo, disable plotting.")
    args = parser.parse_args()
    if args.demo:
        _run_demo(plot=not args.no_plot)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
