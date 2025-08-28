# time_vs_cross_examples.py
# Author: Mohsin Zafar (example)
# Copyright: public domain (adapt as needed)

"""
Concrete examples: Time series vs Cross-sectional data (movie vs photograph).

This module provides small, runnable examples and concise explanation to make the
difference between time series and cross-sectional data intuitive.

For each concept we show:
- Why it matters (short).
- How to construct a minimal synthetic example.
- When to use time-series methods vs cross-sectional methods.
- A short analogy and a small code example you can run in a notebook.

Between the "Time series" and "Cross-sectional" sections below is a compact
markdown table to help compare them at-a-glance.

| Property                 | Time series                        | Cross-sectional                     |
|-------------------------:|-----------------------------------:|------------------------------------:|
| Ordered observations     | Yes, order and timestamps matter   | No, order is irrelevant             |
| Dependence between rows  | Often autocorrelated                | Typically assumed independent (iid) |
| Typical questions        | What happens next? Forecasting      | What explains differences now?      |
| Typical models           | ARIMA, ETS, state-space, ML seq.   | OLS, classification, cross-section regressions |
| Validation style         | Time-aware CV, rolling windows     | Randomized CV, stratified splits    |
| Use cases                | Sales over time, sensor streams    | Survey at a time, customer snapshot |

Requires:
    pandas, numpy, matplotlib, statsmodels (statsmodels optional for OLS/ADF examples)

Install:
    pip install pandas numpy matplotlib statsmodels
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# statsmodels is optional; used to show a simple OLS on cross-sectional data.
try:
    import statsmodels.api as sm  # type: ignore
    _HAS_SM = True
except Exception:
    _HAS_SM = False

__all__ = [
    "make_time_series_example",
    "make_cross_section_example",
    "compare_examples",
    "analogy",
    "analogies",
    "textual_examples",
    "explain_order_irrelevance_cross_section",
    "demonstrate_shuffling_equivalence",
]


# -----------------------------------------------------------------------------
# Analogy helper
# -----------------------------------------------------------------------------

# One-sentence comment: Return teaching analogies used throughout notebooks.
def analogy() -> str:
    """Return the one-line analogy: movie vs photograph."""
    return "Time series is a movie (frames in order). Cross-sectional data is a photograph (many subjects at one time)."


# One-sentence comment: Provide multiple analogies to help different learners.
def analogies() -> Dict[str, str]:
    """Return a set of short analogies for different audiences.

    Why:
        Different learners respond to different metaphors. Provide a small set so
        you can pick the one that clicks for your audience.

    Entries:
        - time_vs_cross: movie vs photograph (core analogy)
        - river_vs_lake: flow vs snapshot
        - novel_vs_short_story: sequence vs single scene
        - train_schedule_vs_roster: ordered timetable vs unordered list of passengers
    """
    return {
        "time_vs_cross": "Movie vs photograph: time series is a movie, cross-sectional is a photograph.",
        "river_vs_lake": "River vs lake: a river's water keeps moving and changing (time dependence); a lake snapshot shows water level at a single time.",
        "novel_vs_short_story": "Novel vs short story: a novel unfolds events over time, a short story captures a single moment.",
        "train_schedule_vs_roster": "Train schedule vs roster: a schedule is ordered and time-dependent, a roster is an unordered list of people at one time.",
    }


# -----------------------------------------------------------------------------
# Time series example
# -----------------------------------------------------------------------------

# One-sentence comment: Create a small synthetic time series with trend, seasonality and autocorrelation.
def make_time_series_example(periods: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Create a simple daily-like time series showing trend, weekly seasonality and autocorrelation.

    Why:
        To demonstrate temporal ordering, lag dependence and the need for forecasting methods.

    How:
        - Linear trend + weekly sine seasonality + AR(1)-style autocorrelated noise.

    When to use:
        Use time-series models (ARIMA/ETS/ML sequence models) when observations are dependent
        across time and order matters.

    Returns:
        DataFrame with DatetimeIndex and columns: ['y', 'trend', 'seasonal', 'noise'].

    Example:
        >>> df_ts = make_time_series_example(30, seed=1)
        >>> df_ts.head().shape
        (5, 4)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")

    # Trend: slow linear increase
    trend = 0.05 * t

    # Weekly seasonality: period 7 (daily data)
    seasonal = 2.0 * np.sin(2 * math.pi * t / 7)

    # Autocorrelated noise (AR(1) style): eps[t] = phi*eps[t-1] + normal
    phi = 0.6
    eps = np.zeros(periods)
    sigma = 0.8
    for i in range(1, periods):
        eps[i] = phi * eps[i - 1] + rng.normal(0, sigma)

    y = 10.0 + trend + seasonal + eps  # level + components

    df = pd.DataFrame(
        {
            "y": y,
            "trend": trend,
            "seasonal": seasonal,
            "noise": eps,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# -----------------------------------------------------------------------------
# Cross-sectional example
# -----------------------------------------------------------------------------

# One-sentence comment: Create a small synthetic cross-sectional dataset for many units at one time.
def make_cross_section_example(n_units: int = 100, seed: int = 0) -> pd.DataFrame:
    """
    Create a cross-sectional dataset: many independent entities observed at one time.

    Why:
        To show that cross-sectional methods assume independence across rows and
        model variation across entities (not over time).

    How:
        Generate features (store_size, distance_to_center) and a sales outcome that
        depends on them plus independent noise (no autocorrelation).

    When to use:
        Use OLS/regression/classification which assume iid observations (or handle clustering).

    Why is order irrelevant:
        In cross-sectional data each row is a different entity observed at the same time.
        Because rows represent different units rather than successive moments, permuting
        the row order does not change the joint distribution of the variables under the
        standard assumption of independence and exchangeability.
        In practice this means summary statistics, regression estimates, and most
        cross-sectional inference are invariant to row order. Order can matter when
        sampling design or clustering induces dependence; then you must account for that
        structure explicitly.

    Returns:
        DataFrame with columns: ['unit_id', 'store_size', 'distance', 'sales'].

    Example:
        >>> df_cs = make_cross_section_example(5, seed=1)
        >>> df_cs.shape
        (5, 4)
    """
    rng = np.random.default_rng(seed)
    unit_id = np.arange(1, n_units + 1)
    # Store size in square meters (random between 50 and 500)
    store_size = rng.uniform(50, 500, size=n_units)
    # Distance to city center in km (0.1 to 30)
    distance = rng.uniform(0.1, 30.0, size=n_units)

    # True relationship: sales positively related to store_size, negatively to distance
    sales = 2.0 * store_size - 10.0 * distance + rng.normal(0, 200.0, size=n_units)

    df = pd.DataFrame(
        {
            "unit_id": unit_id,
            "store_size": store_size,
            "distance": distance,
            "sales": sales,
        }
    )
    return df


# -----------------------------------------------------------------------------
# Explain order irrelevance (text + quick demo)
# -----------------------------------------------------------------------------

# One-sentence comment: Provide a compact explanation of why ordering rows is irrelevant for cross-sectional analysis.
def explain_order_irrelevance_cross_section() -> str:
    """Return a concise explanation of why row order is typically irrelevant in cross-sectional data.

    Why:
        This textual helper is useful when teaching or documenting model assumptions.

    Returns:
        A short paragraph explaining the invariance to row permutation under iid/exchangeability.
    """
    return (
        "In cross-sectional data each row corresponds to a distinct entity observed at the same "
        "point in time. Under the usual iid or exchangeability assumptions, the joint distribution "
        "does not depend on the order of rows, so permuting rows leaves summary statistics and "
        "most inference unchanged. Order only becomes relevant when the sampling design or clustering "
        "introduces dependence across rows (for example, household clusters or spatial correlation)."
    )


# One-sentence comment: Demonstrate that shuffling rows does not change basic cross-sectional summaries.
def demonstrate_shuffling_equivalence(df: pd.DataFrame, columns: Tuple[str, ...] = ("sales", "store_size"), seed: int = 0) -> Dict[str, object]:
    """
    Show that simple summaries (mean, std, correlations) are invariant to random row permutation.

    Why:
        A short runnable demonstration helps students internalize why order is irrelevant.

    How:
        - Compute statistics on the original DataFrame.
        - Shuffle rows and recompute; return both for comparison.

    Returns:
        Dict with 'original' and 'shuffled' sub-dicts containing summaries.
    """
    if not all(col in df.columns for col in columns):
        raise ValueError("requested columns must exist in df")

    original_stats = {}
    for col in columns:
        original_stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}

    original_corr = df[list(columns)].corr().to_dict()

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled_stats = {}
    for col in columns:
        shuffled_stats[col] = {"mean": float(shuffled[col].mean()), "std": float(shuffled[col].std())}

    shuffled_corr = shuffled[list(columns)].corr().to_dict()

    return {
        "original": {"stats": original_stats, "corr": original_corr},
        "shuffled": {"stats": shuffled_stats, "corr": shuffled_corr},
    }


# -----------------------------------------------------------------------------
# Compact comparison helper (plots + simple regression)
# -----------------------------------------------------------------------------

# One-sentence comment: Compare the two datasets with simple visuals and a cross-sectional regression.
def compare_examples(ts_df: pd.DataFrame, cs_df: pd.DataFrame, show_plots: bool = True) -> Dict[str, object]:
    """
    Run a small comparison demonstrating the key differences:

    - For time series: plot the series, show an autocorrelation estimate (lag-1).
    - For cross-section: scatter sales vs store_size and optionally run OLS.

    Returns:
        Dict with simple diagnostics: {'ts_lag1_autocorr': float, 'cs_ols_summary': str or None}
    """
    results: Dict[str, object] = {}

    # Time series: lag-1 autocorrelation
    ts_series = ts_df["y"]
    lag1 = ts_series.autocorr(lag=1)
    results["ts_lag1_autocorr"] = float(lag1)

    if show_plots:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        # Line plot
        ts_series.plot(ax=ax[0, 0], title="Time series (y) - line plot")
        # Components
        ts_df[["trend", "seasonal"]].plot(ax=ax[0, 1], title="Trend and Seasonal components")
        # Lag-1 scatter to visualize autocorrelation
        ax[1, 0].scatter(ts_series[:-1].values, ts_series[1:].values, alpha=0.7)
        ax[1, 0].set_xlabel("y[t-1]")
        ax[1, 0].set_ylabel("y[t]")
        ax[1, 0].set_title(f"Lag-1 scatter (autocorr ~ {lag1:.2f})")
        # Cross-sectional scatter
        ax[1, 1].scatter(cs_df["store_size"], cs_df["sales"], alpha=0.7)
        ax[1, 1].set_xlabel("store_size")
        ax[1, 1].set_ylabel("sales")
        ax[1, 1].set_title("Cross-section: sales vs store_size")
        plt.tight_layout()
        plt.show()

    # Cross-sectional: simple OLS if statsmodels is available
    ols_summary = None
    if _HAS_SM:
        X = cs_df[["store_size", "distance"]]
        X = sm.add_constant(X)  # add intercept
        model = sm.OLS(cs_df["sales"], X).fit()
        ols_summary = model.summary().as_text()
        results["cs_ols_params"] = model.params.to_dict()
        results["cs_ols_summary"] = ols_summary
    else:
        results["cs_ols_params"] = {
            "note": "statsmodels not installed; pip install statsmodels to run OLS example"
        }
        results["cs_ols_summary"] = None

    return results


# -----------------------------------------------------------------------------
# Short textual examples to run in a REPL / notebook
# -----------------------------------------------------------------------------

# One-sentence comment: Provide tiny copy-paste snippets for learning quickly.
def textual_examples() -> Tuple[str, str]:
    """
    Return two small textual examples (one for time series, one for cross-sectional)
    suitable to paste into a notebook cell.

    Example usage (not executed here):
        ts = make_time_series_example()
        ts.head()

        cs = make_cross_section_example()
        cs.head()
    """
    ts_snip = """# Time-series example
from time_vs_cross_examples import make_time_series_example
ts = make_time_series_example(60)
print(ts.head())
print("Lag-1 autocorr:", ts['y'].autocorr(lag=1))"""

    cs_snip = """# Cross-sectional example
from time_vs_cross_examples import make_cross_section_example, demonstrate_shuffling_equivalence
cs = make_cross_section_example(100)
print(cs.head())
print(demonstrate_shuffling_equivalence(cs, ('sales','store_size')))
# quick OLS (optional, requires statsmodels)
if True:
    import statsmodels.api as sm
    X = sm.add_constant(cs[['store_size','distance']])
    print(sm.OLS(cs['sales'], X).fit().summary())"""

    return ts_snip, cs_snip


# -----------------------------------------------------------------------------
# End of module
# -----------------------------------------------------------------------------

# Teaching note (concise):
# - Time series: order matters -> evaluate autocorrelation, use forecasting/backtesting.
# - Cross-sectional: independence assumed -> use regression/classification with iid assumptions.
# - Use the analogies() set to pick a metaphor that fits your audience.
