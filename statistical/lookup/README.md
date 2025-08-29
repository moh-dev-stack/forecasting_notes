Awesome—here’s a big set of **practical decision rules** you can lean on. For each, I include **What** the terms mean, **Why** the rule is sensible, and **How** to quickly check or act. Skim the bold one-liners; dip into the bullets when you need context.

---

# Quick decision rules (with What / Why / How)

### Data & frequency

**1) If timestamps aren’t strictly regular, fix the frequency before anything else.**

* **What:** Regular = equally spaced (e.g., daily).
* **Why:** Most models assume a fixed step; gaps distort seasonality and ACF/PACF.
* **How:** Set frequency (`asfreq`, `resample`) and decide how to impute or mark gaps.

**2) If there is less than \~2 full seasonal cycles, avoid seasonal models.**

* **What:** A “cycle” is one full period (e.g., m=12 for monthly yearly seasonality).
* **Why:** Seasonal components need enough repetitions to estimate reliably.
* **How:** Count cycles; if <2, favor SES/Holt or ARIMA without seasonal terms.

**3) If the horizon is short relative to the data window, prefer simpler models first.**

* **What:** Short horizon = a few steps (e.g., 1–7 days ahead).
* **Why:** Simple models (naive/ETS) often win at short horizons with less variance.
* **How:** Benchmark naive/seasonal-naive; only escalate if you cannot beat them.

---

### Seasonality & trend

**4) If seasonality is clear and amplitude is roughly constant, try ETS with additive seasonality.**

* **What:** Amplitude constant = peaks/troughs have similar absolute size over time.
* **Why:** ETS(A, \*, A) captures fixed seasonal bumps cleanly.
* **How:** Seasonal plot or STL: seasonal component has stable band; choose `seasonal="add"`.

**5) If seasonality scales with the level, try ETS with multiplicative seasonality.**

* **What:** Peaks get taller as the series grows; troughs deeper when the level is high.
* **Why:** Multiplicative seasonality models percentage-like swings.
* **How:** STL shows seasonal range widening with level; use `seasonal="mul"` and ensure values > 0.

**6) If long-run trend seems to slow or saturate, use a damped trend in ETS.**

* **What:** Damping reduces the trend’s effect as horizon grows.
* **Why:** Linear extrapolation can explode; damping improves realism.
* **How:** Compare ETS with vs without `damped_trend=True` on holdout.

**7) If the series has a clear trend but weak seasonality, prefer Holt (double) smoothing or ARIMA without seasonal terms.**

* **What:** Trend = systematic increase/decrease.
* **Why:** Trend-only models are simpler and harder to overfit.
* **How:** Try Holt and ARIMA(p,d,q) with d=1 if ADF indicates non-stationarity.

---

### Stationarity & differencing

**8) If ADF p > 0.05 and KPSS p < 0.05, difference the series (d ≥ 1).**

* **What:** ADF tests unit root (non-stationary). KPSS tests stationarity.
* **Why:** Stationarity is needed for ARIMA to behave.
* **How:** Difference once; re-test. Stop when ADF suggests stationarity and KPSS does not reject.

**9) If seasonality remains after regular differencing, add seasonal differencing (D ≥ 1, period s).**

* **What:** Seasonal differencing removes repeated level shifts every s steps.
* **Why:** Helps eliminate seasonal unit roots.
* **How:** Use `y_t - y_{t-s}` once; re-check ACF at seasonal lags.

**10) If ACF and PACF look short-memory after differencing, favor lower p and q.**

* **What:** Short-memory = few significant lags.
* **Why:** Parsimony prevents overfitting and improves generalization.
* **How:** Start small (p,q in {0,1,2}); escalate only if diagnostics require.

**11) If you suspect over-differencing (excessive noise), reduce d or D.**

* **What:** Over-differencing removes signal and inflates variance.
* **Why:** ARIMA becomes noisy; forecasts degrade.
* **How:** Compare models with d and d-1 using AIC and residual structure.

---

### Model choice: ETS vs ARIMA vs ARIMAX vs Decomposition

**12) If residuals after ETS show autocorrelation, try ARIMA/SARIMA.**

* **What:** Residual autocorrelation = structure left over (ACF significant).
* **Why:** ARIMA handles auto-structure in errors.
* **How:** Ljung-Box p < 0.05 or visible ACF spikes → move to ARIMA/SARIMA.

**13) If you have reliable, forecast-time-available regressors, use ARIMAX.**

* **What:** Regressors like price, promos, weather forecasts.
* **Why:** External drivers add explainability and accuracy.
* **How:** Validate exog availability for the full horizon; create lags if effects are delayed.

**14) If you face multiple or non-integer seasonality (e.g., hourly with daily + weekly), try TBATS or decomposition + ARIMA.**

* **What:** Multiple seasonalities exceed plain SARIMA easily.
* **Why:** TBATS or hybrid STL+ARIMA handles complex seasonal layers.
* **How:** Use STL with different windows or a library that supports multiple m’s.

**15) If variance grows with level, consider Box-Cox transform or multiplicative models.**

* **What:** Heteroskedasticity: wider swings at higher levels.
* **Why:** Stabilizing variance improves fit and intervals.
* **How:** Box-Cox or log transform (only if strictly positive).

---

### Cross-validation & metrics

**16) Always evaluate with rolling-origin CV, not random shuffles.**

* **What:** Rolling/expanding windows respecting time order.
* **Why:** Time leakage breaks honest error estimates.
* **How:** Define origins, horizons; compute distributions of errors.

**17) Report at least MAE and a scale-free metric (MASE or sMAPE).**

* **What:** Scale-free allows cross-series comparison.
* **Why:** RMSE alone can hide bias; MAPE breaks near zero.
* **How:** Use MAE + MASE; add RMSE for large-error sensitivity.

**18) If zeros or near-zeros are common, avoid MAPE and prefer sMAPE/MASE.**

* **What:** MAPE divides by actual value; zero inflates error to infinity.
* **Why:** Unstable and misleading.
* **How:** Drop MAPE or compute on non-zero subset with a clear warning.

**19) Compare models on identical CV protocols and horizons.**

* **What:** Same origins, horizon grid, metrics.
* **Why:** Protocol differences can dwarf model differences.
* **How:** Centralize the CV routine and reuse it for every model.

---

### Exogenous variables (ARIMAX) and leakage

**20) If a regressor is not known at forecast time, do not include it unless you can forecast it first.**

* **What:** Leakage = using future info.
* **Why:** Inflated apparent accuracy that fails in production.
* **How:** Build or procure forecasts for exog; otherwise exclude.

**21) If effects are delayed, create lagged exogenous features.**

* **What:** Promotional uplift 1–2 periods later.
* **Why:** Align cause and effect in the model.
* **How:** Add `promo_lag1`, `promo_lag2`; pick lags based on domain or cross-correlation.

**22) If exog has many columns vs few data points, regularize or reduce.**

* **What:** High dimensionality.
* **Why:** Overfitting and unstable coefficients.
* **How:** Prior feature selection, PCA, or move to ML with proper CV.

---

### Special data shapes

**23) If demand is intermittent with many zeros, consider Croston, SBA, or intermittent-aware models.**

* **What:** Sparse, bursty series (e.g., spare parts).
* **Why:** Standard ETS/ARIMA struggle.
* **How:** Use specialized methods or ML classification + regression hybrids.

**24) If frequent outliers exist, use robust decomposition (STL) or robust loss.**

* **What:** Spikes due to anomalies.
* **Why:** Parameters get biased.
* **How:** Robust STL; winsorize or model outliers separately.

**25) If missing runs are long, model at a coarser frequency first.**

* **What:** Long gaps.
* **Why:** Interpolation becomes guesswork.
* **How:** Aggregate to weekly/monthly; forecast; downscale if justified.

---

### Horizon & objective

**26) If decision needs are near-term only, optimize 1–3 step horizons specifically.**

* **What:** Horizon-specific tuning.
* **Why:** The best model for h=1 may differ from h=14.
* **How:** Report per-horizon metrics and pick the best per use case.

**27) If you need full distributions or service levels, use probabilistic forecasts.**

* **What:** Intervals or quantiles.
* **Why:** Inventory, risk, SLAs require uncertainty, not just points.
* **How:** ARIMA intervals, quantile regression, bootstraps, CRPS/quantile loss.

---

### Hierarchies & coherence

**28) If you forecast across a hierarchy (SKU → category → total), reconcile.**

* **What:** Coherence = lower levels add up to higher levels.
* **Why:** Avoid inconsistent reporting and planning errors.
* **How:** Bottom-up, top-down, middle-out, or MinT reconciliation.

**29) If leaf series are noisy but aggregate is stable, consider middle-out.**

* **What:** Middle levels often best signal-to-noise.
* **Why:** Balanced accuracy and coherence.
* **How:** Forecast at middle level; disaggregate with proportions; reconcile.

---

### Interpretability, compute, and production

**30) If stakeholders need interpretable components, favor ETS or ARIMAX.**

* **What:** Components and coefficients explain “why.”
* **Why:** Adoption and trust.
* **How:** Show level/trend/season charts; coefficient signs with CIs.

**31) If you have thousands of short series, start with ETS/ARIMA via an automated loop.**

* **What:** Scale, limited per-series data.
* **Why:** Cheap, fast, strong baselines at scale.
* **How:** Batch fit with simple model grids; reconcile if hierarchical.

**32) If compute budget is tight, avoid heavy deep models unless classical baselines fail.**

* **What:** Deep nets need data and tuning.
* **Why:** Cost vs marginal gain can be poor.
* **How:** Prove a gap over ETS/ARIMA in CV before escalating.

**33) If retraining latency must be minutes, prefer ETS/ARIMA over deep models.**

* **What:** Latency SLA.
* **Why:** Fast fit and inference.
* **How:** Cache differencing, reuse orders, incremental updates if possible.

**34) If data or patterns change often, design for regular re-training and drift monitoring.**

* **What:** Concept drift.
* **Why:** Old models decay.
* **How:** Schedule retraining, track rolling error, trigger alerts on drift.

---

### Diagnostics & sanity checks

**35) If residuals are biased (mean far from 0), revisit trend/transform/feature set.**

* **What:** Bias = systematic under/over-prediction.
* **Why:** Indicates missing structure.
* **How:** Check residual mean; plot residuals vs time and fitted values.

**36) If residuals show seasonality, increase seasonal orders or model seasonality explicitly.**

* **What:** ACF spikes at seasonal lags.
* **Why:** Missed seasonal structure.
* **How:** Add D, P, Q or use ETS with seasonality.

**37) If prediction intervals are too narrow, use a variance-stabilizing transform or probabilistic method.**

* **What:** Under-coverage (e.g., 90% interval covers only 70% of actuals).
* **Why:** Underestimated uncertainty.
* **How:** Log/Box-Cox, heavier-tailed errors, bootstrap.

**38) If model selection by AIC disagrees with CV, trust CV.**

* **What:** AIC is in-sample information criterion.
* **Why:** Out-of-sample error matters more for forecasting.
* **How:** Pick the model with better rolling-origin performance.

---

# Mini-glossary (plain language)

* **Seasonality (period m):** repeating pattern every m steps (weekly, yearly).
* **Trend:** long-run increase or decrease.
* **Stationarity:** distribution does not change over time; mean/variance and autocorrelation stable.
* **Differencing (d, D):** transform using current minus lagged values to remove trend/seasonal drift.
* **ACF/PACF:** correlation with past values; helps suggest MA (q) and AR (p) orders.
* **Ljung-Box test:** checks residual autocorrelation; we want high p-values (no structure left).
* **Leakage:** training with information you would not have at forecast time.
* **Horizon:** how many steps ahead you predict.
* **MASE:** scale-free error relative to a naive benchmark; <1 means better than naive.
* **Rolling-origin CV:** evaluation that moves the forecast origin forward in time while preserving order.
* **Reconciliation (hierarchical):** adjust forecasts so children add to parents.
* **Damped trend:** reduces trend impact as horizon grows.

---

If you want, I can turn these rules into a **one-page PDF/markdown cheat-sheet** or a **lint-like checklist** that you can run against a series (it would print “pass/fail” next to each rule with quick suggestions).
