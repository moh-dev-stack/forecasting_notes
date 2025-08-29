got it — here’s a **Foundations — Time Series Literacy** playbook with zero code. It’s practical, opinionated, and organized for quick use.

---

# Foundations — Time Series Literacy Playbook (no code)

## 0) Purpose in one line

Make your data **regular, interpretable, and diagnosable** before modeling: validate time indexing and frequency, visualize structure (trend/seasonality/noise), and run light stationarity and correlation checks.

---

## 1) Quick decision rules (What / Why / How)

**1) Enforce a regular frequency first.**

* *What:* equally spaced timestamps (e.g., daily, hourly, 15-minute).
* *Why:* Most techniques assume fixed spacing; gaps distort seasonality and correlation.
* *How:* Detect cadence; reindex to a regular grid and explicitly represent gaps.

**2) Decide timestamp semantics (instant vs period total/average).**

* *What:* Is the value an instantaneous measurement or an aggregate over the preceding period?
* *Why:* Determines how you resample and interpret shifts (sum vs mean).
* *How:* Document it once and treat consistently everywhere.

**3) Handle timezone and DST deliberately.**

* *What:* Local vs UTC; daylight saving shifts.
* *Why:* DST causes duplicated or missing hours; naïve datetimes create silent errors.
* *How:* Choose a canonical zone (often UTC), convert once, and keep it.

**4) Plot before you compute.**

* *What:* Raw line plot; rolling mean and rolling standard deviation; seasonal subseries.
* *Why:* Visuals reveal problems and structure faster than tests.

**5) Use STL to separate structure from noise.**

* *What:* Robust decomposition into trend, seasonality, remainder.
* *Why:* Clarifies additive vs multiplicative behavior and reveals leftover anomalies.

**6) Prefer MAE plus a scale-free metric (MASE or sMAPE) for early comparisons.**

* *What:* Metrics that compare across scales and avoid divide-by-zero pathologies.
* *Why:* MAPE can explode near zeros; RMSE alone overweights large errors.

**7) Don’t confuse seasonality with trend.**

* *What:* Repeating patterns vs long-run drift.
* *Why:* Misclassification leads to the wrong model family.
* *How:* Seasonal subseries plots and STL make the distinction explicit.

**8) Treat stationarity tests as guidance, not gospel.**

* *What:* ADF (null = unit root), KPSS (null = stationary).
* *Why:* Over-differencing removes signal; under-differencing leaves structure in residuals.
* *How:* Use minimal differencing confirmed by visuals.

**9) Label outliers first; transform later.**

* *What:* Spikes from holidays, outages, or glitches.
* *Why:* They skew variance and mislead diagnostics.
* *How:* Flag with simple rules; decide whether to cap, model with exogenous flags, or accept.

**10) Write a one-paragraph EDA summary for every series.**

* *What:* Frequency, range, missingness, trend/seasonality, stationarity hints, recommended next steps.
* *Why:* Forces clarity and speeds team alignment.

---

## 2) Step-by-step workflow (first pass)

### A) Ingest and index

1. Parse timestamps; sort by time; remove true duplicates (or aggregate if intended).
2. Detect the observed cadence; pick the target frequency.
3. Reindex to a regular grid; make gaps explicit (missing values are visible, not hidden).

### B) Validate time semantics

4. Decide whether values are instants or period totals/averages.
5. Define resampling conventions (sum for totals; mean for instants). Document this.

### C) Visual basics

6. Raw line plot over full range; if long, also plot recent windows.
7. Rolling mean and rolling standard deviation with a sensible window (e.g., one seasonal period).
8. Seasonal diagnostics:

   * Seasonal subseries (e.g., each month as a small line in the same axes).
   * Box/violin plots by month and day-of-week (or by hour for intraday).

### D) Decomposition and correlation

9. STL decomposition using candidate period(s) (e.g., 7 for daily-weekly, 12 for monthly-yearly).
10. Autocorrelation (ACF) and partial autocorrelation (PACF) up to \~3–4 seasonal periods.

    * ACF spikes at seasonal lags → seasonality present.
    * Slow ACF decay → trend or unit root.
    * PACF/PACF shapes suggest AR/MA orders (for later stages).

### E) Stationarity sanity checks

11. ADF and KPSS on the raw and lightly transformed series.

    * ADF p < 0.05 and KPSS p > 0.05 → plausibly stationary.
    * Otherwise note what kind of differencing might be needed later (regular or seasonal), but do not overreact.

### F) Data quality and transforms

12. Identify outliers and missing runs; annotate likely causes (promo, holiday, outage).
13. Inspect variance vs level. If swings grow with the level and values are strictly positive, consider a log/Box-Cox transform (record the choice).
14. Finalize frequency, timezone policy, semantics, and any transforms in your EDA summary.

---

## 3) Recognition guide (what patterns look like)

* **Trend:** rolling mean shifts steadily; STL trend is non-flat; ACF decays slowly.
* **Seasonality:** repeated patterns at fixed spacing; ACF bumps at m, 2m, 3m; seasonal subseries align across cycles.
* **Heteroskedasticity:** variability grows with the level; log/Box-Cox can stabilize.
* **Structural break:** sudden level shift; consider segmenting or adding regime flags.
* **Calendar effects:** day-of-week, month, holidays visible in grouped plots; may warrant exogenous indicators later.

---

## 4) ACF/PACF cheat sheet (light, for intuition only)

* **AR(p):** PACF cuts off around p; ACF tails off.
* **MA(q):** ACF cuts off around q; PACF tails off.
* **ARMA:** both tail off.
* **Seasonal AR/MA:** look at multiples of the seasonal period (m, 2m…) for seasonal orders.
  Use these as hints only; confirm with cross-validation later.

---

## 5) Common pitfalls (and how to avoid them)

* **Irregular timestamps treated as regular:** always reindex to a fixed grid and expose gaps.
* **DST chaos:** local times create duplicates/missing hours; pick a canonical timezone and convert once.
* **Wrong resampling rule:** summing a series that represents averages, or averaging a series that represents totals; set rules per variable.
* **MAPE near zeros:** prefer sMAPE or MASE; if you must report MAPE, explicitly exclude zeros and disclose it.
* **Over-differencing due to over-trusting tests:** use minimal differencing; visuals and STL are your guardrails.
* **Reading seasonality as trend:** seasonal subseries and STL help separate them cleanly.
* **Silent outlier handling:** always annotate what you did (cap, flag, ignore) and why.

---

## 6) What “good foundations” look like (acceptance criteria)

* Regular frequency enforced; missingness quantified and visible.
* Time semantics (instant vs period) documented; resampling rules agreed.
* Raw plot, rolling stats, seasonal subseries/boxplots saved.
* STL components reviewed; additive vs multiplicative behavior noted.
* ACF/PACF inspected; candidate seasonal period justified.
* ADF/KPSS recorded (with plain-language interpretation).
* Outliers and structural breaks listed with hypotheses.
* One-paragraph EDA summary written.

---

## 7) One-paragraph EDA summary template (fill this in)

> **Frequency & range:** \[e.g., Daily] from \[start] to \[end], timezone \[UTC/local], regularized to fixed cadence with \[X%] missing periods explicitly represented.
> **Semantics:** Values represent \[instants / period totals]; resampling uses \[mean/sum] accordingly.
> **Structure:** Clear \[weekly/yearly] seasonality (period = \[m]) with \[constant / level-dependent] amplitude; trend is \[rising/flat/falling] since \[year/period].
> **Stationarity hints:** ADF p = \[..], KPSS p = \[..]. Minimal differencing likely \[not needed / regular d=1 / seasonal D=1 at m].
> **Variance:** \[Stable / increases with level]. Transform \[not needed / log recommended].
> **Data quality:** Missing runs at \[dates]; outliers at \[dates] likely due to \[reason].
> **Next steps:** Baselines (naive, seasonal-naive), then \[ETS type] or \[SARIMA with s = m], evaluated with rolling-origin CV at horizons \[..].

---

## 8) Tools to keep on your radar (no commands, just names)

* Data and visuals: pandas, matplotlib (or any plotting stack you prefer).
* Decomposition and tests: statsmodels’ STL, ADF, KPSS, ACF/PACF utilities.
* Metrics to report early: MAE, MASE, sMAPE (plus RMSE if large errors are costly).

---

## 9) Ready-to-use checklist (tick as you go)

* [ ] Timestamps parsed, sorted; timezone policy chosen
* [ ] Regular frequency enforced; gaps explicit and quantified
* [ ] Variable semantics documented (instant vs total/average)
* [ ] Raw plot + rolling mean/std saved
* [ ] Seasonal subseries and grouped box/violin plots saved
* [ ] STL run with candidate period(s); components reviewed
* [ ] ACF/PACF reviewed up to \~3–4 seasonal cycles
* [ ] ADF and KPSS recorded with interpretation
* [ ] Outliers/missing runs identified and policy noted
* [ ] One-paragraph EDA summary completed and archived

---

### Mini-glossary

* **Frequency:** spacing between observations.
* **Stationarity:** stable mean/variance/autocorrelation over time.
* **Trend:** long-run level change.
* **Seasonality:** repeated pattern with fixed period.
* **STL:** robust decomposition into seasonal, trend, remainder.
* **ACF/PACF:** correlations with past values to reveal memory structure.
* **MASE/sMAPE:** scale-free error metrics suited to uneven scales and zeros.

If you want, I can turn this into a one-page PDF/Markdown checklist for your team or adapt it to your specific dataset and write the EDA summary from a brief description.
