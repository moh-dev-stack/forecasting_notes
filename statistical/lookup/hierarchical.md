# Hierarchical & Grouped Forecasting — an extensive playbook

Use this as a practical field guide. Skim the quick decisions, then work top to bottom when building a pipeline.

---

## 0) What this is (in one breath)

You have many related series that sum or group into each other (SKU → subcategory → category → total, or stores → regions → country). You want **coherent** forecasts: numbers at the leaves should add up to their parents. You first make **base** forecasts for each series, then **reconcile** them so they are consistent across the hierarchy. Do the same idea for time aggregations (hour → day → week) if you need **temporal** coherence.

---

## 1) Core concepts (plain language)

* **Cross-sectional hierarchy:** tree across entities at a single time resolution. Example: SKUs roll up to subcategories, then categories, then total.
* **Temporal hierarchy:** tree across time scales for the same series. Example: hourly aggregates to daily, weekly, monthly.
* **Grouped hierarchy:** series that slice multiple dimensions (e.g., product x region). You often reconcile within each dimension and across their combination.
* **Coherence:** forecasts at children sum exactly to their parents. This is a hard constraint you enforce after making base forecasts.
* **Summing matrix (S):** a matrix that says how bottom series add up to all the upper nodes. It encodes the structure.
* **Base forecasts:** forecasts you fit independently for each series (often at the bottom level).
* **Reconciliation:** linear transformation of base forecasts that forces coherence. Methods: Bottom-up, Top-down, Middle-out, and MinT (optimal).

---

## 2) Quick decision rules (one-liners with What/Why/How)

1. **If most information lives at the bottom level and data is not too noisy, use Bottom-up.**

   * What: Sum leaf forecasts to get parents.
   * Why: Simple, unbiased if leaf models are decent.
   * How: Forecast every leaf, then add with the known hierarchy.

2. **If bottom level is very noisy but higher levels are stable, consider Top-down.**

   * What: Forecast the total, split down using proportions.
   * Why: Totals have better signal-to-noise.
   * How: Choose proportions (historical average shares, forecasted shares, or recent shares).

3. **If the “middle” level is most stable, use Middle-out.**

   * What: Forecast the middle tier, roll up to top and allocate down to leaves.
   * Why: Balances stability and detail.
   * How: Pick the middle tier with best SNR on EDA.

4. **If you want the best overall accuracy subject to coherence, use MinT (optimal reconciliation).**

   * What: A weighted least squares reconciliation using residual covariance.
   * Why: Accounts for forecast error correlation and variance across series.
   * How: Estimate a residual covariance matrix, choose a MinT variant (OLS, WLS, shrinkage).

5. **If you have very few historical points per series, avoid estimating big covariance matrices.**

   * What: MinT needs W (residual covariance).
   * Why: Poor estimates degrade reconciliation.
   * How: Use MinT-OLS (identity W), or diagonal WLS (variance only), or shrinkage estimators.

6. **If you need both cross-sectional and temporal coherence, reconcile across both.**

   * What: Cross-temporal reconciliation.
   * Why: Numbers must be consistent across space and time.
   * How: Reconcile within each dimension, then jointly (many teams do cross then temporal).

7. **If a parent has business constraints (non-negativity, caps), apply constraints after reconciliation.**

   * What: Clip or constrained optimization.
   * Why: Keep forecasts feasible.
   * How: Non-negativity clipping or quadratic programming with coherence constraints.

8. **If a leaf is intermittent (many zeros), use intermittent-aware base models before reconciling.**

   * What: Croston, SBA, or zero-inflated methods.
   * Why: Standard models bias sparse demand.
   * How: Fit intermittent models per leaf, then reconcile.

9. **If you are short on time, start with Bottom-up and compare to MinT-OLS.**

   * What: Two strong baselines.
   * Why: Quick to implement, often competitive.
   * How: Build both and compare accuracy at multiple levels.

10. **If your business cares most about accuracy at a particular level, optimize to that level.**

    * What: Target level (e.g., category weekly).
    * Why: Not all levels matter equally.
    * How: Tune base models and reconciliation to minimize error at the target level, while maintaining coherence elsewhere.

---

## 3) Pipeline overview (end-to-end)

1. **Define the hierarchy.**

   * Enumerate nodes. Decide the **bottom level** (disjoint leaf series) and all ancestors up to the **top level** (total).
   * Build the **S matrix** once: rows for all series, columns for leaves; S maps leaves to all nodes.

2. **Data prep.**

   * Standardize frequency, align calendars, fill or mark missing timestamps.
   * Handle outliers consistently across leaves (winsorize or flagged events).
   * Ensure index alignment so summing is exact every timestamp.

3. **EDA and baselines (per series).**

   * Quick EDA per level (aggregate plots).
   * Compute naive and seasonal-naive baselines at leaves and at key aggregate levels.

4. **Fit base models.**

   * Typically at leaves (ETS, ARIMA, ARIMAX if regressors exist).
   * Consider pooled or shared hyperparameters for small data.
   * Save residuals for covariance estimation.

5. **Pick reconciliation method.**

   * Start with Bottom-up and MinT-OLS.
   * If sample size is decent and correlation across leaves matters, try MinT-shrinkage or WLS.

6. **Reconcile.**

   * Apply the reconciliation matrix to base forecasts to obtain coherent forecasts for **all levels**.

7. **Evaluate.**

   * Use **rolling-origin CV**; compute metrics at **multiple levels** and horizons.
   * Report accuracy by level (leaf, mid, top) and overall (weighted average or macro average).
   * Check bias and coherence holds exactly.

8. **Select and iterate.**

   * Choose reconciliation and base model combo that meets the business objective level.
   * Re-tune or switch base models where underperformance clusters (e.g., specific categories).

9. **Productionize.**

   * Freeze hierarchy definition, cache S, and pin model versions.
   * Automate retraining cadence, monitoring, and drift alerts per level.
   * Store both base and reconciled forecasts and the reconciliation matrix used.

---

## 4) Reconciliation methods (what/why/how)

### Bottom-up (BU)

* **What:** Forecast leaves, sum to get parents: $\tilde{y} = S \hat{y}_{\text{leaf}}$.
* **Why:** Simple, coherence guaranteed. Strong if leaf models are decent.
* **How:** Reliable when leaves are not too noisy and coverage is stable.

### Top-down (TD)

* **What:** Forecast the total, split down using proportions.
* **Why:** Totals can be more stable; avoids noisy leaf fits.
* **How:** Choose proportions:

  * historical average share,
  * recent share (more reactive),
  * forecasted proportions from a share model.
* **Caveat:** Errors in proportions cascade down.

### Middle-out (MO)

* **What:** Forecast at a middle tier, aggregate up and allocate down.
* **Why:** Middle level often has best SNR.
* **How:** Pick middle tier empirically (lowest CV of residuals per level).

### MinT (optimal reconciliation)

* **What:** Minimize total forecast error variance subject to coherence.
* **Why:** Uses error covariance to weight adjustments, improving accuracy overall.
* **How (concept):** Estimate a residual covariance $W$. Reconciled forecast $\tilde{y} = S (S' W^{-1} S)^{-1} S' W^{-1} \hat{y}$ (high-level shape).
* **Variants:**

  * **MinT-OLS:** $W = I$ (identity). Robust when data is scarce.
  * **MinT-WLS:** $W$ diagonal (use per-series forecast error variance).
  * **MinT-Shrinkage:** shrink sample covariance toward diagonal for stability.

**Choosing among them:**

* Small data or unstable covariance → MinT-OLS or Bottom-up.
* Moderate data and clear variance differences → MinT-WLS.
* Sufficient data and correlated errors across leaves → MinT-shrinkage.

---

## 5) Temporal hierarchies and cross-temporal coherence

* **Temporal hierarchy:** same series across multiple resolutions (hour, day, week).
* **Why:** Stakeholders consume forecasts at different cadences; totals must match when aggregated.
* **How:**

  1. Generate base forecasts at all required granularities or at the highest and aggregate.
  2. Apply temporal reconciliation so sums match across time scales.
* **Cross-temporal:** require coherence across both cross-sectional and temporal trees.

  * Practical approach: reconcile cross-sectionally at each time scale, then reconcile temporally; or use a joint method if available.
* **Decision rule:** if the primary consumption level is weekly, optimize there and reconcile other time scales to it.

---

## 6) Evaluation and CV design (multi-level)

* **Protocol:** rolling-origin backtest shared by all series.
* **Horizon grid:** include business horizons (e.g., 1, 2, 4, 8 weeks).
* **Metrics:** report at each level. Combine using:

  * **Macro average:** average metric across series at a level.
  * **Volume-weighted average:** weight by mean or recent volume so high-impact series count more.
* **What to report:** per-level MAE and MASE (scale-free), plus RMSE if large errors matter.
* **Coverage:** if producing intervals, check empirical coverage by level.
* **Sanity:** verify coherence on every backtest fold (parents equal sum of children exactly).

---

## 7) Data preparation pitfalls and remedies

* **Misaligned calendars:** ensure every leaf has the same timestamp set; fill with zeros where appropriate (e.g., closed stores) to preserve sums.
* **Dupes or missing leaves:** a missing leaf breaks parent sums. Keep a master list of leaves and enforce presence (even if all zeros).
* **Outliers and structural breaks:** treat consistently across leaves; otherwise reconciliation will spread artifacts.
* **Changing hierarchy:** if SKUs are added/removed, define effective-date hierarchies or mapping tables to maintain coherence over time.

---

## 8) Handling intermittency and sparsity

* **When:** many zeros or sporadic spikes at leaves.
* **Approach:**

  * Fit intermittent models per leaf (Croston, SBA) or classification-then-regression.
  * Aggregate intermittent leaves into pseudo-leaves with enough signal, then allocate down using historical proportions.
  * Reconcile after using base models suited to sparsity.

---

## 9) Exogenous variables in hierarchies

* **Use case:** promotions or price by SKU, holidays by country.
* **Rule:** avoid leakage; exogenous inputs must be known or forecastable for the horizon.
* **Granularity:** regressors should exist at the same level as the model. If only available at a higher level, consider top-down allocation of effects.
* **Tip:** consistent feature engineering across leaves (same lags, calendars) improves MinT covariance stability.

---

## 10) Business constraints and post-processing

* **Non-negativity:** clip negatives after reconciliation or solve a constrained projection (least squares with non-negativity).
* **Caps and quotas:** apply linear constraints then project back onto the coherent subspace.
* **Rounding:** round at leaves only after reconciliation; minor rounding errors can be re-distributed to preserve sums.

---

## 11) Monitoring and drift (by level)

Track over time:

* **MAPE/MASE by level and horizon** (rolling window).
* **Bias by level** (mean error).
* **Share stability** (proportions used in top-down).
* **Residual covariance stability** for MinT; re-estimate or shrink more aggressively if unstable.
* **Hierarchy changes** (new leaves, retired leaves) and their impact on accuracy.

Alerts:

* Drop in coverage or spike in MAE at any key level.
* Coherence violations (should never happen if pipeline is correct).
* Covariance estimation warnings (ill-conditioned matrices).

---

## 12) Method selection cheat sheet

| Situation                                        | Prefer                                    |
| ------------------------------------------------ | ----------------------------------------- |
| Leaf SNR high, data not scarce                   | Bottom-up                                 |
| Totals very stable, leaves very noisy            | Top-down (recent proportions)             |
| Best SNR at mid level                            | Middle-out                                |
| Need best overall accuracy with enough history   | MinT-shrinkage                            |
| Very small samples                               | MinT-OLS or Bottom-up                     |
| Multiple seasonalities/time resolutions required | Temporal or cross-temporal reconciliation |
| Intermittent demand at leaves                    | Intermittent base models + reconcile      |

---

## 13) Minimal mathematics (only shapes)

* Let $y$ be the stacked vector of all series (all levels).
* Let $b$ be the stacked vector of **base** forecasts (same shape as $y$).
* Let $S$ map leaves to all levels (known from the tree).
* Coherent forecasts live in the column space of $S$: $\tilde{y} = S g$ for some vector $g$.
* **Bottom-up:** forecast leaves $\hat{y}_{\text{leaf}}$. Then $g = \hat{y}_{\text{leaf}}$, so $\tilde{y} = S \hat{y}_{\text{leaf}}$.
* **MinT (concept):** choose $\tilde{y}$ closest to $b$ in a metric defined by residual covariance $W$, subject to $\tilde{y}$ being coherent. Closed form exists and uses $S$ and $W$.

You rarely need more math than this to implement or audit.

---

## 14) Practical build plan (first implementation)

1. **Codify hierarchy:** build and unit-test `S`. Add a check that `sum(children) == parent` on raw data.
2. **Base models:** start with ETS at leaves; store base forecasts and residuals.
3. **Reconciliation:** implement Bottom-up and MinT-OLS. Keep both as baselines.
4. **Evaluation:** rolling-origin CV; report MAE/MASE at leaf, mid, top; include weighted average.
5. **Upgrade:** add MinT-WLS (diagonal) using per-series residual variance; optionally shrinkage.
6. **Temporal:** add temporal reconciliation if required by consumers.
7. **Production:** version S, base models, reconciliation method; log metadata and metrics by level.

---

## 15) Common pitfalls (and fixes)

* **Incoherent input history:** historical parents do not equal sum of children due to missing leaves or inconsistent preprocessing.

  * Fix: rebuild history from leaves upward; enforce alignment and consistent imputations.

* **Covariance blow-ups in MinT:** W is ill-conditioned with short history or too many leaves.

  * Fix: MinT-OLS or diagonal W; shrinkage estimators; aggregate leaves temporarily.

* **Over-fitting leaf models:** too many parameters per leaf with tiny samples.

  * Fix: share hyperparameters, regularize, or use simpler baselines at leaves.

* **Ignoring business constraints:** negative or infeasible reconciled forecasts.

  * Fix: constrained projection or post-clipping with redistribution.

* **Using different CV setups across levels:** misleading comparisons.

  * Fix: single shared rolling-origin generator for all series.

---

## 16) Tools you can use (pragmatic)

* **Python:**

  * `scikit-hts` for basic hierarchical reconciliation and workflows.
  * Statsmodels for base ETS/ARIMA.
  * Your own MinT implementation (small linear algebra with NumPy) is feasible when you already have `S` and residuals.

* **R (more mature):**

  * `hts` and `forecast` packages for tried-and-tested HTS workflows.

* **Tip:** even if you use a library, keep your own tests that validate coherence and compare BU vs MinT on synthetic data.

---

## 17) Starter checklists

**Build checklist**

* [ ] Hierarchy defined and validated (children sum to parent in history)
* [ ] `S` constructed and unit-tested
* [ ] Base models trained at leaves (saved forecasts and residuals)
* [ ] Reconciliation implemented (BU + MinT-OLS)
* [ ] CV configured (origins, horizons, shared across series)
* [ ] Metrics by level and weighted overall metric
* [ ] Monitoring plan (bias, coverage, drift)

**Go-live checklist**

* [ ] Non-negativity and business caps enforced
* [ ] New leaf handling policy (cold start)
* [ ] Hierarchy change protocol (effective dates)
* [ ] Retrain cadence and backfill rules
* [ ] Alerts wired for key levels

---

If you want, I can convert this playbook into a **single .py gist** that:

* takes a CSV of leaf series,
* builds and validates the hierarchy,
* fits simple base models,
* runs Bottom-up and MinT-OLS reconciliation,
* evaluates with rolling-origin CV at multiple levels,
* and outputs a concise report.
