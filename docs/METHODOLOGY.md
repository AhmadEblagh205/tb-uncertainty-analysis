# Methodology and audit notes

## Scope

This repository is a reconstruction of my UCL BENG0091 uncertainty-analysis coursework. The original notebook is preserved under `notebooks/`. The v2 rebuild is not an epidemiological TB risk model and does not claim that the coursework uncertainty assumptions are validated measurement-error distributions. Its purpose is to make the numerical uncertainty analysis internally consistent, testable and explicit.

The real county-level dataset was supplied inside the module and is not redistributed. Public CI uses the deterministic synthetic schema fixture.

## 1. Linear model

After the selected preprocessing strategy, an ordinary least-squares model is fitted:

\[
Y = \beta_0 + \sum_j \beta_j X_j.
\]

The coursework propagation then treats each predictor as

\[
X_j = \mu_j + \sigma_j Z_j, \qquad Z_j\sim N(0,1),
\]

independently across predictors. This uses the sample standard deviation as the scale of the input-variability scenario. It should not be confused with a validated county-level measurement-error model.

Seven relative bias factors modify the intercept and six selected coefficients:

\[
\tilde\beta_j = \beta_j(1+B_j).
\]

The distribution parameter called `scale` has family-specific meaning: normal uses a standard deviation; uniform and symmetric triangular use a half-width. Calling all of these values "standard uncertainties" would therefore be incorrect.

## 2. Full factor set

The propagated model has 25 independent factors:

- 18 input-variability factors, one for every predictor;
- one intercept-bias factor;
- six selected coefficient-bias factors.

An earlier rebuild computed Sobol/Morris results on only 13 factors: six selected input variables plus seven biases. That scope did **not** match the Monte Carlo model, which varied all 18 predictor inputs. It therefore overstated the importance of `avgResidualsPerCapitaPerYear`. V2 aligns the factor set.

## 3. Exact variance budget

For an unbiased selected feature,

\[
Y_j = \beta_j(\mu_j + \sigma_j Z_j)(1+B_j),
\]

with independent zero-mean `Z_j` and `B_j`. Its ANOVA variance terms are available exactly:

\[
V_{X_j} = \beta_j^2\sigma_j^2,
\]

\[
V_{B_j} = \beta_j^2\operatorname{Var}(B_j)\mu_j^2,
\]

\[
V_{X_j,B_j} = \beta_j^2\operatorname{Var}(B_j)\sigma_j^2.
\]

The intercept-bias term is

\[
V_{B_0}=\beta_0^2\operatorname{Var}(B_0).
\]

Because the factors are assumed independent, total output variance is the sum of all first-order terms plus one copy of each input:bias interaction. Exact first-order and total-order Sobol shares then follow directly.

For the coursework dataset after the original all-column IQR cleaning, the exact propagated standard uncertainty is **45.665**. The leading total-order shares are:

- `PublicCoveragePerCapita`: **0.4709**;
- `avgResidualsPerCapitaPerYear`: **0.2468**;
- `PublicCoverageAlonePerCapita`: **0.2278**.

Interactions account for only about **0.013%** of the propagated variance under this specified budget.

## 4. Monte Carlo propagation and convergence

The Monte Carlo sampler implements the same 25-factor assumptions. At 100,000 draws the simulated standard uncertainty is **45.828**, only about **0.36%** above the exact 45.665 benchmark.

The original coursework-style stopping rule asked whether the *change between two consecutive running standard deviations* remained below 0.4% for 1,000 steps. In the audited run it stops around **1,755** draws, but the estimated relative Monte Carlo standard error of the standard deviation is still about **1.72%**. Small step-to-step changes are therefore not the same thing as a 0.4% uncertainty estimate.

V2 keeps that rule only as a historical diagnostic. The primary adaptive diagnostic uses a plug-in fourth-moment estimate of the Monte Carlo standard error of the sample standard deviation. The 0.4% relative-MCSE criterion is reached around **31,000** draws in the reference run.

The exact analytical variance is available here because the model is linear with simple independent multiplicative biases. In a model without an analytical benchmark, convergence would need to be assessed numerically rather than inferred from this closed form.

## 5. Numerical Sobol validation

`sobol.py` transforms 25 independent uniform variables into the **same distributions used by the Monte Carlo propagation**, then applies Saltelli/Jansen estimators. At 8,192 base samples (221,184 model evaluations), the largest absolute numerical error versus the exact total-order indices is about **0.013** in the reference run.

The numerical Sobol result is therefore a validation of the sensitivity implementation rather than the sole source of the ranking.

## 6. Morris screening

Morris uses bounded ranges rather than probability distributions. V2 deliberately calls this a **scenario screen** rather than a probabilistic variance decomposition.

- predictor ranges: mean ± 2 sample SD;
- normal bias factors: ± 2 bias SD;
- bounded uniform/triangular biases: their full stated half-widths.

All 25 factors are included. The top Morris ranking now agrees qualitatively with the exact variance budget: `PublicCoveragePerCapita`, `avgResidualsPerCapitaPerYear` and `PublicCoverageAlonePerCapita` dominate.

## 7. IQR-cleaning sensitivity

The original coursework drops a row if **any** column, including the target, lies outside its 1.5-IQR fence. That behaviour is preserved for faithful reproduction but is no longer silently treated as a neutral preprocessing choice.

The exact propagated standard uncertainty is:

- no IQR removal: **48.816**;
- predictor-only IQR removal: **46.686**;
- original all-column coursework IQR: **45.665**.

Thus the original cleaning reduces the propagated SD by about **6.45%** relative to no IQR removal in this dataset. This is a preprocessing-sensitivity result, not a claim that one strategy is epidemiologically correct.

## 8. What the output uncertainty is not

The propagated distribution is **not** a predictive interval for an individual county or a calibrated epidemiological uncertainty statement. It does not currently include:

- regression residual/model-discrepancy uncertainty;
- regression-coefficient estimation uncertainty;
- dependence between the 18 predictor input perturbations;
- independently validated measurement-error distributions;
- causal interpretation of regression coefficients.

The strongest claim is narrower: given the coursework uncertainty scenario and independence assumptions, the numerical Monte Carlo, exact analytical variance decomposition and numerical Sobol check are now mutually consistent.
