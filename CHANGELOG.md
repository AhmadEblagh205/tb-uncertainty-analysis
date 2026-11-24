# Changelog

## 0.3.0 — defensibility and uncertainty-budget audit

- Aligned sensitivity analysis with the full 25-factor Monte Carlo budget instead of the old 13-factor subset.
- Added an exact analytical variance/Sobol decomposition for the linear multiplicative-bias model.
- Corrected the old headline that `avgResidualsPerCapitaPerYear` carries ~99% of propagated variance; the full-budget leading total-order shares are PublicCoverage ~47.1%, avgResiduals ~24.7%, and PublicCoverageAlone ~22.8%.
- Added a numerical 25-factor Sobol check using the same factor distributions as Monte Carlo.
- Expanded Morris to all 25 factors and labelled it explicitly as bounded scenario screening.
- Retained the coursework 0.4%/1,000-step stopping rule only as a legacy diagnostic; added a fourth-moment MCSE criterion for the estimated standard deviation.
- Added exact-vs-Monte-Carlo convergence benchmarking: 45.828 simulated vs 45.665 analytical SD at 100k draws.
- Added preprocessing sensitivity for no-IQR, predictor-only IQR and coursework all-column IQR strategies.
- Corrected terminology around bias distribution scales: normal scale is an SD, uniform/triangular scales are half-widths.
- Added schema validation, methodology documentation and 12 tests.
- Removed a stale unrelated `mc_summary.json` artifact.
- Raw coursework data remains intentionally unredistributed.

## 0.2.0 — update round 2

Added Sobol first and total order indices via Saltelli sampling. This historical implementation used a restricted 13-factor scope and is superseded by v0.3.0.

## 0.1.0

Initial release.
