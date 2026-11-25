# TB Regression Uncertainty Budget — Monte Carlo, Exact Variance and Sensitivity

[![CI](https://github.com/AhmadEblagh205/tb-uncertainty-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/tb-uncertainty-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I rebuilt and checked my UCL BENG0091 uncertainty-analysis coursework: a linear regression on county-level TB-demographic data, with Monte Carlo propagation of the coursework uncertainty budget, an exact variance decomposition, numerical Sobol validation and Morris screening. The coursework dataset was supplied inside the module and I do not redistribute it. Derived aggregate reports are included, and the code and tests also run on a synthetic schema fixture.

## What it does

- Cleans the data with the coursework IQR rule, which takes 3,047 rows down to 2,050.
- Propagates 25 independent uncertain factors (18 input-variability factors and seven coefficient or intercept biases) by Monte Carlo.
- Computes the variance budget exactly, because the model is linear.
- Checks the exact budget with numerical Sobol indices and screens all 25 factors with Morris.
- Reports how the cleaning choice changes the propagated uncertainty.

## Results

Under the coursework all-column cleaning, three factors carry most of the variance: `PublicCoveragePerCapita` input variability (47.09%), `avgResidualsPerCapitaPerYear` (24.68%) and `PublicCoverageAlonePerCapita` (22.78%), about 95.3% together. Input and bias interactions add about 0.013%. An earlier version of my Sobol and Morris code varied only some of the factors that the Monte Carlo varied, which made `avgResidualsPerCapitaPerYear` look like it carried about 99% of the variance. With the full factor set it does not.

A 100,000-draw Monte Carlo run gives a standard uncertainty of 45.828 against an exact 45.665 (0.36% apart), with an exact k=2 expanded uncertainty of 91.330. The old adaptive stopping rule halted at about 1,755 samples while the relative Monte Carlo error of the SD was still about 1.72%. A 0.4% relative-error criterion is reached at about 31,000 samples. Sobol with 8,192 base samples (221,184 model evaluations) is within about 0.013 of the exact total-order indices.

| Preprocessing | Rows | Exact propagated SD |
|---|---:|---:|
| no IQR removal | 3,047 | 48.816 |
| predictor-only IQR | 2,067 | 46.686 |
| coursework all-column IQR | 2,050 | 45.665 |

Morris screening of all 25 factors ranks the same three factors first. The coursework cleaning lowers the propagated SD by about 6.45% against no removal. Figures are in `reports/figures/`, for example `exact_variance_budget.png` and `mcm_convergence.png`.

## How to run

```bash
pip install -r requirements.txt

# Offline synthetic-schema run
python src/mcm.py --synthetic
python src/variance_budget.py --synthetic
python src/sobol.py --synthetic --base 4096
python src/morris.py --synthetic
python src/cleaning_sensitivity.py --synthetic

# With the coursework file at data/raw/TB_demographics.csv
python src/mcm.py
python src/variance_budget.py
python src/sobol.py
python src/morris.py
python src/cleaning_sensitivity.py
```

`pytest -q` runs 12 tests covering the cleaning rules, stopping rules, the 25-factor budget, exact-versus-Monte-Carlo agreement, Sobol recovery and Morris trajectories.

## Notes

The propagated distribution comes from the coursework uncertainty scenario. It is not a county-level prediction interval, and it leaves out residual error, coefficient-estimation uncertainty and dependence between predictors. The normal bias parameter is a standard deviation while the uniform and triangular ones are half-widths, so I call them distribution scales. The original notebook is in `notebooks/`, and the county data is described in `data/README.md`. The synthetic generator is a test fixture and is not epidemiological data. Derivations and audit notes are in `docs/METHODOLOGY.md`.

## Licence

MIT, see [LICENSE](LICENSE).
