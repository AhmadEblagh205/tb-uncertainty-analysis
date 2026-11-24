import numpy as np
import pandas as pd

from data import (BIAS_SPECS, TARGET, make_synthetic, remove_outliers_iqr,
                  remove_predictor_outliers_iqr)
from mcm import legacy_adaptive_stop, mcse_stop, run_mcm, running_std, std_mcse
from model import fit, sample_biases
from morris import morris_trajectory
from sobol import sobol_indices
from uncertainty import (bias_variance, exact_variance_budget, factor_names,
                         unit_cube_to_output)


def test_iqr_cleaning_only_removes_rows_and_preserves_schema():
    df = make_synthetic(n=500, seed=1)
    cleaned = remove_outliers_iqr(df)
    assert len(cleaned) <= len(df)
    assert list(cleaned.columns) == list(df.columns)


def test_predictor_only_cleaning_does_not_use_target_for_fences():
    df = make_synthetic(n=500, seed=2)
    altered = df.copy()
    altered.loc[0, TARGET] = 1e12
    a = remove_predictor_outliers_iqr(df)
    b = remove_predictor_outliers_iqr(altered)
    assert a.index.equals(b.index)


def test_running_std_matches_numpy():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    rs = running_std(y)
    assert np.isclose(rs[-1], y.std(ddof=1), atol=1e-9)


def test_legacy_adaptive_stop_is_retained_but_mcse_rule_is_separate():
    rng = np.random.default_rng(1)
    y = rng.normal(size=50000)
    legacy = legacy_adaptive_stop(y)
    modern = mcse_stop(y, rel_tol=0.01, min_samples=1000, step=500)
    assert legacy < len(y)
    assert modern <= len(y)
    s = y[:modern].std(ddof=1)
    assert std_mcse(y[:modern]) / s <= 0.0105


def test_morris_trajectory_changes_one_factor_per_step():
    rng = np.random.default_rng(2)
    traj, delta = morris_trajectory(k=8, rng=rng)
    assert traj.shape == (9, 8)
    for i in range(8):
        changed = np.nonzero(np.abs(traj[i + 1] - traj[i]) > 1e-12)[0]
        assert len(changed) == 1
        assert np.isclose(abs(traj[i + 1, changed[0]] - traj[i, changed[0]]), delta)


def test_bias_samples_cover_all_seven_biases():
    rng = np.random.default_rng(3)
    b = sample_biases(rng, 50)
    assert set(b) == set(BIAS_SPECS)
    assert all(v.shape == (50,) for v in b.values())


def test_bias_distribution_variances_match_parameterisation():
    assert np.isclose(bias_variance(BIAS_SPECS["popEst2015"]), 0.02**2)
    assert np.isclose(bias_variance(BIAS_SPECS["_intercept"]), 0.005**2 / 3)
    assert np.isclose(bias_variance(BIAS_SPECS["Employed16_OverPerCapita"]),
                      0.0025**2 / 6)


def test_full_uncertainty_budget_has_25_factors():
    m = fit(remove_outliers_iqr(make_synthetic(n=2000, seed=4)))
    assert len(m.features) == 18
    assert len(factor_names(m)) == 25
    b = exact_variance_budget(m)
    assert len(b["components"]) == 25
    assert 0.999 <= b["sum_first_order"] <= 1.0


def test_exact_variance_matches_large_monte_carlo_on_synthetic_data():
    m = fit(remove_outliers_iqr(make_synthetic(n=2500, seed=5)))
    exact = exact_variance_budget(m)["standard_uncertainty"]
    y = run_mcm(m, np.random.default_rng(6), 120000)
    assert abs(y.std(ddof=1) / exact - 1.0) < 0.015


def test_unit_cube_transform_has_expected_output_mean():
    m = fit(remove_outliers_iqr(make_synthetic(n=1800, seed=7)))
    rng = np.random.default_rng(8)
    U = rng.random((50000, len(factor_names(m))))
    y = unit_cube_to_output(m, U)
    exact = exact_variance_budget(m)
    assert abs(y.mean() - exact["mean"]) < 0.02 * exact["standard_uncertainty"]


def test_numerical_sobol_recovers_dominant_exact_factor():
    m = fit(remove_outliers_iqr(make_synthetic(n=2200, seed=9)))
    exact = exact_variance_budget(m)["components"].set_index("factor")
    S1, ST, _ = sobol_indices(m, n_base=5000, seed=10)
    top_exact = exact["ST"].idxmax()
    top_num = max(ST, key=ST.get)
    assert top_num == top_exact
    assert abs(ST[top_exact] - float(exact.loc[top_exact, "ST"])) < 0.06


def test_target_aware_and_predictor_only_cleaning_are_distinct_operations():
    df = make_synthetic(n=1000, seed=11)
    df.loc[df.index[:5], TARGET] = df[TARGET].mean() + 100 * df[TARGET].std()
    all_cols = remove_outliers_iqr(df)
    pred_only = remove_predictor_outliers_iqr(df)
    assert len(all_cols) < len(pred_only)
