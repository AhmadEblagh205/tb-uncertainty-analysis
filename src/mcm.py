"""Monte Carlo propagation for the coursework uncertainty scenario.

V2 keeps the historical 0.4%/1000-step rule as a *legacy diagnostic* but no
longer labels it a rigorous convergence guarantee.  The primary convergence
criterion is a plug-in Monte Carlo standard error (MCSE) for the estimated
standard deviation.  The model also has an exact analytical variance, which is
used as an independent implementation benchmark.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import TARGET, load, remove_outliers_iqr
from model import fit, sample_biases
from uncertainty import exact_variance_budget

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def run_mcm(m, rng, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    biases = sample_biases(rng, n)
    y = np.full(n, m.intercept * (1 + biases["_intercept"]))
    for name in m.features:
        b = biases.get(name, np.zeros(n))
        x = m.x_mean[name] + m.x_std[name] * rng.standard_normal(n)
        y += m.coef[name] * (1 + b) * x
    return y


def running_std(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = np.arange(1, len(y) + 1)
    csum, csum2 = np.cumsum(y), np.cumsum(y ** 2)
    var = np.maximum(csum2 / n - (csum / n) ** 2, 0.0)
    return np.sqrt(var * n / np.maximum(n - 1, 1))


def legacy_adaptive_stop(y: np.ndarray, tol: float = 0.004,
                         patience: int = 1000) -> int:
    """Historical coursework heuristic, retained for comparison only."""
    run_std = running_std(y)
    count = 0
    for i in range(2, len(run_std)):
        prev = run_std[i - 1]
        rel = abs(1 - run_std[i] / prev) if prev > 0 else np.inf
        count = count + 1 if rel <= tol else 0
        if count >= patience:
            return i + 1
    return len(run_std)


# Backwards-compatible name for old tests/users, but README now calls it legacy.
adaptive_stop = legacy_adaptive_stop


def std_mcse(y: np.ndarray) -> float:
    """Plug-in Monte Carlo SE of the sample standard deviation.

    Uses the large-sample variance of s^2 based on the fourth central moment,
    followed by a delta-method transform to s.  This adapts to non-normal
    output rather than assuming Gaussian kurtosis.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return float("inf")
    s = float(y.std(ddof=1))
    if s == 0:
        return 0.0
    c = y - y.mean()
    mu4 = float(np.mean(c ** 4))
    var_s2 = (mu4 - ((n - 3) / (n - 1)) * s ** 4) / n
    return float(np.sqrt(max(var_s2, 0.0)) / (2.0 * s))


def mcse_stop(y: np.ndarray, rel_tol: float = 0.004,
              min_samples: int = 2000, step: int = 500) -> int:
    """First checkpoint whose estimated relative MCSE for s is <= rel_tol."""
    if not (0 < rel_tol < 1):
        raise ValueError("rel_tol must lie in (0,1)")
    n_total = len(y)
    for n in range(min_samples, n_total + 1, step):
        s = float(np.std(y[:n], ddof=1))
        if s > 0 and std_mcse(y[:n]) / s <= rel_tol:
            return n
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "TB_demographics.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(None if args.synthetic else args.data, synthetic=args.synthetic)
    n0 = len(df)
    df = remove_outliers_iqr(df)
    m = fit(df)
    exact = exact_variance_budget(m)

    y = run_mcm(m, np.random.default_rng(args.seed), args.samples)
    rstd = running_std(y)
    legacy = legacy_adaptive_stop(y)
    stop = mcse_stop(y)
    full_sd = float(y.std(ddof=1))
    exact_sd = exact["standard_uncertainty"]

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(df[TARGET], bins=50, density=True, alpha=0.5,
            label="coursework target distribution")
    ax.hist(y, bins=50, density=True, alpha=0.5,
            label="propagated coursework uncertainty scenario")
    ax.set(xlabel=TARGET, ylabel="Density",
           title="Observed target spread vs propagated coursework scenario")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "output_distribution.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4.6))
    x = np.arange(1, len(rstd) + 1)
    ax2.semilogx(x, rstd, lw=1.2, label="running Monte Carlo SD")
    ax2.axhline(exact_sd, color="black", ls=":", lw=1.4,
                label=f"exact SD = {exact_sd:.2f}")
    ax2.axvline(legacy, color="tab:red", ls="--", lw=1.1,
                label=f"legacy heuristic = {legacy:,}")
    ax2.axvline(stop, color="tab:green", ls="--", lw=1.1,
                label=f"0.4% relative-MCSE stop = {stop:,}")
    ax2.set(xlabel="Monte Carlo samples (log)", ylabel="Estimated standard uncertainty",
            title="Monte Carlo convergence against an analytical benchmark")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig(FIGURES / "mcm_convergence.png", dpi=150)
    plt.close(fig2)

    lo, hi = np.percentile(y, [2.5, 97.5])
    stop_sd = float(np.std(y[:stop], ddof=1))
    legacy_sd = float(np.std(y[:legacy], ddof=1))
    summary = {
        "rows_raw": int(n0),
        "rows_after_coursework_iqr_cleaning": int(len(df)),
        "mcm_samples": int(args.samples),
        "mean_output_mcm": round(float(y.mean()), 3),
        "mean_output_analytical": round(float(exact["mean"]), 3),
        "standard_uncertainty_mcm_full": round(full_sd, 3),
        "standard_uncertainty_analytical": round(exact_sd, 3),
        "mcm_vs_analytical_relative_error": round(abs(full_sd / exact_sd - 1.0), 5),
        "expanded_uncertainty_k2_analytical": round(2 * exact_sd, 3),
        "propagated_output_95pct_interval_mcm": [round(float(lo), 2), round(float(hi), 2)],
        "legacy_0p4pct_step_heuristic": {
            "stop_at": int(legacy),
            "sd_at_stop": round(legacy_sd, 3),
            "estimated_relative_mcse_at_stop": round(std_mcse(y[:legacy]) / legacy_sd, 5),
        },
        "relative_mcse_0p4pct_rule": {
            "stop_at": int(stop),
            "sd_at_stop": round(stop_sd, 3),
            "estimated_relative_mcse_at_stop": round(std_mcse(y[:stop]) / stop_sd, 5),
        },
        "scope_note": "This is propagation under the coursework uncertainty scenario, not a predictive interval for county TB incidence.",
    }
    out = ROOT / "reports" / "mcm_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
