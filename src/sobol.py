"""Numerical Sobol check against the exact 25-factor variance decomposition.

Unlike the old 13-factor implementation, this uses the same 18 input-noise
factors and seven bias distributions as the Monte Carlo propagation.  Because
the fitted model is linear with simple multiplicative biases, exact Sobol
indices are available; numerical Saltelli/Jansen estimates are therefore a
validation exercise, not the primary result.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load, remove_outliers_iqr
from model import fit
from uncertainty import exact_variance_budget, factor_names, unit_cube_to_output

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def sobol_indices(m, n_base: int, seed: int = 0):
    if n_base < 128:
        raise ValueError("n_base must be at least 128")
    names = factor_names(m)
    k = len(names)
    rng = np.random.default_rng(seed)
    A = rng.random((n_base, k))
    B = rng.random((n_base, k))
    fA = unit_cube_to_output(m, A)
    fB = unit_cube_to_output(m, B)
    f0 = float(np.mean(np.concatenate([fA, fB])))
    a = fA - f0
    b = fB - f0
    var = float(np.var(np.concatenate([a, b]), ddof=1))
    S1, ST = {}, {}
    for i, name in enumerate(names):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        ab = unit_cube_to_output(m, ABi) - f0
        # Saltelli 2010 first-order + Jansen total-order estimators.
        S1[name] = float(np.mean(b * (ab - a)) / var)
        ST[name] = float(0.5 * np.mean((a - ab) ** 2) / var)
    return S1, ST, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "TB_demographics.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--base", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = remove_outliers_iqr(load(None if args.synthetic else args.data,
                                  synthetic=args.synthetic))
    m = fit(df)
    S1, ST, var = sobol_indices(m, args.base, args.seed)
    exact = exact_variance_budget(m)["components"].set_index("factor")

    names = factor_names(m)
    err_s1 = {n: S1[n] - float(exact.loc[n, "S1"]) for n in names}
    err_st = {n: ST[n] - float(exact.loc[n, "ST"]) for n in names}
    order = sorted(names, key=lambda n: float(exact.loc[n, "ST"]))

    FIGURES.mkdir(parents=True, exist_ok=True)
    show = order[-12:]
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.plot([float(exact.loc[n, "ST"]) for n in show], y, "o", label="exact ST")
    ax.plot([ST[n] for n in show], y, "x", label="numerical ST")
    ax.set_yticks(y, [n.replace("PerCapita", "") for n in show], fontsize=8)
    ax.set_xlabel("Total-order variance share")
    ax.set_title(f"Numerical Sobol check against exact decomposition (N={args.base:,})")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIGURES / "sobol_indices.png", dpi=150)
    plt.close(fig)

    top = sorted(names, key=lambda n: -float(exact.loc[n, "ST"]))[:8]
    summary = {
        "factor_count": len(names),
        "base_samples": args.base,
        "model_runs": (len(names) + 2) * args.base,
        "numerical_output_variance": round(var, 3),
        "exact_output_variance": round(float(exact_variance_budget(m)["variance"]), 3),
        "max_abs_error_S1": round(max(abs(v) for v in err_s1.values()), 4),
        "max_abs_error_ST": round(max(abs(v) for v in err_st.values()), 4),
        "top_exact_total_order": {
            n: {"S1_exact": round(float(exact.loc[n, "S1"]), 4),
                "ST_exact": round(float(exact.loc[n, "ST"]), 4),
                "ST_numerical": round(ST[n], 4)} for n in top
        },
        "scope_note": "All 25 factors match the MCM factor set/distributions; independent-factor assumption retained from the coursework rebuild.",
    }
    (ROOT / "reports" / "sobol_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
