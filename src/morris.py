"""Morris elementary-effects screening over a bounded 25-factor envelope.

Morris is intentionally labelled a *scenario screen*: the bounded ranges are
not probability distributions.  The full 25 factors mirror the Monte Carlo
factor set (18 inputs + seven relative biases), fixing the old 13-factor scope
mismatch.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load, remove_outliers_iqr
from model import fit
from uncertainty import bounded_morris_ranges, point_output

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def morris_trajectory(k: int, rng, p: int = 4) -> tuple[np.ndarray, float]:
    if p < 4 or p % 2:
        raise ValueError("p must be an even integer >=4")
    delta = p / (2 * (p - 1))
    B = np.tril(np.ones((k + 1, k)), -1)
    J = np.ones((k + 1, k))
    D = np.diagflat(rng.choice((-1, 1), size=k))
    grid = np.arange(p) / (p - 1)
    levels = grid[grid <= 1 - delta + 1e-12]
    x_star = rng.choice(levels, size=(1, k))
    P = np.eye(k)[rng.permutation(k)]
    B_star = (J * x_star + (delta / 2) * ((2 * B - J) @ D + J)) @ P
    return np.clip(B_star, 0, 1), delta


def build_factor_space(m):
    return bounded_morris_ranges(m)


def model_output(m, point, names):
    return point_output(m, point, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "TB_demographics.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = remove_outliers_iqr(load(None if args.synthetic else args.data,
                                  synthetic=args.synthetic))
    m = fit(df)
    names, lows, highs = build_factor_space(m)
    rng = np.random.default_rng(args.seed)
    effects = {n: [] for n in names}
    for _ in range(args.trajectories):
        traj, _ = morris_trajectory(len(names), rng)
        phys = lows + traj * (highs - lows)
        vals = [model_output(m, row, names) for row in phys]
        for i in range(len(names)):
            changed = np.flatnonzero(np.abs(traj[i + 1] - traj[i]) > 1e-12)
            if len(changed) != 1:
                raise RuntimeError("Morris trajectory changed !=1 factor")
            j = int(changed[0])
            step = traj[i + 1, j] - traj[i, j]
            effects[names[j]].append((vals[i + 1] - vals[i]) / step)

    stats = pd.DataFrame({
        "mu_star": {n: float(np.mean(np.abs(v))) for n, v in effects.items()},
        "sigma": {n: float(np.std(v, ddof=1)) for n, v in effects.items()},
    }).sort_values("mu_star", ascending=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    ax.scatter(stats["mu_star"], stats["sigma"], s=35)
    for n, row in stats.head(12).iterrows():
        ax.annotate(n.replace("PerCapita", ""), (row["mu_star"], row["sigma"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 3))
    ax.set(xlabel="mu* over chosen bounded scenario ranges",
           ylabel="sigma of elementary effects",
           title=f"Morris scenario screening: {len(names)} factors, {args.trajectories} trajectories")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "morris_screening.png", dpi=150)
    plt.close(fig)

    summary = {
        "factor_count": len(names),
        "trajectories": args.trajectories,
        "scope_note": "Bounded scenario screening, not a probabilistic variance decomposition.",
        "ranking": stats.round(4).to_dict("index"),
    }
    (ROOT / "reports" / "morris_ranking.json").write_text(json.dumps(summary, indent=2))
    print(stats.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
