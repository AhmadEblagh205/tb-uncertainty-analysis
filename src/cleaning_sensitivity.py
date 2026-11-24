"""Measure how the coursework IQR-cleaning choice changes the UQ result."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import load, remove_outliers_iqr, remove_predictor_outliers_iqr
from model import fit
from uncertainty import exact_variance_budget

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def analyse(df, label):
    b = exact_variance_budget(fit(df))
    top = b["components"].iloc[0]
    return {
        "strategy": label,
        "rows": len(df),
        "standard_uncertainty": b["standard_uncertainty"],
        "top_factor": top["factor"],
        "top_factor_ST": float(top["ST"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "TB_demographics.csv")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    raw = load(None if args.synthetic else args.data, synthetic=args.synthetic)
    rows = [
        analyse(raw, "no IQR removal"),
        analyse(remove_predictor_outliers_iqr(raw), "predictor-only IQR"),
        analyse(remove_outliers_iqr(raw), "coursework all-column IQR"),
    ]
    tab = pd.DataFrame(rows)
    ref = float(tab.loc[tab["strategy"] == "no IQR removal", "standard_uncertainty"].iloc[0])
    tab["change_vs_raw_pct"] = 100 * (tab["standard_uncertainty"] / ref - 1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(tab["strategy"], tab["standard_uncertainty"])
    ax.set_ylabel("Exact propagated standard uncertainty")
    ax.set_title("Sensitivity of the UQ result to IQR preprocessing")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "cleaning_sensitivity.png", dpi=150)
    plt.close(fig)

    summary = {"strategies": [
        {"strategy": r.strategy, "rows": int(r.rows),
         "standard_uncertainty": round(float(r.standard_uncertainty), 3),
         "change_vs_raw_pct": round(float(r.change_vs_raw_pct), 2),
         "top_factor": r.top_factor,
         "top_factor_ST": round(float(r.top_factor_ST), 4)}
        for r in tab.itertuples()
    ],
    "note": "This is a preprocessing-sensitivity audit, not a claim that one IQR strategy is epidemiologically correct."}
    (ROOT / "reports" / "cleaning_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
