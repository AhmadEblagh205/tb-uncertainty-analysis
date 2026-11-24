"""Exact propagated-variance decomposition for the linear coursework model."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from data import load, remove_outliers_iqr
from model import fit
from uncertainty import exact_variance_budget

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "TB_demographics.csv")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    df = remove_outliers_iqr(load(None if args.synthetic else args.data,
                                  synthetic=args.synthetic))
    m = fit(df)
    b = exact_variance_budget(m)
    tab = b["components"].copy()

    FIGURES.mkdir(parents=True, exist_ok=True)
    top = tab.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    ax.barh(range(len(top)), top["ST"])
    ax.set_yticks(range(len(top)), [s.replace("PerCapita", "") for s in top["factor"]], fontsize=8)
    ax.set_xlabel("Exact total-order share of propagated variance")
    ax.set_title("Full 25-factor uncertainty budget")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIGURES / "exact_variance_budget.png", dpi=150)
    plt.close(fig)

    summary = {
        "factor_count": int(len(tab)),
        "analytical_mean": round(b["mean"], 4),
        "analytical_variance": round(b["variance"], 4),
        "analytical_standard_uncertainty": round(b["standard_uncertainty"], 4),
        "sum_first_order": round(b["sum_first_order"], 6),
        "interaction_share": round(b["interaction_share"], 6),
        "top_total_order": {
            row.factor: {"S1": round(float(row.S1), 4), "ST": round(float(row.ST), 4)}
            for row in tab.head(10).itertuples()
        },
        "note": "Exact under the repository's independent-input coursework propagation assumptions.",
    }
    out = ROOT / "reports" / "variance_budget.json"
    out.write_text(json.dumps(summary, indent=2))
    tab.to_csv(ROOT / "reports" / "variance_budget.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
