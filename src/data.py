"""Data loading and preprocessing for the TB uncertainty coursework rebuild.

The real county-level dataset was supplied inside UCL BENG0091 and is not
redistributed by this repository.  A synthetic generator with the same schema
keeps the code and CI runnable without the coursework file.

The original coursework removed a row when *any* column, including the target,
was beyond a 1.5-IQR fence.  That behaviour is preserved as
``remove_outliers_iqr`` for faithful reproduction.  V2 also exposes a
predictor-only variant so the effect of that preprocessing choice can be
reported rather than hidden.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "TARGET_residualsRatePerCapita"


@dataclass(frozen=True)
class BiasSpec:
    """Distribution parameterisation for a relative coefficient bias.

    ``scale`` is deliberately *not* called a standard uncertainty because its
    meaning depends on ``family``:
      - normal: standard deviation;
      - uniform: symmetric half-width;
      - triangular: symmetric half-width with mode zero.
    """

    family: str
    scale: float


# These distribution families/scales are the assumptions used by the rebuilt
# coursework propagation.  They are scenario assumptions, not independently
# validated epidemiological measurement-error estimates.
BIAS_SPECS: dict[str, BiasSpec] = {
    "_intercept": BiasSpec("uniform", 0.005),
    "popEst2015": BiasSpec("normal", 0.02),
    "HS25_OverPerCapita": BiasSpec("normal", 0.01),
    "Employed16_OverPerCapita": BiasSpec("triangular", 0.0025),
    "Unemployed16_OverPerCapita": BiasSpec("normal", 0.01),
    "avgResidualsPerCapitaPerYear": BiasSpec("uniform", 0.04),
    "povertyPerCapita": BiasSpec("uniform", 0.015),
}

# Backwards-compatible aliases used by the original modules/tests.
SELECTED = {k: v.scale for k, v in BIAS_SPECS.items() if k != "_intercept"}
INTERCEPT_U = BIAS_SPECS["_intercept"].scale


def iqr_bounds(col: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Faithful coursework cleaning: drop a row if *any* column is outside
    its 1.5-IQR fence, including the target column."""
    keep = pd.Series(True, index=df.index)
    for col in df.columns:
        low, up = iqr_bounds(df[col])
        keep &= df[col].between(low, up)
    return df[keep].copy()


def remove_predictor_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity alternative: apply the same IQR rule to predictors only.

    This is not asserted to be the uniquely correct preprocessing strategy; it
    exists so the effect of target-aware coursework cleaning can be measured.
    """
    keep = pd.Series(True, index=df.index)
    for col in df.columns:
        if col == TARGET:
            continue
        low, up = iqr_bounds(df[col])
        keep &= df[col].between(low, up)
    return df[keep].copy()


def validate_schema(df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(f"All coursework columns must be numeric; non-numeric: {bad}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; define an explicit policy before fitting")


def load(path: str | Path | None = None, synthetic: bool = False,
         seed: int = 0) -> pd.DataFrame:
    if synthetic or path is None:
        df = make_synthetic(seed=seed)
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Coursework data not found at {path}. See data/README.md or use --synthetic."
            )
        df = pd.read_csv(path)
    validate_schema(df)
    return df


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the coursework schema and rough numerical scales.

    The generator is only an offline engineering fixture; it is not intended to
    emulate the epidemiology or joint distribution of the original dataset.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "incidenceRatePerCapita": rng.normal(450, 50, n),
        "popEst2015": rng.lognormal(10.3, 1.1, n),
        "MarriedPerCapita": rng.normal(52000, 6400, n),
        "NoHS18_24PerCapita": rng.normal(18000, 7500, n),
        "HS18_24PerCapita": rng.normal(35000, 8800, n),
        "BachDeg18_24PerCapita": rng.normal(6000, 3900, n),
        "HS25_OverPerCapita": rng.normal(35000, 6800, n),
        "BachDeg25_OverPerCapita": rng.normal(13000, 5100, n),
        "Employed16_OverPerCapita": rng.normal(54000, 7800, n),
        "Unemployed16_OverPerCapita": rng.normal(7800, 3200, n),
        "PrivateCoveragePerCapita": rng.normal(64500, 10300, n),
        "EmpPrivCoveragePerCapita": rng.normal(41000, 9200, n),
        "PublicCoveragePerCapita": rng.normal(36000, 7600, n),
        "PublicCoverageAlonePerCapita": rng.normal(19000, 5800, n),
        "MarriedHouseholdsPerCapita": rng.normal(51000, 6100, n),
        "avgResidualsPerCapitaPerYear": rng.normal(229, 58, n),
        "povertyPerCapita": rng.normal(16700, 6000, n),
        "AvgHouseholdSizePerCapita": rng.lognormal(2.0, 1.0, n),
    })
    y = (95 + 2.0e-5 * df["popEst2015"]
         + 9.0e-4 * df["HS25_OverPerCapita"]
         + 4.0e-4 * df["Employed16_OverPerCapita"]
         + 1.6e-3 * df["Unemployed16_OverPerCapita"]
         + 0.12 * df["avgResidualsPerCapitaPerYear"]
         + 8.0e-4 * df["povertyPerCapita"]
         + rng.normal(0, 12, n))
    df.insert(0, TARGET, y)
    return df
