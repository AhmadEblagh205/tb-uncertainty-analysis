"""Linear measurement model and the coursework relative-bias assumptions."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data import BIAS_SPECS, TARGET


@dataclass
class FittedModel:
    intercept: float
    coef: pd.Series
    x_mean: pd.Series
    x_std: pd.Series

    @property
    def features(self):
        return list(self.coef.index)


def fit(df: pd.DataFrame) -> FittedModel:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    reg = LinearRegression().fit(X, y)
    return FittedModel(
        float(reg.intercept_),
        pd.Series(reg.coef_, index=X.columns),
        X.mean(),
        X.std(ddof=1),
    )


def _sample_one_bias(rng: np.random.Generator, family: str,
                     scale: float, n: int) -> np.ndarray:
    if family == "normal":
        return rng.normal(0.0, scale, n)
    if family == "uniform":
        return rng.uniform(-scale, scale, n)
    if family == "triangular":
        return rng.triangular(-scale, 0.0, scale, n)
    raise ValueError(f"Unknown bias family: {family}")


def sample_biases(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Draw relative coefficient/intercept biases from the stated scenario.

    The ``scale`` parameters do not all mean the same statistical quantity:
    normal uses a standard deviation, while uniform/triangular use half-widths.
    """
    return {
        name: _sample_one_bias(rng, spec.family, spec.scale, n)
        for name, spec in BIAS_SPECS.items()
    }


def evaluate(m: FittedModel, x: pd.DataFrame | pd.Series,
             bias: dict[str, float] | None = None) -> np.ndarray:
    bias = bias or {}
    coef = m.coef.copy()
    for name, b in bias.items():
        if name in coef.index:
            coef[name] = coef[name] * (1 + b)
    inter = m.intercept * (1 + bias.get("_intercept", 0.0))
    return inter + np.asarray(x[m.features]) @ coef.to_numpy()
