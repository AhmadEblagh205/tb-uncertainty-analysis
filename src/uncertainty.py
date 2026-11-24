"""Shared uncertainty-budget utilities.

The Monte Carlo study uses 25 independent factors:
- 18 standardised input-variability factors z_j ~ N(0,1), one per predictor;
- seven relative coefficient/intercept-bias factors.

For this linear model with multiplicative coefficient biases the propagated
variance and Sobol decomposition can be written down exactly.  That closed-form
budget is used as the primary audit benchmark; Monte Carlo and numerical Sobol
are then checks on the implementation rather than the only source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import BIAS_SPECS, BiasSpec
from model import FittedModel


@dataclass(frozen=True)
class VarianceComponent:
    factor: str
    first_order_variance: float
    interaction_variance: float = 0.0

    @property
    def total_order_variance(self) -> float:
        return self.first_order_variance + self.interaction_variance


def bias_variance(spec: BiasSpec) -> float:
    if spec.family == "normal":
        return spec.scale ** 2
    if spec.family == "uniform":
        return spec.scale ** 2 / 3.0
    if spec.family == "triangular":
        return spec.scale ** 2 / 6.0
    raise ValueError(f"Unknown bias family: {spec.family}")


def exact_variance_budget(m: FittedModel) -> dict:
    """Exact variance/Sobol budget under the repository's independent-factor
    propagation assumptions.

    For selected feature j,
        Y_j = beta_j (mu_j + sigma_j Z_j) (1 + B_j),
    with E[Z]=E[B]=0 and independent Z,B.  The ANOVA pieces are therefore
        X main: beta^2 sigma^2
        bias main: beta^2 Var(B) mu^2
        interaction: beta^2 Var(B) sigma^2.
    """
    components: list[VarianceComponent] = []
    interactions: dict[str, float] = {}

    # Intercept relative bias.
    vb0 = bias_variance(BIAS_SPECS["_intercept"])
    components.append(VarianceComponent(
        "b: intercept", m.intercept ** 2 * vb0, 0.0
    ))

    for f, beta in m.coef.items():
        mu = float(m.x_mean[f])
        sd = float(m.x_std[f])
        vx = float(beta ** 2 * sd ** 2)
        if f in BIAS_SPECS:
            vb = bias_variance(BIAS_SPECS[f])
            v_b = float(beta ** 2 * vb * mu ** 2)
            v_int = float(beta ** 2 * vb * sd ** 2)
            components.append(VarianceComponent(f"x: {f}", vx, v_int))
            components.append(VarianceComponent(f"b: {f}", v_b, v_int))
            interactions[f"x:b {f}"] = v_int
        else:
            components.append(VarianceComponent(f"x: {f}", vx, 0.0))

    # Sum each ANOVA term exactly once: all first-order pieces + one copy of
    # each x:b interaction.  Components carry interaction on both factors only
    # for total-order indices.
    first_sum = sum(c.first_order_variance for c in components)
    interaction_sum = sum(interactions.values())
    total_var = first_sum + interaction_sum

    rows = []
    for c in components:
        rows.append({
            "factor": c.factor,
            "first_order_variance": c.first_order_variance,
            "interaction_variance_in_total_order": c.interaction_variance,
            "S1": c.first_order_variance / total_var,
            "ST": c.total_order_variance / total_var,
        })
    frame = pd.DataFrame(rows).sort_values("ST", ascending=False).reset_index(drop=True)
    return {
        "mean": float(m.intercept + np.dot(m.x_mean.to_numpy(), m.coef.to_numpy())),
        "variance": float(total_var),
        "standard_uncertainty": float(np.sqrt(total_var)),
        "sum_first_order": float(first_sum / total_var),
        "interaction_share": float(interaction_sum / total_var),
        "components": frame,
        "interactions": interactions,
    }


def factor_names(m: FittedModel) -> list[str]:
    return [f"x: {f}" for f in m.features] + [
        "b: intercept",
        *[f"b: {f}" for f in BIAS_SPECS if f != "_intercept"],
    ]


def _triangular_ppf(u: np.ndarray, half_width: float) -> np.ndarray:
    """PPF of symmetric triangular(-a, 0, +a)."""
    a = half_width
    out = np.empty_like(u, dtype=float)
    left = u < 0.5
    out[left] = -a + a * np.sqrt(2.0 * u[left])
    out[~left] = a - a * np.sqrt(2.0 * (1.0 - u[~left]))
    return out


def bias_ppf(u: np.ndarray, spec: BiasSpec) -> np.ndarray:
    eps = np.finfo(float).eps
    u = np.clip(np.asarray(u, dtype=float), eps, 1.0 - eps)
    if spec.family == "normal":
        return norm.ppf(u) * spec.scale
    if spec.family == "uniform":
        return (2.0 * u - 1.0) * spec.scale
    if spec.family == "triangular":
        return _triangular_ppf(u, spec.scale)
    raise ValueError(f"Unknown bias family: {spec.family}")


def unit_cube_to_output(m: FittedModel, U: np.ndarray) -> np.ndarray:
    """Map independent U(0,1) factors to the same distributions as MCM."""
    U = np.asarray(U, dtype=float)
    names = factor_names(m)
    if U.ndim != 2 or U.shape[1] != len(names):
        raise ValueError(f"Expected U with shape (n, {len(names)}), got {U.shape}")

    p = len(m.features)
    eps = np.finfo(float).eps
    z = norm.ppf(np.clip(U[:, :p], eps, 1.0 - eps))
    X = m.x_mean.to_numpy()[None, :] + m.x_std.to_numpy()[None, :] * z

    col = p
    b0 = bias_ppf(U[:, col], BIAS_SPECS["_intercept"])
    col += 1
    B = np.zeros_like(X)
    feature_to_idx = {f: i for i, f in enumerate(m.features)}
    for f, spec in BIAS_SPECS.items():
        if f == "_intercept":
            continue
        B[:, feature_to_idx[f]] = bias_ppf(U[:, col], spec)
        col += 1

    return (m.intercept * (1.0 + b0)
            + np.sum(X * m.coef.to_numpy()[None, :] * (1.0 + B), axis=1))


def bounded_morris_ranges(m: FittedModel) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Bounded scenario envelope used only for Morris screening.

    Input factors use mean +/- 2 sample SD.  Normal bias factors use +/-2 SD;
    bounded uniform/triangular biases use their full half-width.  These are
    screening ranges, not probability distributions.
    """
    names: list[str] = []
    lows: list[float] = []
    highs: list[float] = []
    for f in m.features:
        names.append(f"x: {f}")
        lows.append(float(m.x_mean[f] - 2 * m.x_std[f]))
        highs.append(float(m.x_mean[f] + 2 * m.x_std[f]))
    names.append("b: intercept")
    spec = BIAS_SPECS["_intercept"]
    width = 2 * spec.scale if spec.family == "normal" else spec.scale
    lows.append(-width); highs.append(width)
    for f, spec in BIAS_SPECS.items():
        if f == "_intercept":
            continue
        names.append(f"b: {f}")
        width = 2 * spec.scale if spec.family == "normal" else spec.scale
        lows.append(-width); highs.append(width)
    return names, np.asarray(lows), np.asarray(highs)


def point_output(m: FittedModel, point: Iterable[float], names: list[str]) -> float:
    x = m.x_mean.copy()
    bias: dict[str, float] = {}
    for val, name in zip(point, names):
        kind, _, col = name.partition(": ")
        if kind == "x":
            x[col] = val
        else:
            bias["_intercept" if col == "intercept" else col] = val
    coef = m.coef.copy()
    for f, b in bias.items():
        if f in coef:
            coef[f] *= (1.0 + b)
    inter = m.intercept * (1.0 + bias.get("_intercept", 0.0))
    return float(inter + np.dot(x[m.features].to_numpy(), coef.to_numpy()))
