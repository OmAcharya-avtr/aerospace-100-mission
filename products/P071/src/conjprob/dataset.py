"""Deterministic synthetic conjunction population for the screening benchmark.

What this is, stated before anything else: a declared synthetic process in the
dimensionless encounter-plane parameterisation. It is **not** a model of the
real conjunction population. No element of it was identified from a catalogue,
an operator's screening output or any published distribution of close
approaches, because this product has no access to such data. Every conclusion
drawn from it is a conclusion about this generator.

What makes it usable anyway is that the Pc integral depends on the encounter
only through ``(sigma_x, sigma_y, xbar, ybar, R)``, so sampling those directly
samples the complete input space of the quantity being screened. The generator
covers that space broadly rather than realistically, which is the right choice
for measuring where a screen fails and the wrong choice for predicting how
often it will.

Three regimes are provided:

* ``"nominal"`` -- effective position uncertainty from 10 m to 3 km, hard-body
  radius 1 m to 30 m, Mahalanobis distance uniform on ``[0, 5]``.
* ``"dilution"`` -- effective position uncertainty from 300 m to 10 km and
  Mahalanobis distance uniform on ``[0, 2]``, so that essentially every case
  lies on the dilution branch (Mahalanobis below ``sqrt(2)``, see
  :mod:`conjprob.dilution`). Used as an out-of-distribution test set, never for
  training.
* ``"homogeneous"`` -- one object class and one covariance class: effective
  position uncertainty 400 m to 600 m, aspect ratio 1 to 1.5, hard-body radius
  9 m to 11 m. Here ``R / sqrt(sigma_x sigma_y)`` varies by less than a factor
  of two, which is the condition under which the Mahalanobis screen is close to
  sufficient. It exists so that the *reason* the Mahalanobis screen does badly
  on ``"nominal"`` is measured rather than argued.

Labels are ``Pc_foster > threshold`` with the Foster-Estes value computed by
adaptive quadrature, so the label is the operational default's own answer and
not an approximation of it.

Units: metres and metres per second in the record arrays; probabilities
dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .analytic import pc_foster
from .features import features_from_arrays

REGIMES: tuple[str, ...] = ("nominal", "dilution", "homogeneous")
"""Declared generator regimes."""

DEFAULT_PC_THRESHOLD = 1e-4
"""Pc threshold used for the screening label, dimensionless."""

_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "nominal": {
        "sigma_effective_m": (10.0, 3000.0),
        "aspect_ratio": (1.0, 20.0),
        "radius_m": (1.0, 30.0),
        "relative_speed_mps": (100.0, 15000.0),
        "sigma_out_ratio": (0.2, 5.0),
        "mahalanobis": (0.0, 5.0),
    },
    "dilution": {
        "sigma_effective_m": (300.0, 10000.0),
        "aspect_ratio": (1.0, 20.0),
        "radius_m": (1.0, 30.0),
        "relative_speed_mps": (100.0, 15000.0),
        "sigma_out_ratio": (0.2, 5.0),
        "mahalanobis": (0.0, 2.0),
    },
    "homogeneous": {
        "sigma_effective_m": (400.0, 600.0),
        "aspect_ratio": (1.0, 1.5),
        "radius_m": (9.0, 11.0),
        "relative_speed_mps": (100.0, 15000.0),
        "sigma_out_ratio": (0.2, 5.0),
        "mahalanobis": (0.0, 5.0),
    },
}


@dataclass(frozen=True)
class ConjunctionPopulation:
    """A generated population with its features, Pc values and labels.

    Attributes
    ----------
    sigma_x_m, sigma_y_m
        Principal standard deviations, metres, with ``sigma_x_m >= sigma_y_m``.
    xbar_m, ybar_m
        Miss-vector components in the principal frame, metres.
    radius_m
        Combined hard-body radius, metres.
    relative_speed_mps
        Relative speed, metres per second. A nuisance variable by construction.
    sigma_out_of_plane_m
        Out-of-plane standard deviation, metres. Also a nuisance variable.
    pc
        Foster-Estes collision probability, dimensionless.
    features
        Feature matrix, shape ``(n, 10)``, in :data:`conjprob.features.FEATURE_NAMES`
        order.
    labels
        ``pc > threshold``, boolean.
    threshold
        The Pc threshold used, dimensionless.
    regime
        Generator regime name.
    seed
        Seed used, so the population is reproducible exactly.
    """

    sigma_x_m: NDArray[np.float64]
    sigma_y_m: NDArray[np.float64]
    xbar_m: NDArray[np.float64]
    ybar_m: NDArray[np.float64]
    radius_m: NDArray[np.float64]
    relative_speed_mps: NDArray[np.float64]
    sigma_out_of_plane_m: NDArray[np.float64]
    pc: NDArray[np.float64]
    features: NDArray[np.float64]
    labels: NDArray[np.bool_]
    threshold: float
    regime: str
    seed: int

    def __len__(self) -> int:
        return int(self.pc.size)

    @property
    def positive_rate(self) -> float:
        """Fraction of the population exceeding the Pc threshold, dimensionless."""
        return float(np.mean(self.labels))


def _log_uniform(
    rng: np.random.Generator, low: float, high: float, size: int
) -> NDArray[np.float64]:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def generate_population(
    n_samples: int,
    *,
    seed: int = 0,
    regime: str = "nominal",
    threshold: float = DEFAULT_PC_THRESHOLD,
) -> ConjunctionPopulation:
    """Generate a reproducible synthetic conjunction population.

    Parameters
    ----------
    n_samples
        Number of conjunctions.
    seed
        Seed for :func:`numpy.random.default_rng`. The same seed and regime
        reproduce the same population bit for bit on the pinned numpy version.
    regime
        One of :data:`REGIMES`.
    threshold
        Pc threshold for the screening label, dimensionless.

    Returns
    -------
    ConjunctionPopulation

    Notes
    -----
    Cost is one adaptive quadrature per sample for the label. 40000 samples
    takes of order ten seconds on two cores, which is the whole dataset budget
    for this product.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    if regime not in _BOUNDS:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold!r}")
    bounds = _BOUNDS[regime]
    rng = np.random.default_rng(seed)
    sigma_eff = _log_uniform(rng, *bounds["sigma_effective_m"], n_samples)
    aspect = _log_uniform(rng, *bounds["aspect_ratio"], n_samples)
    radius = _log_uniform(rng, *bounds["radius_m"], n_samples)
    speed = _log_uniform(rng, *bounds["relative_speed_mps"], n_samples)
    out_ratio = _log_uniform(rng, *bounds["sigma_out_ratio"], n_samples)
    mahalanobis = rng.uniform(*bounds["mahalanobis"], n_samples)
    psi = rng.uniform(0.0, 2.0 * np.pi, n_samples)

    root = np.sqrt(aspect)
    sigma_x = sigma_eff * root
    sigma_y = sigma_eff / root
    xbar = mahalanobis * sigma_x * np.cos(psi)
    ybar = mahalanobis * sigma_y * np.sin(psi)
    sigma_out = sigma_eff * out_ratio

    pc = np.array(
        [
            pc_foster(float(sx), float(sy), float(xb), float(yb), float(r))
            for sx, sy, xb, yb, r in zip(sigma_x, sigma_y, xbar, ybar, radius, strict=True)
        ]
    )
    features = features_from_arrays(
        sigma_x, sigma_y, xbar, ybar, radius, speed, sigma_out
    )
    return ConjunctionPopulation(
        sigma_x_m=sigma_x,
        sigma_y_m=sigma_y,
        xbar_m=xbar,
        ybar_m=ybar,
        radius_m=radius,
        relative_speed_mps=speed,
        sigma_out_of_plane_m=sigma_out,
        pc=pc,
        features=features,
        labels=pc > threshold,
        threshold=float(threshold),
        regime=regime,
        seed=int(seed),
    )
