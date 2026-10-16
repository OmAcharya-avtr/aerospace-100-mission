"""Covered prior mass as a function of dwell index, and the spiral closed form.

Coverage here is *probability* coverage, not area coverage: the quantity of
interest is how much of the initial pointing-uncertainty distribution the union
of the dwell footprints has swept, because that is what bounds the probability
of acquisition.

References
----------
Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and
    Stochastic Processes*, 4th ed. McGraw-Hill, Ch. 6 (Rayleigh radial law).
Koopman, B. O. (1980). *Search and Screening*. Pergamon Press, Ch. 4-5
    (coverage of a sweep and the definition of search effort).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import PointingUncertainty
from .patterns import ScanPath

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class CoverageCurve:
    """Covered prior mass after each dwell of a path.

    ``dwell_index`` is 1-based. ``covered_mass`` is the Monte Carlo estimate of
    the prior mass inside the union of the first ``k`` footprint discs, and
    ``standard_error`` is its binomial standard error, both dimensionless.
    """

    dwell_index: np.ndarray
    covered_mass: np.ndarray
    standard_error: np.ndarray
    n_samples: int

    def final_coverage(self) -> float:
        """Covered mass after the whole path, dimensionless."""
        return float(self.covered_mass[-1]) if self.covered_mass.size else 0.0


def coverage_curve(
    path: ScanPath,
    uncertainty: PointingUncertainty,
    footprint_radius_rad: float,
    n_samples: int = 20000,
    seed: int = 20261016,
) -> CoverageCurve:
    """Monte Carlo covered prior mass after each dwell of ``path``.

    A sample is covered at dwell ``k`` when its distance to the ``k``-th dwell
    centre is at most ``footprint_radius_rad``. The estimator is the fraction
    of prior samples covered by the union of the first ``k`` discs; its
    standard error is ``sqrt(p (1 - p) / n)``. Units: radians in, dimensionless
    out. Seeded, so the curve is reproducible.
    """
    if footprint_radius_rad <= 0.0 or not math.isfinite(footprint_radius_rad):
        raise ValueError("footprint_radius_rad must be finite and > 0")
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    rng = np.random.default_rng(int(seed))
    samples = uncertainty.sample(int(n_samples), rng)
    uncovered = np.ones(int(n_samples), dtype=bool)
    counts = np.empty(path.n_points, dtype=float)
    covered_total = 0
    r2 = footprint_radius_rad**2
    for k in range(path.n_points):
        if covered_total < n_samples:
            delta = samples[uncovered] - path.points_rad[k]
            hit = (delta[:, 0] ** 2 + delta[:, 1] ** 2) <= r2
            if hit.any():
                index = np.nonzero(uncovered)[0][hit]
                uncovered[index] = False
                covered_total += int(hit.sum())
        counts[k] = covered_total
    fraction = counts / float(n_samples)
    se = np.sqrt(np.clip(fraction * (1.0 - fraction), 0.0, None) / float(n_samples))
    return CoverageCurve(
        dwell_index=np.arange(1, path.n_points + 1),
        covered_mass=fraction,
        standard_error=se,
        n_samples=int(n_samples),
    )


def spiral_closed_form_valid(
    pitch_rad: float,
    along_track_spacing_rad: float,
    footprint_radius_rad: float,
) -> bool:
    """Whether the spiral coverage closed form applies to this geometry.

    The union of discs of radius ``a`` centred every ``d`` along a track covers
    a strip of half-width ``sqrt(a^2 - (d/2)^2)``; consecutive turns separated
    radially by ``p`` leave no gap when

        p <= 2 * sqrt(a^2 - (d/2)^2)        and        d <= 2 a.

    Both are necessary for :func:`spiral_closed_form_coverage` to be an
    equality rather than an upper bound. Dimensionless result.
    """
    a = float(footprint_radius_rad)
    d = float(along_track_spacing_rad)
    p = float(pitch_rad)
    if a <= 0.0 or d <= 0.0 or p <= 0.0:
        raise ValueError("pitch, spacing and footprint radius must all be > 0")
    if d > 2.0 * a:
        return False
    return p <= 2.0 * math.sqrt(a**2 - (0.5 * d) ** 2)


def spiral_closed_form_coverage(
    pitch_rad: float,
    turns: int,
    footprint_radius_rad: float,
    uncertainty: PointingUncertainty,
) -> float:
    """Covered prior mass of a fully-covering Archimedean spiral, closed form.

    Derivation. With ``r = p theta / (2 pi)`` and an integer number of turns
    ``n``, the outermost pass at polar angle ``phi`` in [0, 2 pi) occurs at
    radius ``(n - 1) p + p phi / (2 pi)``, so the covered region at that angle
    is the disc sector out to ``R(phi) = (n - 1) p + p phi / (2 pi) + a``,
    provided consecutive turns do not leave a gap
    (:func:`spiral_closed_form_valid`). Averaging the Rayleigh CDF over angle
    and substituting ``u = R(phi)``:

        C = (1 / p) * integral_{u0}^{u1} [1 - exp(-u^2 / (2 sigma^2))] du
          = 1 - (sigma / p) sqrt(pi / 2) [erf(u1 / (sigma sqrt 2))
                                          - erf(u0 / (sigma sqrt 2))]

    with ``u0 = (n - 1) p + a`` and ``u1 = n p + a``.

    Units: all angles in radians, result dimensionless in [0, 1]. Validity:
    integer ``turns >= 1``, and the inter-turn condition above; outside it the
    expression is an upper bound on the true coverage and this function does
    not check it, because the check needs the along-track spacing that this
    signature does not take.

    Numerical validity range. The expression is a difference of two error
    functions divided by the pitch, so it loses significance when the pitch is
    very small compared with ``sigma``. Hypothesis found the worst case in
    this repository at ``pitch / sigma = 1.28e-10``, where the returned value
    is 0.3934688006264304 against an exact limit of 0.3934693402873666: an
    absolute error of 5.40e-7, which puts the value slightly *below* the
    disc bound it should never go below. The counterexample is pinned in
    ``tests/test_regressions.py``. For ``pitch / sigma >= 1e-6``, which covers
    every geometry a terminal would actually scan, the agreement with a direct
    quadrature of the same integral is better than 1e-12 relative.
    """
    if int(turns) != turns or turns < 1:
        raise ValueError(f"turns must be an integer >= 1, got {turns!r}")
    if pitch_rad <= 0.0 or footprint_radius_rad <= 0.0:
        raise ValueError("pitch_rad and footprint_radius_rad must be > 0")
    sigma = uncertainty.sigma_rad
    p = float(pitch_rad)
    a = float(footprint_radius_rad)
    u0 = (int(turns) - 1) * p + a
    u1 = int(turns) * p + a
    bracket = math.erf(u1 / (sigma * _SQRT2)) - math.erf(u0 / (sigma * _SQRT2))
    return 1.0 - (sigma / p) * math.sqrt(0.5 * math.pi) * bracket


def disc_coverage(radius_rad: float, uncertainty: PointingUncertainty) -> float:
    """Covered prior mass of a single disc of radius ``radius_rad`` at boresight.

    Rayleigh CDF, ``1 - exp(-r^2 / (2 sigma^2))``. Dimensionless.
    """
    return uncertainty.radial_cdf(radius_rad)
