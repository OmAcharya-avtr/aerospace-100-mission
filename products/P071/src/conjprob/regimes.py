"""Measured validity regimes for the three analytic Pc paths.

The Pc integral is invariant under a common rescaling of the hard-body radius,
the miss vector and both principal standard deviations, so the whole family of
encounters is described by three dimensionless numbers and one angle:

* ``rho = R / sqrt(sigma_x sigma_y)``, the hard body measured against the
  effective position uncertainty;
* ``a = sigma_x / sigma_y >= 1``, the covariance aspect ratio;
* ``M``, the Mahalanobis distance of the miss vector;
* ``psi``, the direction of the miss vector in the principal frame, defined by
  ``xbar = M sigma_x cos(psi)``, ``ybar = M sigma_y sin(psi)``, which keeps the
  Mahalanobis distance equal to ``M`` for every ``psi``.

This module evaluates each analytic method against the independent 2-D
quadrature reference over a grid in those parameters and reduces the result to
**measured boundary curves**: for each ``(a, M)`` the value of ``rho`` at which
a method's relative error first exceeds a stated tolerance. Nothing here
asserts a regime; everything is interpolated from values that were computed.

Units: the sweep is dimensionless. :func:`geometry_from_dimensionless` returns
metres under a declared effective sigma so that the same parameters can be fed
to the Monte Carlo reference, which is not scale free in its sample count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .analytic import (
    chan_recommended_order,
    pc_alfano,
    pc_chan,
    pc_foster,
    pc_small_body_approximation,
)
from .nonlinear import first_crossing
from .quadrature import pc_quadrature

METHOD_NAMES: tuple[str, ...] = ("foster", "chan", "alfano", "small_body")
"""Analytic methods compared against the quadrature reference, in report order."""


def geometry_from_dimensionless(
    radius_over_sigma: float,
    aspect_ratio: float,
    mahalanobis: float,
    *,
    psi_rad: float = 0.25 * math.pi,
    sigma_effective_m: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Encounter-plane geometry in metres from the four dimensionless parameters.

    Returns ``(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)``. The returned
    principal standard deviations satisfy
    ``sqrt(sigma_x sigma_y) = sigma_effective_m`` and
    ``sigma_x / sigma_y = aspect_ratio`` exactly in exact arithmetic, and the
    returned miss vector has Mahalanobis distance ``mahalanobis``.

    Raises
    ------
    ValueError
        If the aspect ratio is below one, the radius ratio is negative, the
        Mahalanobis distance is negative, or the effective sigma is not
        positive.
    """
    if aspect_ratio < 1.0:
        raise ValueError(f"aspect_ratio must be >= 1, got {aspect_ratio!r}")
    if radius_over_sigma < 0.0:
        raise ValueError(f"radius_over_sigma must be non-negative, got {radius_over_sigma!r}")
    if mahalanobis < 0.0:
        raise ValueError(f"mahalanobis must be non-negative, got {mahalanobis!r}")
    if sigma_effective_m <= 0.0:
        raise ValueError(f"sigma_effective_m must be positive, got {sigma_effective_m!r}")
    root = math.sqrt(aspect_ratio)
    sigma_x = sigma_effective_m * root
    sigma_y = sigma_effective_m / root
    xbar = mahalanobis * sigma_x * math.cos(psi_rad)
    ybar = mahalanobis * sigma_y * math.sin(psi_rad)
    return sigma_x, sigma_y, xbar, ybar, radius_over_sigma * sigma_effective_m


def evaluate_methods(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    chan_order: int | None = None,
    alfano_nodes: int = 201,
) -> dict[str, float]:
    """Every analytic Pc plus the quadrature reference for one geometry.

    Returns a dictionary with the four analytic values, the key ``reference``
    for the 2-D quadrature result and ``reference_abserr`` for the quadrature
    routine's own error estimate. All values dimensionless. ``chan_order`` of
    ``None`` uses :func:`conjprob.analytic.chan_recommended_order`, so that the
    measured Chan error is its equivalent-area error and not a truncation
    artefact.
    """
    if chan_order is None:
        chan_order = chan_recommended_order(
            math.hypot(xbar_m / sigma_x_m, ybar_m / sigma_y_m)
        )
    reference, abserr = pc_quadrature(
        np.diag([sigma_x_m**2, sigma_y_m**2]), (xbar_m, ybar_m), radius_m
    )
    return {
        "foster": pc_foster(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m),
        "chan": pc_chan(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m, order=chan_order),
        "alfano": pc_alfano(
            sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m, nodes=alfano_nodes
        ),
        "small_body": pc_small_body_approximation(
            sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
        ),
        "reference": reference,
        "reference_abserr": abserr,
    }


@dataclass
class RegimeSweep:
    """Relative errors of each analytic method over the dimensionless grid.

    Attributes
    ----------
    radius_over_sigma, aspect_ratios, mahalanobis_values
        The grid axes, dimensionless.
    psi_rad
        Miss-vector direction used throughout the sweep, radians.
    reference
        Quadrature Pc, shape ``(n_aspect, n_mahalanobis, n_rho)``.
    errors
        Relative error of each method against ``reference``, same shape, keyed
        by method name. Signed: positive means the method overestimates.
    """

    radius_over_sigma: NDArray[np.float64]
    aspect_ratios: NDArray[np.float64]
    mahalanobis_values: NDArray[np.float64]
    psi_rad: float
    reference: NDArray[np.float64]
    errors: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def boundary(self, method: str, tolerance: float) -> NDArray[np.float64]:
        """Measured ``rho`` at which ``method`` first exceeds ``tolerance``.

        Returns an array of shape ``(n_aspect, n_mahalanobis)``. Entries are
        ``nan`` where the tolerance is never exceeded across the swept range of
        ``rho``, which is reported rather than clipped.
        """
        if method not in self.errors:
            raise KeyError(f"unknown method {method!r}; have {sorted(self.errors)}")
        errors = self.errors[method]
        out = np.full(errors.shape[:2], np.nan)
        for i in range(errors.shape[0]):
            for j in range(errors.shape[1]):
                out[i, j] = first_crossing(
                    self.radius_over_sigma, errors[i, j, :], tolerance
                )
        return out

    def worst_error(self, method: str) -> float:
        """Largest absolute relative error of ``method`` anywhere on the grid."""
        return float(np.nanmax(np.abs(self.errors[method])))


def regime_sweep(
    radius_over_sigma: NDArray[np.float64],
    aspect_ratios: NDArray[np.float64],
    mahalanobis_values: NDArray[np.float64],
    *,
    psi_rad: float = 0.25 * math.pi,
    chan_order: int | None = None,
    alfano_nodes: int = 201,
) -> RegimeSweep:
    """Evaluate every analytic method against quadrature over the parameter grid.

    The grid is the Cartesian product of the three axes. Cost is one adaptive
    2-D quadrature per grid point, which dominates; a 5 x 5 x 25 grid is about
    600 quadratures and runs in a few seconds on two cores.
    """
    rho = np.asarray(radius_over_sigma, dtype=float)
    aspects = np.asarray(aspect_ratios, dtype=float)
    mahalanobis = np.asarray(mahalanobis_values, dtype=float)
    for name, axis in (
        ("radius_over_sigma", rho),
        ("aspect_ratios", aspects),
        ("mahalanobis_values", mahalanobis),
    ):
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
    shape = (aspects.size, mahalanobis.size, rho.size)
    reference = np.empty(shape)
    errors = {name: np.empty(shape) for name in METHOD_NAMES}
    for i, aspect in enumerate(aspects):
        for j, mdist in enumerate(mahalanobis):
            for k, ratio in enumerate(rho):
                geometry = geometry_from_dimensionless(
                    float(ratio), float(aspect), float(mdist), psi_rad=psi_rad
                )
                values = evaluate_methods(
                    *geometry, chan_order=chan_order, alfano_nodes=alfano_nodes
                )
                ref = values["reference"]
                reference[i, j, k] = ref
                for name in METHOD_NAMES:
                    errors[name][i, j, k] = (
                        (values[name] - ref) / ref if ref > 0.0 else np.nan
                    )
    return RegimeSweep(
        radius_over_sigma=rho,
        aspect_ratios=aspects,
        mahalanobis_values=mahalanobis,
        psi_rad=psi_rad,
        reference=reference,
        errors=errors,
    )
