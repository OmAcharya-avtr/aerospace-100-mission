"""Ground truth 1: direct two-dimensional quadrature on the encounter plane.

This module integrates the bivariate normal density over the hard-body disc
*without* diagonalising the covariance and *without* the error-function
reduction used by :mod:`conjprob.analytic`. It therefore shares no algebra with
the three analytic paths beyond the definition of the integral itself, which is
what makes it usable as a reference for them.

The integral is taken in polar coordinates centred on the primary,

.. math::

    P_c = \\int_0^R \\int_0^{2\\pi} f(r\\cos\\theta, r\\sin\\theta)\\, r
          \\,\\mathrm{d}\\theta\\,\\mathrm{d}r ,

with ``f`` the two-dimensional Gaussian density with full covariance
``C_p`` and mean equal to the miss vector. Polar coordinates are used rather
than Cartesian because the Cartesian form has a square-root boundary whose
derivative is singular at ``x = +/- R``, which costs adaptive quadrature
several digits.

Units: metres for ``radius_m`` and the miss vector, metres squared for the
covariance; the result is dimensionless.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import dblquad

_TWO_PI = 2.0 * math.pi


def pc_quadrature(
    covariance_2d_m2: ArrayLike,
    miss_2d_m: ArrayLike,
    radius_m: float,
    *,
    epsabs: float = 0.0,
    epsrel: float = 1e-11,
) -> tuple[float, float]:
    """Pc by adaptive 2-D quadrature, with the quadrature's own error estimate.

    Parameters
    ----------
    covariance_2d_m2
        ``(2, 2)`` projected covariance in any in-plane basis, metres squared.
        It is used as supplied: no eigendecomposition is performed.
    miss_2d_m
        Miss vector in the same basis, metres, shape ``(2,)``.
    radius_m
        Combined hard-body radius, metres.
    epsabs, epsrel
        Absolute and relative tolerances passed to
        :func:`scipy.integrate.dblquad`. ``epsabs`` defaults to zero so that
        the tolerance is purely relative and the result is scale invariant; a
        non-zero absolute tolerance makes the reference stop immediately for any
        Pc near or below it, which is the regime this product spends most of its
        time in.

    Returns
    -------
    tuple of float
        ``(pc, abserr)``. ``pc`` is dimensionless in ``[0, 1]``; ``abserr`` is
        the quadrature routine's own absolute error estimate and is reported
        alongside every use of this function as a reference, so that an
        analytic-versus-reference discrepancy can be attributed.

    Raises
    ------
    ValueError
        If the covariance is not a positive-definite 2x2 matrix or the radius
        is negative.
    """
    cov = np.asarray(covariance_2d_m2, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covariance_2d_m2 must have shape (2, 2), got {cov.shape}")
    sym = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(sym)):
        raise ValueError("covariance_2d_m2 must be finite")
    determinant = float(np.linalg.det(sym))
    eigenvalues = np.linalg.eigvalsh(sym)
    if eigenvalues.min() <= 0.0 or determinant <= 0.0:
        raise ValueError(
            "covariance_2d_m2 must be positive definite; eigenvalues are "
            f"{eigenvalues[0]:.6e}, {eigenvalues[1]:.6e} m^2"
        )
    miss = np.asarray(miss_2d_m, dtype=float).reshape(-1)
    if miss.shape != (2,):
        raise ValueError(f"miss_2d_m must have 2 components, got shape {miss.shape}")
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError(f"radius_m must be a non-negative finite number, got {radius_m!r}")
    if radius_m == 0.0:
        return 0.0, 0.0

    inverse = np.linalg.inv(sym)
    a, b, _, c = inverse.ravel()
    norm = 1.0 / (_TWO_PI * math.sqrt(determinant))
    mx, my = float(miss[0]), float(miss[1])

    def density_times_r(theta: float, r: float) -> float:
        dx = r * math.cos(theta) - mx
        dy = r * math.sin(theta) - my
        exponent = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
        return norm * math.exp(exponent) * r

    value, abserr = dblquad(
        density_times_r,
        0.0,
        radius_m,
        0.0,
        _TWO_PI,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(min(max(value, 0.0), 1.0)), float(abserr)
