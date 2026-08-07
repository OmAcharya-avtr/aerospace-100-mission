"""Analytic wavefront gradient fields sampled at the subaperture centres.

These are *not* an atmosphere model. They are exactly-known gradient fields
used as inputs to the sensor simulation and as known-answer references for
validation. Turbulence phase screens are out of scope for this product
(see README, Related work).

Convention: a wavefront is an optical path difference ``W(X, Y)`` in metres
over pupil coordinates in metres; the slope reported for a subaperture is the
gradient at its centre, ``(dW/dX, dW/dY)``, in radians (dimensionless m/m).
Using the centre value rather than the area average is exact for a linear
wavefront (tilt) and is a first-order approximation otherwise; for the
quadratic (defocus) field the centre gradient equals the area average as well,
because the gradient is itself linear in position.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometry import LensletArray

__all__ = ["tilt_slopes", "defocus_slopes", "random_slopes", "slope_rms"]


def tilt_slopes(
    array: LensletArray, gx: float, gy: float = 0.0
) -> NDArray[np.float64]:
    """Uniform slope field from a global tilt ``W(X, Y) = gx * X + gy * Y``.

    Since the gradient of a linear wavefront is constant, **every** illuminated
    subaperture must report exactly ``(gx, gy)``. This is the primary
    known-answer test of the whole sensor chain (validation §1).

    Parameters
    ----------
    array: lenslet geometry.
    gx, gy: wavefront gradient components [rad]. ``|g|`` should stay below
        ``array.max_slope`` or the spot leaves its own pixel block.

    Returns
    -------
    ``(n_valid, 2)`` array of slopes [rad], all rows identical.
    """
    for name, value in (("gx", gx), ("gy", gy)):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    return np.tile([float(gx), float(gy)], (array.n_valid, 1))


def defocus_slopes(array: LensletArray, curvature: float) -> NDArray[np.float64]:
    """Slope field of a quadratic wavefront ``W(X, Y) = c (X^2 + Y^2)``.

    Gradient: ``(dW/dX, dW/dY) = 2c (X, Y)`` — a radial fan whose magnitude
    grows linearly with pupil radius. Useful as a non-uniform but exactly known
    field.

    Parameters
    ----------
    array: lenslet geometry.
    curvature: ``c`` [m^-1]. A wavefront with peak-to-valley ``PV`` over a
        pupil of radius ``R`` has ``c = PV / R^2``.

    Returns
    -------
    ``(n_valid, 2)`` slopes [rad].
    """
    c = float(curvature)
    if not np.isfinite(c):
        raise ValueError(f"curvature must be finite, got {curvature!r}")
    centres = array.valid_centres()
    return 2.0 * c * centres


def random_slopes(
    array: LensletArray, rms: float, seed: int | np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Spatially uncorrelated Gaussian slopes of the given per-axis RMS [rad].

    **This is white noise, not turbulence.** Atmospheric slopes are strongly
    spatially correlated (Kolmogorov statistics); this helper deliberately
    models no such correlation and must not be read as an atmospheric case.
    It exists to exercise the sensor with a non-trivial, reproducible field.
    """
    r = float(rms)
    if not np.isfinite(r) or r < 0:
        raise ValueError(f"rms must be finite and >= 0, got {rms!r}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.normal(0.0, r, size=(array.n_valid, 2))


def slope_rms(slopes: NDArray[np.float64]) -> float:
    """Per-axis RMS of a slope field [rad]: ``sqrt(mean(g^2))`` over all components."""
    s = np.asarray(slopes, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"slopes must have shape (n, 2), got {s.shape}")
    return float(np.sqrt(np.mean(s**2)))
