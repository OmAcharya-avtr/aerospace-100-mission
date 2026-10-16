"""Pointing-uncertainty geometry for an optical-terminal acquisition search.

Angles are plane angles in radians throughout. The two-axis pointing error is
treated as a point in a local tangent plane (``x`` = cross-elevation, ``y`` =
elevation); for the angles of interest here (hundreds of microradians) the
difference between that plane and the sphere is below 1e-10 rad and is ignored.

References
----------
Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and
    Stochastic Processes*, 4th ed. McGraw-Hill. Ch. 6: the radial marginal of
    an isotropic bivariate normal is Rayleigh distributed.
Siegman, A. E. (1986). *Lasers*. University Science Books, Ch. 17: the
    far-field irradiance of a TEM00 Gaussian beam falls as exp(-2 rho^2 / w^2),
    where ``w`` is the 1/e^2 half-width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT2 = math.sqrt(2.0)


def _standard_normal_cdf(z: float) -> float:
    """Phi(z) for the standard normal, via ``math.erf``. Dimensionless."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


@dataclass(frozen=True)
class PointingUncertainty:
    """Isotropic 2-D Gaussian prior over the initial pointing error.

    Parameters
    ----------
    sigma_rad:
        Per-axis 1-sigma pointing uncertainty, radians. Both axes share this
        value and are assumed independent; a correlated or anisotropic cone is
        not modelled (see README Limitations).

    Assumptions
    -----------
    Zero mean (the open-loop pointing estimate is unbiased), independent axes,
    stationary over the whole search. Validity range: ``sigma_rad > 0``; the
    small-angle tangent-plane treatment is valid for ``sigma_rad`` well below
    1e-2 rad.
    """

    sigma_rad: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_rad) or self.sigma_rad <= 0.0:
            raise ValueError(f"sigma_rad must be finite and > 0, got {self.sigma_rad!r}")

    def radial_cdf(self, radius_rad: float) -> float:
        """P(|error| <= radius). Rayleigh CDF ``1 - exp(-r^2 / (2 sigma^2))``.

        Source: Papoulis & Pillai 2002 Ch. 6. Units: ``radius_rad`` in radians,
        result dimensionless in [0, 1]. Valid for ``radius_rad >= 0``.
        """
        if radius_rad < 0.0:
            raise ValueError(f"radius_rad must be >= 0, got {radius_rad!r}")
        return 1.0 - math.exp(-(radius_rad**2) / (2.0 * self.sigma_rad**2))

    def rectangle_mass(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """Prior mass of the axis-aligned rectangle [x0, x1] x [y0, y1], radians.

        Exact for the independent-axis Gaussian: a product of two normal CDF
        differences. Dimensionless result in [0, 1].
        """
        if x1 < x0 or y1 < y0:
            raise ValueError("rectangle bounds must satisfy x1 >= x0 and y1 >= y0")
        s = self.sigma_rad
        px = _standard_normal_cdf(x1 / s) - _standard_normal_cdf(x0 / s)
        py = _standard_normal_cdf(y1 / s) - _standard_normal_cdf(y0 / s)
        return px * py

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` pointing errors. Returns an ``(n, 2)`` array in radians."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        return rng.normal(0.0, self.sigma_rad, size=(int(n), 2))


@dataclass(frozen=True)
class TerminalGeometry:
    """Declared beam, detector and gimbal geometry of the scanning terminal.

    Parameters
    ----------
    beam_divergence_rad:
        Full transmit divergence at the 1/e^2 irradiance points, radians.
        The 1/e^2 half-width is ``beam_divergence_rad / 2`` (Siegman 1986).
    detector_fov_rad:
        Full acquisition-detector field of view, radians.
    slew_rate_rad_s:
        Constant angular rate used to repoint between dwells, rad/s. A
        trapezoidal rate profile is not modelled; see README Limitations.
    settle_time_s:
        Fixed settling time charged after every repoint, seconds.
    """

    beam_divergence_rad: float
    detector_fov_rad: float
    slew_rate_rad_s: float
    settle_time_s: float

    def __post_init__(self) -> None:
        for name in ("beam_divergence_rad", "detector_fov_rad", "slew_rate_rad_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if not math.isfinite(self.settle_time_s) or self.settle_time_s < 0.0:
            raise ValueError(f"settle_time_s must be finite and >= 0, got {self.settle_time_s!r}")

    @property
    def beam_half_angle_rad(self) -> float:
        """Gaussian-beam 1/e^2 half-width ``w``, radians."""
        return 0.5 * self.beam_divergence_rad

    @property
    def footprint_radius_rad(self) -> float:
        """Radius of the angular region a single dwell can acquire, radians.

        Taken as ``min(beam half-angle, detector FOV half-angle)``: the target
        must be inside the illuminated spot and inside the detector field at
        the same time. This is a hard geometric cut used for coverage; the
        graded detection probability inside it is handled in
        :mod:`acqsearch.detection`.
        """
        return min(self.beam_half_angle_rad, 0.5 * self.detector_fov_rad)

    def slew_time_s(self, distance_rad: float | np.ndarray) -> float | np.ndarray:
        """Time to repoint through ``distance_rad``, seconds.

        ``distance / slew_rate + settle_time``. Constant-rate assumption; valid
        when the commanded step is large compared with the rate-limited
        acceleration distance of the steering mechanism, which is not modelled.
        """
        distance = np.asarray(distance_rad, dtype=float)
        if np.any(distance < 0.0):
            raise ValueError("distance_rad must be >= 0")
        out = distance / self.slew_rate_rad_s + self.settle_time_s
        if np.isscalar(distance_rad) or np.ndim(distance_rad) == 0:
            return float(out)
        return out


@dataclass(frozen=True)
class CellGrid:
    """Square-cell partition of the search region used by the analytic model.

    Cells are squares of side ``side_rad = footprint_radius * sqrt(2)``, the
    square inscribed in the footprint disc, so a dwell centred on a cell covers
    that cell entirely. ``prior_mass`` is the exact bivariate-normal mass of
    each square; ``mass_outside`` is the prior mass that falls outside the
    retained cells and is, by construction, never acquired by a scan confined
    to the grid.
    """

    centres_rad: np.ndarray
    side_rad: float
    prior_mass: np.ndarray
    mass_outside: float
    extent_rad: float

    @property
    def n_cells(self) -> int:
        """Number of cells in the partition."""
        return int(self.centres_rad.shape[0])

    @property
    def radii_rad(self) -> np.ndarray:
        """Distance of each cell centre from boresight, radians."""
        return np.hypot(self.centres_rad[:, 0], self.centres_rad[:, 1])

    def nearest_cell(self, points_rad: np.ndarray) -> np.ndarray:
        """Index of the nearest cell centre for each point in an ``(n, 2)`` array."""
        pts = np.atleast_2d(np.asarray(points_rad, dtype=float))
        if pts.shape[1] != 2:
            raise ValueError("points_rad must have shape (n, 2)")
        d2 = ((pts[:, None, :] - self.centres_rad[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)


def build_cell_grid(
    uncertainty: PointingUncertainty,
    geometry: TerminalGeometry,
    extent_sigma: float = 2.5,
) -> CellGrid:
    """Build the square-cell partition out to ``extent_sigma`` standard deviations.

    Parameters
    ----------
    extent_sigma:
        Search-region radius in units of ``uncertainty.sigma_rad``. The prior
        mass beyond it is reported as ``CellGrid.mass_outside`` and is treated
        as unacquirable, which is the honest accounting: a scan that does not
        go there cannot find a target there.
    """
    if not math.isfinite(extent_sigma) or extent_sigma <= 0.0:
        raise ValueError(f"extent_sigma must be finite and > 0, got {extent_sigma!r}")
    side = geometry.footprint_radius_rad * _SQRT2
    extent = extent_sigma * uncertainty.sigma_rad
    half_n = int(math.ceil(extent / side))
    indices = np.arange(-half_n, half_n + 1, dtype=float)
    gx, gy = np.meshgrid(indices * side, indices * side, indexing="ij")
    centres = np.column_stack([gx.ravel(), gy.ravel()])
    keep = np.hypot(centres[:, 0], centres[:, 1]) <= extent
    centres = centres[keep]
    if centres.shape[0] == 0:
        raise ValueError("cell grid is empty; extent_sigma is too small for this footprint")
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    centres = centres[order]
    half = 0.5 * side
    mass = np.array(
        [
            uncertainty.rectangle_mass(cx - half, cx + half, cy - half, cy + half)
            for cx, cy in centres
        ],
        dtype=float,
    )
    return CellGrid(
        centres_rad=centres,
        side_rad=side,
        prior_mass=mass,
        mass_outside=float(max(0.0, 1.0 - mass.sum())),
        extent_rad=float(extent),
    )
