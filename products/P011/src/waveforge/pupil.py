"""Pupil sampling grid for an adaptive-optics system.

A :class:`Pupil` is a square Cartesian sampling of a circular telescope /
terminal aperture of diameter ``D`` [m], optionally with a circular central
obscuration. Everything downstream in ``waveforge`` -- Zernike polynomials,
phase screens, the Shack-Hartmann model, the deformable mirror and the Strehl
metrics -- is defined on this grid.

Conventions
-----------
* Physical pupil coordinates ``(X, Y)`` are in metres, measured from the pupil
  centre; ``+x`` is increasing column index, ``+y`` is increasing row index.
  Arrays are stored ``a[row, col] == a[y, x]``.
* Normalised pupil coordinates are ``(x, y) = (X, Y) / (D/2)``, so the pupil
  edge is at radius ``rho = 1``. Zernike polynomials use these.
* Phase is **optical phase in radians** at the working wavelength. Optical path
  difference in metres is ``OPD = phase * lambda / (2*pi)``.
* Wavefront **slope** is the OPD gradient ``dOPD/dX`` [rad of ray angle, which
  is numerically m/m in the small-angle limit].

Sampling requirement
--------------------
The grid pitch ``dx = D / n_grid`` must resolve the smallest structure the
model claims to represent. For a Kolmogorov screen the relevant scale is the
Fried parameter ``r0``; for the deformable mirror it is the actuator pitch.
:meth:`Pupil.check_sampling` enforces a documented minimum of 2 samples per
``r0`` and 4 samples per actuator pitch, which are the usual rules of thumb in
AO simulation practice (see Schmidt, J. D. 2010, *Numerical Simulation of
Optical Wave Propagation with Examples in MATLAB*, SPIE Press, ch. 9, on
sampling phase screens).

References
----------
Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford
University Press -- AO system conventions, chapters 3, 5, 7 and 9.
Noll, R. J. (1976), "Zernike polynomials and atmospheric turbulence",
*Journal of the Optical Society of America* **66**, 207-211.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["Pupil"]


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


@dataclass(frozen=True)
class Pupil:
    """Square sampling grid over a circular aperture.

    Parameters
    ----------
    diameter:
        Aperture diameter ``D`` [m]. Must be > 0.
    n_grid:
        Number of samples across the full diameter [-]. Must be >= 8. The grid
        spans exactly ``[-D/2, +D/2]`` in each axis with pitch ``D / n_grid``,
        sample centres offset by half a pixel from the array edges.
    obscuration:
        Linear central-obscuration ratio ``eps = d_inner / D`` [-], in
        ``[0, 0.9]``. Default 0 (unobscured).

    Attributes
    ----------
    dx:
        Grid pitch [m].
    mask:
        Boolean ``(n_grid, n_grid)`` array, True inside the clear aperture.
    n_valid:
        Number of True entries in ``mask`` [-].
    area:
        Geometric clear area ``pi/4 * D^2 * (1 - eps^2)`` [m^2].
    """

    diameter: float
    n_grid: int
    obscuration: float = 0.0

    _cache: dict[str, NDArray[np.float64]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "diameter", _check_positive("diameter", self.diameter))
        n = int(self.n_grid)
        if n < 8:
            raise ValueError(f"n_grid must be >= 8, got {n}")
        object.__setattr__(self, "n_grid", n)
        eps = float(self.obscuration)
        if not np.isfinite(eps) or not (0.0 <= eps <= 0.9):
            raise ValueError(f"obscuration must be in [0, 0.9], got {self.obscuration!r}")
        object.__setattr__(self, "obscuration", eps)

    # ---------------------------------------------------------------- geometry
    @property
    def dx(self) -> float:
        """Grid pitch [m]."""
        return self.diameter / self.n_grid

    def coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Physical pupil coordinate grids ``(X, Y)`` [m], shape ``(n, n)``."""
        if "X" not in self._cache:
            n = self.n_grid
            axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.dx
            x, y = np.meshgrid(axis, axis, indexing="xy")
            self._cache["X"] = x
            self._cache["Y"] = y
        return self._cache["X"], self._cache["Y"]

    def normalized_coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised coordinates ``(x, y)`` with the pupil edge at radius 1 [-]."""
        x, y = self.coords()
        r_out = self.diameter / 2.0
        return x / r_out, y / r_out

    def polar(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised polar coordinates ``(rho, theta)``; ``rho`` [-], ``theta`` [rad]."""
        x, y = self.normalized_coords()
        return np.hypot(x, y), np.arctan2(y, x)

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Boolean clear-aperture mask, True inside the annulus."""
        if "mask" not in self._cache:
            rho, _ = self.polar()
            m = (rho <= 1.0) & (rho >= self.obscuration)
            self._cache["mask"] = m
        return self._cache["mask"]  # type: ignore[return-value]

    @property
    def n_valid(self) -> int:
        """Number of grid samples inside the clear aperture [-]."""
        return int(np.count_nonzero(self.mask))

    @property
    def area(self) -> float:
        """Geometric clear area [m^2]: ``pi/4 D^2 (1 - eps^2)``."""
        return np.pi / 4.0 * self.diameter**2 * (1.0 - self.obscuration**2)

    # ----------------------------------------------------------------- helpers
    def masked_mean(self, values: NDArray[np.float64]) -> float:
        """Mean of ``values`` over the clear aperture [same units as input]."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (self.n_grid, self.n_grid):
            raise ValueError(
                f"values must have shape {(self.n_grid, self.n_grid)}, got {arr.shape}"
            )
        return float(arr[self.mask].mean())

    def piston_removed(self, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``phase`` with its aperture-average (piston) subtracted.

        Outside the aperture the result is set to 0. Piston carries no imaging
        information, so every variance and Strehl figure in ``waveforge`` is
        computed after piston removal.
        """
        arr = np.asarray(phase, dtype=np.float64)
        out = np.zeros_like(arr)
        m = self.mask
        out[m] = arr[m] - arr[m].mean()
        return out

    def variance(self, phase: NDArray[np.float64]) -> float:
        """Piston-removed spatial variance over the aperture [rad^2].

        This is the quantity that enters the Marechal Strehl approximation
        (Born & Wolf 1999, *Principles of Optics*, 7th ed., sec. 9.1.3).
        """
        arr = np.asarray(phase, dtype=np.float64)
        if arr.shape != (self.n_grid, self.n_grid):
            raise ValueError(
                f"phase must have shape {(self.n_grid, self.n_grid)}, got {arr.shape}"
            )
        vals = arr[self.mask]
        return float(np.mean((vals - vals.mean()) ** 2))

    def check_sampling(self, r0: float | None = None, actuator_pitch: float | None = None) -> None:
        """Raise ``ValueError`` if the grid under-samples ``r0`` or the DM pitch.

        Criteria (AO simulation rules of thumb, Schmidt 2010 ch. 9):
        at least 2 grid samples per Fried parameter and at least 4 grid samples
        per actuator pitch.
        """
        if r0 is not None:
            r0 = _check_positive("r0", r0)
            if r0 / self.dx < 2.0:
                raise ValueError(
                    f"grid under-samples turbulence: r0/dx = {r0 / self.dx:.2f} < 2 "
                    f"(r0={r0:.4g} m, dx={self.dx:.4g} m). Increase n_grid."
                )
        if actuator_pitch is not None:
            actuator_pitch = _check_positive("actuator_pitch", actuator_pitch)
            if actuator_pitch / self.dx < 4.0:
                raise ValueError(
                    f"grid under-samples the DM: pitch/dx = {actuator_pitch / self.dx:.2f} < 4 "
                    f"(pitch={actuator_pitch:.4g} m, dx={self.dx:.4g} m). Increase n_grid."
                )
