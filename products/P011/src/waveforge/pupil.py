"""Pupil sampling grids and basic wavefront statistics.

All wavefronts in :mod:`waveforge` are stored as 2-D arrays of **phase in
radians** sampled on a square Cartesian grid, together with a boolean mask
marking the illuminated pupil.  Optical path difference (OPD) in metres is
``opd = phase * lambda / (2 * pi)``.

Conventions
-----------
* ``x`` increases with column index, ``y`` increases with row index.
* The pupil is centred on the grid; normalised radius ``rho = r / R`` where
  ``R = D / 2`` is the pupil radius.
* A grid of ``n`` samples across a pupil diameter ``D`` has sample spacing
  ``d = D / n`` metres.  Sample centres sit at ``(i + 0.5 - n/2) * d`` so the
  grid is symmetric about the origin and no sample lands exactly on ``r = 0``.

Reference for the circular-pupil / Zernike convention: R. J. Noll, "Zernike
polynomials and atmospheric turbulence", *J. Opt. Soc. Am.* **66**(3),
207-211 (1976).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PupilGrid",
    "piston_removed",
    "rms",
    "strehl_from_field",
    "variance",
]


@dataclass(frozen=True)
class PupilGrid:
    """A square Cartesian sampling grid carrying a circular pupil mask.

    Parameters
    ----------
    n_pix:
        Number of samples across one side of the square grid (``n_pix >= 2``).
    diameter_m:
        Pupil (telescope / aperture) diameter ``D`` in metres, ``D > 0``.
    obscuration:
        Central obscuration as a fraction of the pupil radius, in ``[0, 1)``.
        ``0`` (default) is an unobscured circular pupil.

    Notes
    -----
    The grid spans exactly the pupil diameter, so the sample spacing is
    ``d = D / n_pix`` metres and the pupil is inscribed in the array.
    """

    n_pix: int
    diameter_m: float
    obscuration: float = 0.0

    def __post_init__(self) -> None:
        if int(self.n_pix) != self.n_pix or self.n_pix < 2:
            raise ValueError(f"n_pix must be an integer >= 2, got {self.n_pix!r}")
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0.0:
            raise ValueError(f"diameter_m must be finite and > 0, got {self.diameter_m!r}")
        if not (0.0 <= self.obscuration < 1.0):
            raise ValueError(f"obscuration must lie in [0, 1), got {self.obscuration!r}")

    # -- geometry ---------------------------------------------------------
    @property
    def radius_m(self) -> float:
        """Pupil radius ``R = D / 2`` [m]."""
        return 0.5 * self.diameter_m

    @property
    def sample_spacing_m(self) -> float:
        """Grid sample spacing ``d = D / n_pix`` [m]."""
        return self.diameter_m / self.n_pix

    def coords_m(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` sample-centre coordinate arrays in metres."""
        d = self.sample_spacing_m
        axis = (np.arange(self.n_pix) + 0.5 - 0.5 * self.n_pix) * d
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return x, y

    def normalised_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x/R, y/R)`` — dimensionless pupil coordinates."""
        x, y = self.coords_m()
        return x / self.radius_m, y / self.radius_m

    def polar(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rho, theta)``: normalised radius and angle [rad] from +x."""
        xn, yn = self.normalised_coords()
        return np.hypot(xn, yn), np.arctan2(yn, xn)

    @property
    def mask(self) -> np.ndarray:
        """Boolean illuminated-pupil mask (``True`` inside the aperture)."""
        rho, _ = self.polar()
        inside = rho <= 1.0
        if self.obscuration > 0.0:
            inside &= rho >= self.obscuration
        return inside

    @property
    def n_valid(self) -> int:
        """Number of illuminated samples."""
        return int(np.count_nonzero(self.mask))

    @property
    def area_m2(self) -> float:
        """Illuminated area estimated from the sample count [m^2]."""
        return self.n_valid * self.sample_spacing_m**2


def _masked(phase: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    if mask is None:
        return phase.ravel()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != phase.shape:
        raise ValueError(f"mask shape {mask.shape} does not match phase shape {phase.shape}")
    if not mask.any():
        raise ValueError("mask selects no samples")
    return phase[mask]


def piston_removed(phase: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Subtract the mean over the mask (piston term).

    Piston is unobservable in an imaging or heterodyne system and carries
    infinite power in the Kolmogorov spectrum, so it is removed everywhere in
    this package.  Values outside the mask are set to zero.
    """
    phase = np.asarray(phase, dtype=float)
    if mask is None:
        return phase - phase.mean()
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(phase)
    out[mask] = phase[mask] - phase[mask].mean()
    return out


def variance(phase: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Spatial variance of the phase over the mask [rad^2].

    This is the piston-removed mean-square phase, the quantity that enters the
    Marechal approximation.
    """
    return float(np.var(_masked(phase, mask)))


def rms(phase: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Piston-removed RMS phase over the mask [rad]."""
    return float(np.sqrt(variance(phase, mask)))


def strehl_from_field(phase: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Numerical on-axis Strehl ratio from the complex pupil field.

    ``S = |<exp(i phi)>|^2`` averaged over the illuminated pupil, with uniform
    amplitude (no scintillation).  This is the definition of the on-axis
    intensity of the aberrated PSF relative to the diffraction limit for an
    unapodised pupil (Born & Wolf, *Principles of Optics*, 7th ed., Sec. 9.1).

    Units: dimensionless, in ``(0, 1]``.  Assumptions: uniform illumination,
    monochromatic, far field, aberration is pure phase.
    """
    values = _masked(phase, mask)
    return float(np.abs(np.mean(np.exp(1j * values))) ** 2)
