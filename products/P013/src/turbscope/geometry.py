r"""Propagation path, path-weighting kernels and weighted path averages.

Conventions used everywhere in :mod:`turbscope`
-----------------------------------------------
* ``z``  distance from the **transmitter / beacon** along the path, metres,
  ``0 <= z <= L``.  The receiver (the instrument) sits at ``z = L``.
* ``L``  total path length, metres.  Horizontal, homogeneous-in-the-mean path.
* ``lambda`` optical wavelength, metres; ``k = 2*pi/lambda`` in rad/m.
* ``Cn2`` refractive-index structure parameter, m^(-2/3).
* ``u = z / L`` the dimensionless path coordinate.

Why weighting kernels matter
----------------------------
No optical turbulence sensor measures ``Cn2`` at a point, and no two sensors
measure the *same* average.  Each estimator responds to a different weighted
integral of ``Cn2(z)`` along the path:

* scintillation (spherical wave):   ``W(u) = u^(5/6) (1 - u)^(5/6)``  -- symmetric,
  peaked at mid-path, blind to both endpoints;
* scintillation (plane wave):       ``W(u) = (1 - u)^(5/6)``          -- weighted
  toward the transmitter;
* wave coherence (spherical wave):  ``W(u) = u^(5/3)``                -- weighted
  toward the receiver, which is what a DIMM on a ground beacon sees;
* wave coherence (plane wave):      ``W(u) = 1``                      -- uniform.

Sources
-------
Andrews, L. C. & Phillips, R. L. (2005), *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press: ch. 8 (weak-fluctuation scintillation, Eqs. 8.9-8.13),
ch. 6 (spherical- and plane-wave coherence).  Fried, D. L. (1966), *JOSA* 56(10),
1372-1379 (coherence length).  Tatarskii, V. I. (1971), *The Effects of the
Turbulent Atmosphere on Wave Propagation*, Israel Program for Scientific
Translations (Rytov theory).

Validity of everything in this module: Kolmogorov spectrum with no inner/outer
scale, ``l0 << sqrt(L/k) << L0``; horizontal path with statistically homogeneous
transverse structure; unpolarised, monochromatic, narrow-band light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from ._validate import check_geometry, check_path_samples, check_positive, check_wavelength

__all__ = [
    "PathGeometry",
    "coherence_weight",
    "scintillation_weight",
    "wavenumber",
    "weight_normalisation",
    "weighted_path_average",
]


def wavenumber(wavelength_m: float) -> float:
    """Optical wavenumber ``k = 2*pi/lambda`` in rad/m."""
    return 2.0 * np.pi / check_wavelength(wavelength_m)


def scintillation_weight(u: np.ndarray, geometry: str = "spherical") -> np.ndarray:
    """Rytov (weak-fluctuation) scintillation path weight, dimensionless.

    ``W_sc(u) = u^(5/6) (1-u)^(5/6)`` for a spherical wave and ``(1-u)^(5/6)`` for
    a plane wave, with ``u = z/L`` measured from the transmitter.

    Source: Andrews & Phillips (2005) Eqs. (8.10)-(8.13); the kernels follow from
    the Rytov double integral evaluated in :mod:`turbscope.scintillation`.
    Validity: weak fluctuations only (the *kernel* keeps its meaning at all
    strengths but the linear relation to ``sigma_I^2`` does not).
    """
    geom = check_geometry(geometry)
    x = np.asarray(u, dtype=float)
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("path coordinate u must lie in [0, 1]")
    x = np.clip(x, 0.0, 1.0)
    if geom == "spherical":
        return x ** (5.0 / 6.0) * (1.0 - x) ** (5.0 / 6.0)
    return (1.0 - x) ** (5.0 / 6.0)


def coherence_weight(u: np.ndarray, geometry: str = "spherical") -> np.ndarray:
    """Wave-structure-function (coherence-length) path weight, dimensionless.

    ``W_co(u) = u^(5/3)`` for a spherical wave from a beacon at ``u = 0`` observed
    at ``u = 1``; ``W_co(u) = 1`` for a plane wave.

    Source: Andrews & Phillips (2005) ch. 6; Fried (1966).  The ``u^(5/3)`` factor
    is the 5/3 power of the geometric ray-separation ratio ``u`` for a diverging
    spherical wave, so turbulence at the beacon contributes nothing and turbulence
    at the receiver contributes fully.
    """
    geom = check_geometry(geometry)
    x = np.asarray(u, dtype=float)
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("path coordinate u must lie in [0, 1]")
    x = np.clip(x, 0.0, 1.0)
    if geom == "spherical":
        return x ** (5.0 / 3.0)
    return np.ones_like(x)


def weight_normalisation(kind: str, geometry: str = "spherical") -> float:
    """Return ``int_0^1 W(u) du`` in closed form for the four supported kernels.

    * scintillation / spherical: ``B(11/6, 11/6) = 0.220535655...``
    * scintillation / plane:     ``6/11 = 0.545454...``
    * coherence / spherical:     ``3/8 = 0.375``
    * coherence / plane:         ``1``

    These are exact (Euler beta function / elementary integrals) and are used as
    known answers in the test suite.
    """
    geom = check_geometry(geometry)
    if kind not in ("scintillation", "coherence"):
        raise ValueError(f"kind must be 'scintillation' or 'coherence', got {kind!r}")
    if kind == "scintillation":
        from scipy import special

        return float(special.beta(11.0 / 6.0, 11.0 / 6.0)) if geom == "spherical" else 6.0 / 11.0
    return 3.0 / 8.0 if geom == "spherical" else 1.0


@dataclass(frozen=True)
class PathGeometry:
    """A horizontal optical propagation path.

    Parameters
    ----------
    length_m
        Path length ``L`` in metres, transmitter at ``z = 0`` and receiver at
        ``z = L``.
    wavelength_m
        Optical wavelength in metres, 300 nm - 3 um.
    geometry
        ``"spherical"`` (point beacon, the usual terrestrial scintillometer /
        beacon-DIMM case) or ``"plane"`` (collimated / distant source).

    Notes
    -----
    The Fresnel scale ``sqrt(L/k)`` is reported by :meth:`fresnel_scale_m`; the
    Kolmogorov relations used here assume ``l0 << sqrt(L/k) << L0``, i.e. an inner
    scale of a few millimetres and an outer scale of tens of metres bracket it.
    """

    length_m: float
    wavelength_m: float
    geometry: str = "spherical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_m", check_positive("length_m", self.length_m))
        object.__setattr__(self, "wavelength_m", check_wavelength(self.wavelength_m))
        object.__setattr__(self, "geometry", check_geometry(self.geometry))

    @property
    def k(self) -> float:
        """Wavenumber ``2*pi/lambda``, rad/m."""
        return 2.0 * np.pi / self.wavelength_m

    def fresnel_scale_m(self) -> float:
        """Fresnel scale ``sqrt(L/k)`` in metres."""
        return float(np.sqrt(self.length_m / self.k))

    def uniform_grid(self, n: int = 401) -> np.ndarray:
        """A uniform Simpson-friendly path grid of ``n`` points (odd ``n`` preferred)."""
        if n < 3:
            raise ValueError(f"n must be >= 3, got {n}")
        return np.linspace(0.0, self.length_m, int(n))


def weighted_path_average(
    z_m: np.ndarray,
    cn2: np.ndarray,
    *,
    kind: str = "scintillation",
    geometry: str = "spherical",
    length_m: float | None = None,
) -> float:
    """Weighted path average of ``Cn2``, m^(-2/3).

    ``<Cn2>_W = int_0^L W(z/L) Cn2(z) dz / int_0^L W(z/L) dz``

    evaluated with composite Simpson's rule on the supplied grid.  This is the
    quantity a single-sensor closed-form inversion actually estimates: the
    scintillation kernel for a scintillometer, the coherence kernel for a DIMM.

    Parameters
    ----------
    z_m, cn2
        Path samples, metres and m^(-2/3).
    kind
        ``"scintillation"`` or ``"coherence"``.
    geometry
        ``"spherical"`` or ``"plane"``.
    length_m
        Path length; defaults to ``z_m[-1]``.  Supplying it explicitly lets you
        integrate over a sub-span of a longer path.

    Notes
    -----
    Quadrature accuracy: the scintillation kernel has an unbounded derivative at
    both endpoints, so Simpson's rule converges at roughly ``O(N^(-11/6))`` rather
    than ``O(N^-4)``.  Measured relative error of the normalisation integral is
    9.2e-06 at 201 points and 2.6e-06 at 401 points
    (``validation/validate_forward_models.py``).
    """
    z, c = check_path_samples(z_m, cn2)
    total = float(z[-1]) if length_m is None else check_positive("length_m", length_m)
    if total <= 0.0:
        raise ValueError("path length must be positive")
    u = z / total
    if kind == "scintillation":
        w = scintillation_weight(u, geometry)
    elif kind == "coherence":
        w = coherence_weight(u, geometry)
    else:
        raise ValueError(f"kind must be 'scintillation' or 'coherence', got {kind!r}")
    num = float(integrate.simpson(w * c, x=z))
    den = float(integrate.simpson(w, x=z))
    if den <= 0.0:
        raise ValueError("weight normalisation integrated to zero; check the path grid")
    return num / den
