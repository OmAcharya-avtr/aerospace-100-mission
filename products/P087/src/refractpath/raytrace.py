"""Baseline 2: a ray trace through a layered spherically symmetric refractivity profile.

This is the reference the closed forms of :mod:`refractpath.models` are scored
against. It is not ground truth. Ground truth for atmospheric refraction is a
measurement with a theodolite under a known profile, and this product has
neither; what it has is a numerical integration of the refraction integral
through a declared profile, whose numerical error is measured and whose
**profile** error is acknowledged as the dominant term and reported as a
first-class output.

Equation (18), the invariant of a ray in a spherically symmetric medium
(Bouguer's formula, the spherical analogue of Snell's law):

    n(r) r sin(theta(r)) = n_0 r_0 sin(z_app) = C

Source: Bouguer's formula, as given in Smart (1977) chapter 3 and in
Auer, L. H. and Standish, E. M. (2000), "Astronomical refraction: computation
for all zenith angles", Astronomical Journal 119, 2472-2474, equation (2).
Units: r in metres, angles in radians. Assumptions: a spherically symmetric,
horizontally homogeneous atmosphere; no horizontal refractivity gradient. A
real horizontal gradient at low elevation is not small and this package cannot
see it; that is one of the structural limits reported by
:class:`refractpath.verdict.ValidityVerdict`.

Equation (19), the refraction integral:

    R = - integral from n_0 to 1 of tan(theta) d(ln n)

Source: Auer and Standish (2000), equation (4), and the classical derivation
in Smart (1977) chapter 3. Units: radians.

Equation (20), the discretisation used here. The atmosphere is divided into
``n_shells`` shells of constant refractive index. Across shell ``i``, bounded
by refractive indices ``n_i`` above ``n_{i+1}``, the contribution is
``tan(theta_i) (ln n_i - ln n_{i+1})`` with ``theta_i`` evaluated at the shell
midpoint. Summing telescopes exactly onto equation (19):

    R = sum over i of tan(theta_i) (ln n_i - ln n_{i+1})

Shell boundaries are placed at ``z_j = z_top (j / N)^q`` with ``q = 3`` by
default, so that the shells are dense where the refractivity gradient is
largest. **The midpoint evaluation is not a convenience.** For a ray launched
exactly horizontally the observer is the tangent point, ``sin(theta) = 1``
there and ``tan(theta)`` is infinite; the integral still converges because the
singularity is integrable like ``z^{-1/2}``, but any rule that evaluates the
integrand at the observer returns an infinity. Evaluating at shell midpoints
never touches it. The grid convergence this buys is measured in
``validation/validate_raytrace_convergence.py``.

Equation (21), the stable form of ``cos^2(theta)``. Writing
``cos^2 = (nr - C)(nr + C) / (nr)^2`` and expanding ``nr - C`` as

    nr - C = r_0 (1 - sin z_app) + z + (n-1) r - (n_0 - 1) r_0 sin z_app

removes the catastrophic cancellation that a direct ``1 - sin^2(theta)``
suffers when ``sin(theta)`` is within 1e-14 of unity, which is exactly the
regime of a horizon ray in the first few metres above the observer. Each term
on the right is computed separately and none of them is a difference of two
nearly equal large numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import AtmosphericState, Observer, build_profile
from .refractivity import REFRACTIVITY_MODELS

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi

#: Default number of shells. Chosen from the measured grid-convergence study in
#: ``validation/validate_raytrace_convergence.py``, not guessed.
DEFAULT_SHELLS = 10_000

#: Default top of the integrated atmosphere, metres above the observer. The
#: refraction contributed above this height is measured in the same script.
DEFAULT_TOP_M = 90_000.0

#: Default grid exponent of equation (20).
DEFAULT_GRID_EXPONENT = 3.0


@dataclass(frozen=True)
class RayTraceResult:
    """One ray trace, with the diagnostics needed to judge it.

    Attributes:
        refraction_arcsec: ``h_apparent - h_true``, arcseconds.
        apparent_elevation_deg: The elevation the trace was asked for.
        shells: Number of shells used.
        top_m: Top of the integrated atmosphere, metres above the observer.
        grid_exponent: ``q`` of equation (20).
        observer_refractivity: ``n - 1`` at the observer.
        tangent_shell: Index of the first shell at which ``cos^2(theta)``
            underflowed to zero, or -1 if none did. A non-negative value means
            the ray grazed its tangent point inside the grid, which only
            happens for a ray launched at or below the horizontal.
        refractivity_model: Which formulation supplied ``n(r)``.
    """

    refraction_arcsec: float
    apparent_elevation_deg: float
    shells: int
    top_m: float
    grid_exponent: float
    observer_refractivity: float
    tangent_shell: int
    refractivity_model: str


def _refractivity_on_grid(
    name: str,
    state: AtmosphericState,
    pressure_pa: np.ndarray,
    temperature_c: np.ndarray,
    vapour_pressure_pa: np.ndarray,
) -> np.ndarray:
    if name not in REFRACTIVITY_MODELS:
        raise KeyError(
            f"unknown refractivity model {name!r}; choose from {sorted(REFRACTIVITY_MODELS)}"
        )
    function = REFRACTIVITY_MODELS[name]
    if name == "ciddor_1996":
        return np.asarray(
            function(
                state.wavelength_um,
                pressure_pa,
                temperature_c,
                vapour_pressure_pa,
                state.co2_ppm,
            )
        )
    return np.asarray(
        function(state.wavelength_um, pressure_pa, temperature_c, vapour_pressure_pa)
    )


def shell_heights(
    shells: int = DEFAULT_SHELLS,
    top_m: float = DEFAULT_TOP_M,
    grid_exponent: float = DEFAULT_GRID_EXPONENT,
) -> np.ndarray:
    """Shell boundary heights above the observer, metres, from equation (20).

    Args:
        shells: Number of shells; the returned array has ``shells + 1`` entries.
        top_m: Top of the integrated atmosphere, metres above the observer.
        grid_exponent: ``q`` of equation (20). Larger concentrates shells near
            the observer.

    Returns:
        Ascending array of ``shells + 1`` boundary heights starting at 0.

    Raises:
        ValueError: On a non-positive shell count, top or exponent.
    """
    if not isinstance(shells, int) or isinstance(shells, bool):
        raise TypeError(f"shells must be an int, got {type(shells).__name__}")
    if shells < 2:
        raise ValueError(f"shells must be at least 2, got {shells!r}")
    if not math.isfinite(top_m) or top_m <= 0.0:
        raise ValueError(f"top_m must be positive and finite, got {top_m!r}")
    if not math.isfinite(grid_exponent) or grid_exponent <= 0.0:
        raise ValueError(f"grid_exponent must be positive and finite, got {grid_exponent!r}")
    fraction = np.arange(shells + 1, dtype=float) / float(shells)
    return top_m * fraction**grid_exponent


def trace(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
    shells: int = DEFAULT_SHELLS,
    top_m: float = DEFAULT_TOP_M,
    grid_exponent: float = DEFAULT_GRID_EXPONENT,
) -> RayTraceResult:
    """Integrate equation (19) by the discretisation of equations (20) and (21).

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees, in
            [-5, 90]. Below the horizontal the ray enters the atmosphere, grazes
            a tangent point below the observer and the spherically symmetric
            invariant of equation (18) no longer describes a single-valued
            path from the observer upward; such a request returns ``nan`` with
            ``tangent_shell`` set, rather than a number.
        state: Atmospheric state at the observer.
        observer: Observer. Defaults to sea level on the equator.
        refractivity_model: Key of
            :data:`refractpath.refractivity.REFRACTIVITY_MODELS`.
        shells: Number of shells.
        top_m: Top of the integrated atmosphere, metres above the observer.
        grid_exponent: ``q`` of equation (20).

    Returns:
        The trace and its diagnostics.

    Raises:
        TypeError: On a non-numeric elevation.
        ValueError: On an out-of-range elevation or an invalid grid.
        KeyError: On an unknown refractivity model.
    """
    if not isinstance(apparent_elevation_deg, (int, float)) or isinstance(
        apparent_elevation_deg, bool
    ):
        raise TypeError(
            "apparent_elevation_deg must be a real number, got "
            f"{type(apparent_elevation_deg).__name__}"
        )
    elevation = float(apparent_elevation_deg)
    if not math.isfinite(elevation):
        raise ValueError(f"apparent_elevation_deg must be finite, got {elevation!r}")
    if not -5.0 <= elevation <= 90.0:
        raise ValueError(f"apparent_elevation_deg must lie in [-5, 90], got {elevation!r}")
    here = Observer() if observer is None else observer

    boundaries = shell_heights(shells, top_m, grid_exponent)
    profile = build_profile(state, boundaries, top_m=top_m)
    refractivity = _refractivity_on_grid(
        refractivity_model,
        state,
        profile.pressure_pa,
        profile.temperature_k - 273.15,
        profile.vapour_pressure_pa,
    )
    log_n = np.log1p(refractivity)

    radius_0 = here.radius_m
    refractivity_0 = float(refractivity[0])
    sin_z = math.sin(math.radians(90.0 - elevation))
    invariant = (1.0 + refractivity_0) * radius_0 * sin_z

    midpoint_height = 0.5 * (boundaries[:-1] + boundaries[1:])
    midpoint_refractivity = 0.5 * (refractivity[:-1] + refractivity[1:])
    midpoint_radius = radius_0 + midpoint_height
    n_r = (1.0 + midpoint_refractivity) * midpoint_radius

    # Equation (21).
    n_r_minus_c = (
        radius_0 * (1.0 - sin_z)
        + midpoint_height
        + midpoint_refractivity * midpoint_radius
        - refractivity_0 * radius_0 * sin_z
    )
    cos_sq = np.maximum(n_r_minus_c * (n_r + invariant), 0.0) / n_r**2
    sin_theta = np.clip(invariant / n_r, -1.0, 1.0)
    tangent_indices = np.flatnonzero(cos_sq <= 0.0)
    tangent_shell = int(tangent_indices[0]) if tangent_indices.size else -1

    tan_theta = np.divide(
        sin_theta, np.sqrt(cos_sq), out=np.zeros_like(sin_theta), where=cos_sq > 0.0
    )
    radians = float(np.sum(tan_theta * (log_n[:-1] - log_n[1:])))
    value = math.nan if elevation < 0.0 else radians * ARCSEC_PER_RADIAN
    return RayTraceResult(
        refraction_arcsec=value,
        apparent_elevation_deg=elevation,
        shells=int(shells),
        top_m=float(top_m),
        grid_exponent=float(grid_exponent),
        observer_refractivity=refractivity_0,
        tangent_shell=tangent_shell,
        refractivity_model=refractivity_model,
    )


def raytrace_refraction(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (19) at the package defaults, arcseconds.

    Signature-compatible with every function in
    :data:`refractpath.models.CLOSED_FORM_MODELS`, so that the ray trace can
    be a member of the disagreement ensemble on the same footing as the closed
    forms.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Observer. Defaults to sea level on the equator.
        refractivity_model: Which formulation supplies ``n(r)``.

    Returns:
        ``h_apparent - h_true`` in arcseconds.
    """
    return trace(
        apparent_elevation_deg, state, observer, refractivity_model
    ).refraction_arcsec
