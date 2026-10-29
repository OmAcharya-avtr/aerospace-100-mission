"""The checks against :mod:`refractpath.references`, as library code.

They live in the library rather than in a validation script for one reason: a
check against external published material is a capability of the product, not a
build-time ritual. ``python -m refractpath references`` runs them, the test
suite runs them, and ``validation/validate_published_references.py`` runs them
and writes the transcript. All three run the same code, so there is no way for
the transcript and the shipped behaviour to disagree.

Every function here returns a :class:`refractpath.references.ReferenceComparison`.
None of them raises on a miss. A miss is the result.
"""

from __future__ import annotations

import math

import numpy as np

from .atmosphere import AtmosphericState, Observer, saturation_vapour_pressure_pa
from .disagreement import default_floor_grid
from .models import bennett_1982, observer_refractivity
from .raytrace import raytrace_refraction
from .references import (
    BENNETT_ACCURACY_CLAIM,
    CONSTANT_OF_REFRACTION,
    DENSITY_STANDARD_AIR,
    HORIZON_REFRACTION_CONVENTION,
    REFRACTIVITY_633NM,
    SATURATION_VAPOUR_PRESSURE_20C,
    ReferenceComparison,
    compare,
)
from .refractivity import (
    R_CIDDOR,
    birch_downs_1994,
    ciddor_compressibility,
    ciddor_molar_mass_air,
    ciddor_saturation_pressure_pa,
    ciddor_vapour_pressure_pa,
)

#: Conditions of the constant-of-refraction check and the horizon check. The
#: wavelength is this product's choice and the checks report the sensitivity to
#: it rather than hiding it.
STANDARD_STATE = AtmosphericState(
    pressure_hpa=1013.25, temperature_c=10.0, relative_humidity=0.0, wavelength_um=0.55
)

#: Conditions equations (15) and (16) were published for.
BENNETT_STATE = AtmosphericState(
    pressure_hpa=1010.0, temperature_c=10.0, relative_humidity=0.0, wavelength_um=0.55
)

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi


def check_refractivity_633nm() -> ReferenceComparison:
    """Equation (6) at the published standard-air conditions against 2.7653e-4."""
    measured = float(birch_downs_1994(0.633, 101325.0, 15.0, 0.0))
    sigma_sq = 1.0 / 0.633**2
    dispersion_only = (
        8342.54 + 2406147.0 / (130.0 - sigma_sq) + 15998.0 / (38.9 - sigma_sq)
    ) * 1e-8
    detail = (
        f"equation (6) with the equation (7) density factor at 15 C and 101325 Pa gives "
        f"{measured:.9e}; the published five-figure value is "
        f"{REFRACTIVITY_633NM.value:.5e}, difference "
        f"{measured - REFRACTIVITY_633NM.value:+.3e}. The bare dispersion term of "
        f"equation (6) is {dispersion_only:.9e}, so the equation (7) density factor at "
        f"the standard state is {measured / dispersion_only:.12f}, which is unity to "
        f"{abs(measured / dispersion_only - 1.0):.1e} as it must be."
    )
    return compare(REFRACTIVITY_633NM, measured, detail)


def check_density_standard_air() -> ReferenceComparison:
    """Equation (11) at 15 C and 101325 Pa, dry, against the 1.2250 kg/m^3 standard."""
    molar_mass = float(ciddor_molar_mass_air(450.0))
    compressibility = float(ciddor_compressibility(101325.0, 288.15, 0.0))
    measured = 101325.0 * molar_mass / (compressibility * R_CIDDOR * 288.15)
    ideal = 101325.0 * molar_mass / (R_CIDDOR * 288.15)
    detail = (
        f"equation (11) gives Z = {compressibility:.9f} and a real-gas density of "
        f"{measured:.6f} kg/m^3; the ideal-gas density with the same molar mass is "
        f"{ideal:.6f} kg/m^3, so the compressibility correction is "
        f"{(measured - ideal) / ideal:+.3e} relative and accounts for the whole "
        f"departure from the published 1.2250. Molar mass used: "
        f"{molar_mass * 1000.0:.6f} g/mol at 450 ppm CO2."
    )
    return compare(DENSITY_STANDARD_AIR, measured, detail)


def check_saturation_vapour_pressure() -> ReferenceComparison:
    """Equations (4) and (12) at 20 C against the tabulated 2339 Pa.

    The published 2339 Pa is the saturation vapour pressure of **pure water**,
    so the comparison uses the bare saturation form of each equation. The
    Ciddor enhancement factor, which accounts for the presence of air and is
    what :func:`refractpath.refractivity.ciddor_vapour_pressure_pa` applies,
    is a different quantity and is reported separately rather than compared
    against this reference.
    """
    buck = saturation_vapour_pressure_pa(20.0)
    iapws = float(ciddor_saturation_pressure_pa(293.15))
    with_enhancement = float(ciddor_vapour_pressure_pa(20.0, 1.0, 101325.0))
    detail = (
        f"equation (4), Buck (1981), gives {buck:.4f} Pa; the bare IAPWS saturation "
        f"form of equation (12) gives {iapws:.4f} Pa. They differ by "
        f"{iapws - buck:+.4f} Pa, which is {(iapws - buck) / buck:+.3e} relative, and "
        f"both are inside the tolerance of this check. Applying the Ciddor "
        f"enhancement factor at 101325 Pa raises the effective partial pressure in "
        f"air to {with_enhancement:.4f} Pa, which is "
        f"{with_enhancement - iapws:+.4f} Pa above the pure-water value and is NOT "
        f"compared against this reference, because the published figure is for pure "
        f"water. The measured value reported is the Buck form."
    )
    return compare(SATURATION_VAPOUR_PRESSURE_20C, buck, detail)


def constant_of_refraction_arcsec(
    state: AtmosphericState, observer: Observer | None = None
) -> float:
    """The coefficient of ``tan z`` in equation (14), arcseconds.

    Args:
        state: Atmospheric state at the observer.
        observer: Observer, for the Earth radius.

    Returns:
        ``kappa (1 - beta)`` in arcseconds.
    """
    from .atmosphere import EARTH_RADIUS_M, G0, R_DRY

    kappa = observer_refractivity(state, "ciddor_1996")
    radius = EARTH_RADIUS_M if observer is None else observer.radius_m
    beta = (R_DRY * state.temperature_k / G0) / radius
    return kappa * (1.0 - beta) * ARCSEC_PER_RADIAN


def check_constant_of_refraction() -> ReferenceComparison:
    """Equation (14)'s ``tan z`` coefficient against the published 58.2 to 58.3 arcsec."""
    measured = constant_of_refraction_arcsec(STANDARD_STATE)
    at_633 = constant_of_refraction_arcsec(
        AtmosphericState(pressure_hpa=1013.25, temperature_c=10.0, wavelength_um=0.633)
    )
    at_450 = constant_of_refraction_arcsec(
        AtmosphericState(pressure_hpa=1013.25, temperature_c=10.0, wavelength_um=0.450)
    )
    at_1010 = constant_of_refraction_arcsec(
        AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)
    )
    detail = (
        f"at 10 C, 1013.25 hPa, dry and 550 nm the measured coefficient is "
        f"{measured:.5f} arcsec. The wavelength is this product's choice and it "
        f"matters: {at_450:.5f} arcsec at 450 nm and {at_633:.5f} arcsec at 633 nm, a "
        f"span of {at_450 - at_633:.5f} arcsec across the visible, which is "
        f"{(at_450 - at_633) / CONSTANT_OF_REFRACTION.tolerance:.2f} times the "
        f"tolerance. At 1010 hPa instead of 1013.25 it is {at_1010:.5f} arcsec."
    )
    return compare(CONSTANT_OF_REFRACTION, measured, detail)


def check_horizon_refraction() -> ReferenceComparison:
    """The ray trace at zero apparent elevation against the adopted 34 arcmin."""
    measured = raytrace_refraction(0.0, STANDARD_STATE, None, "ciddor_1996")
    from .verdict import profile_envelope_arcsec

    envelope, members = profile_envelope_arcsec(0.0, STANDARD_STATE, None, "ciddor_1996")
    detail = (
        f"equation (19) at zero apparent elevation, 10 C, 1013.25 hPa, dry, 550 nm and "
        f"the US Standard Atmosphere 1976 surface lapse rate gives {measured:.3f} "
        f"arcsec = {measured / 60.0:.4f} arcmin against the adopted "
        f"{HORIZON_REFRACTION_CONVENTION.value / 60.0:.0f} arcmin. The profile "
        f"envelope of equation (23) at this elevation is "
        f"{envelope:.1f} arcsec = {envelope / 60.0:.3f} arcmin, which is "
        f"{envelope / HORIZON_REFRACTION_CONVENTION.tolerance:.2f} times the tolerance of "
        f"this check: "
        + ", ".join(f"{name} {value / 60.0:.4f} arcmin" for name, value in members.items())
        + ". Agreement here therefore says very little."
    )
    return compare(HORIZON_REFRACTION_CONVENTION, measured, detail)


def bennett_departure_arcsec(
    state: AtmosphericState | None = None,
    grid_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute departure of equation (15) from the ray trace, arcseconds.

    Args:
        state: Atmospheric state. Defaults to the conditions equation (15) was
            published for.
        grid_deg: Elevation grid. Defaults to
            :func:`refractpath.disagreement.default_floor_grid`.

    Returns:
        The grid and the absolute departure on it.
    """
    here = BENNETT_STATE if state is None else state
    grid = default_floor_grid() if grid_deg is None else np.asarray(grid_deg, dtype=float)
    departure = np.array(
        [
            abs(
                bennett_1982(float(elevation), here)
                - raytrace_refraction(float(elevation), here, None, "ciddor_1996")
            )
            for elevation in grid
        ]
    )
    return grid, departure


def check_bennett_accuracy_claim() -> ReferenceComparison:
    """Bennett's published 0.07 arcmin maximum error, measured against the ray trace.

    Reported whether it passes or not. See
    :data:`refractpath.references.BENNETT_ACCURACY_CLAIM` for what a miss does
    and does not show.
    """
    grid, departure = bennett_departure_arcsec()
    measured = float(np.max(departure))
    worst = float(grid[int(np.argmax(departure))])
    inside = departure <= BENNETT_ACCURACY_CLAIM.value
    if inside.all():
        holds_above = float(grid[0])
    else:
        last_violation = int(np.flatnonzero(~inside)[-1])
        holds_above = (
            float(grid[last_violation + 1])
            if last_violation + 1 < grid.size
            else float("nan")
        )
    at_633 = float(
        np.max(
            bennett_departure_arcsec(
                AtmosphericState(
                    pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.633
                )
            )[1]
        )
    )
    detail = (
        f"over {grid.size} elevations from {grid[0]:.1f} to {grid[-1]:.1f} deg at 1010 hPa, "
        f"10 C, dry and 550 nm, the maximum absolute departure of equation (15) from "
        f"equation (19) is {measured:.3f} arcsec = {measured / 60.0:.4f} arcmin, at "
        f"{worst:.4f} deg. The published claim is "
        f"{BENNETT_ACCURACY_CLAIM.value / 60.0:.2f} arcmin = "
        f"{BENNETT_ACCURACY_CLAIM.value:.2f} arcsec. The claim holds at and above "
        f"{holds_above:.4f} deg and fails below it. At 633 nm instead of 550 nm the "
        f"maximum departure is {at_633:.3f} arcsec, so the wavelength this product "
        f"chooses for a formula that carries none moves the measured departure by "
        f"{at_633 - measured:+.3f} arcsec."
    )
    return compare(BENNETT_ACCURACY_CLAIM, measured, detail)


#: Every reference check, in report order.
REFERENCE_CHECKS = (
    check_refractivity_633nm,
    check_density_standard_air,
    check_saturation_vapour_pressure,
    check_constant_of_refraction,
    check_horizon_refraction,
    check_bennett_accuracy_claim,
)


def run_reference_checks() -> tuple[ReferenceComparison, ...]:
    """Run every check against :mod:`refractpath.references`.

    Returns:
        One comparison per published reference, in report order. Nothing is
        raised on a miss.
    """
    return tuple(check() for check in REFERENCE_CHECKS)
