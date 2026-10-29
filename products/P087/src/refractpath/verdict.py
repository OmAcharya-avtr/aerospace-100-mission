"""The validity verdict: a computed, returned output, not a limitations section.

Every call to :func:`refractpath.refraction` returns a
:class:`ValidityVerdict` as the **first** field of its result, computed before
anything is said about the correction itself. The verdict answers one question:
for the elevation, the atmospheric state and the pointing tolerance you gave
me, is the refraction correction I am about to hand you worth using?

There are three answers.

``usable``
    The ensemble of independent standard methods agrees to within the declared
    tolerance at this elevation, every member of the ensemble is inside its
    own published validity range, the atmospheric state is inside the declared
    domains, and the profile the library had to invent above the observer does
    not move the answer by more than half the tolerance.

``degraded``
    The ensemble agrees to within the tolerance but at least one of the other
    four conditions fails. The number is probably fine and the reason it might
    not be is named.

``refuse``
    Either the ensemble does not agree to within the tolerance at this
    elevation, or the unmeasured profile above the observer moves the answer by
    more than the tolerance on its own. In this regime no correction from this
    package, and none from any of the published closed forms it implements, is
    suitable for operational pointing: the disagreement between methods that
    are all defensible is larger than the error you said you could accept.

Equation (23), the profile envelope. The library is given one pressure, one
temperature and one humidity, at one point. Everything above the observer is
invented by equation (2). The envelope is the spread of the ray trace over
three declared surface lapse rates, each physically attainable and each
consistent with the observer's own measurements:

    E(h) = max over L in {-9.8, -6.5, 0.0} K/km of R_raytrace(h; L)
         - min over the same set

The dry adiabatic rate, the US Standard Atmosphere 1976 rate and an isothermal
surface layer. ``E`` is a *lower* bound on the profile uncertainty, not an
upper one: a real surface inversion is steeper than isothermal, and the three
members are a declared bracket rather than a distribution. It is reported in
arcseconds beside the tolerance, and ``validation/validate_profile_sensitivity.py``
measures it over elevation and over inversion depth.

The structural limits listed in :attr:`ValidityVerdict.undeterminable` are
things this library cannot determine from the inputs it is given, at any
elevation, by any amount of extra computation. They are not a to-do list.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .atmosphere import (
    PRESSURE_DOMAIN_HPA,
    TEMPERATURE_DOMAIN_C,
    WAVELENGTH_DOMAIN_UM,
    AtmosphericState,
    Observer,
)
from .disagreement import (
    EnsembleMember,
    SpreadPoint,
    ValidityFloor,
    default_ensemble,
    ensemble_at,
    validity_floor,
)
from .models import (
    CLOSED_FORM_BY_NAME,
    oriani_breakdown_elevation_deg,
    plane_parallel_floor_deg,
)
from .raytrace import raytrace_refraction
from .refractivity import REFRACTIVITY_VALIDITY_UM, refractivity_spread

#: The three surface lapse rates of equation (23), K/m, with their labels.
PROFILE_ENVELOPE_LAPSE_RATES: tuple[tuple[str, float], ...] = (
    ("dry_adiabatic", -0.0098),
    ("us_standard_1976", -0.0065),
    ("isothermal", 0.0),
)

#: Limits this library cannot resolve from the inputs it accepts, at any
#: elevation. Each entry names the quantity and says what it would take.
UNDETERMINABLE: tuple[str, ...] = (
    (
        "the actual temperature profile above the observer: one surface temperature "
        "cannot determine it, and a radiosonde or a numerical weather profile would "
        "be required"
    ),
    (
        "the depth and strength of any surface temperature inversion: it is the "
        "single largest term in horizon refraction and nothing in the inputs "
        "constrains it"
    ),
    (
        "the horizontal refractivity gradient: the ray trace of equation (19) assumes "
        "spherical symmetry, and a real gradient along the line of sight produces a "
        "refraction that is not a function of elevation alone"
    ),
    (
        "the turbulent component: this package computes the mean refraction only and "
        "says nothing about seeing, scintillation or short-term pointing jitter"
    ),
    (
        "whether the published reference values this package is scored against are "
        "themselves right: they are reproduced, not verified"
    ),
)


@dataclass(frozen=True)
class ValidityVerdict:
    """Computed verdict on whether a refraction correction is worth using.

    Attributes:
        verdict: ``"usable"``, ``"degraded"`` or ``"refuse"``.
        apparent_elevation_deg: The elevation asked for.
        tolerance_arcsec: The declared pointing tolerance.
        disagreement_arcsec: Equation (22) at this elevation.
        profile_envelope_arcsec: Equation (23) at this elevation.
        validity_floor_deg: Measured elevation at which equation (22) crosses
            the tolerance, from :func:`refractpath.disagreement.validity_floor`.
        above_validity_floor: Whether this elevation is above that floor.
        models_inside_published_validity: Per member, whether this elevation is
            at or above the member's own declared or published validity floor.
        models_undefined: Members that returned ``nan`` at this elevation.
        plane_parallel_analytic_floor_deg: ``90 - arcsin(1/n)``, equation
            (13)'s analytic floor.
        oriani_breakdown_elevation_deg: Equation (14a)'s breakdown elevation,
            computed in closed form from the series coefficients.
        state_inside_declared_domains: Per quantity, whether the atmospheric
            state is inside this product's declared domain.
        refractivity_inside_published_wavelength_range: Per refractivity
            formulation, whether the requested wavelength is inside the range
            its source publishes.
        refractivity_relative_spread: Fractional spread of ``n - 1`` across the
            four formulations at the observer. This propagates very nearly
            linearly into every refraction angle.
        undeterminable: :data:`UNDETERMINABLE`.
        reasons: Why the verdict is what it is, one string per reason, in the
            order they were checked. Empty only for ``"usable"``.
    """

    verdict: str
    apparent_elevation_deg: float
    tolerance_arcsec: float
    disagreement_arcsec: float
    profile_envelope_arcsec: float
    validity_floor_deg: float
    above_validity_floor: bool
    models_inside_published_validity: dict[str, bool]
    models_undefined: tuple[str, ...]
    plane_parallel_analytic_floor_deg: float
    oriani_breakdown_elevation_deg: float
    state_inside_declared_domains: dict[str, bool]
    refractivity_inside_published_wavelength_range: dict[str, bool]
    refractivity_relative_spread: float
    undeterminable: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """Whether the verdict is ``"usable"``."""
        return self.verdict == "usable"

    @property
    def refused(self) -> bool:
        """Whether the verdict is ``"refuse"``."""
        return self.verdict == "refuse"


def profile_envelope_arcsec(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> tuple[float, dict[str, float]]:
    """Equation (23) at one elevation.

    Args:
        apparent_elevation_deg: Apparent elevation, degrees.
        state: Atmospheric state at the observer. Its
            ``surface_lapse_rate_k_per_m`` is overridden, its
            ``surface_layer_top_m`` is kept.
        observer: Observer.
        refractivity_model: Which formulation supplies ``n(r)``.

    Returns:
        The envelope width in arcseconds and the per-lapse-rate refractions.
    """
    values: dict[str, float] = {}
    for label, lapse in PROFILE_ENVELOPE_LAPSE_RATES:
        variant = replace(state, surface_lapse_rate_k_per_m=lapse)
        values[label] = raytrace_refraction(
            apparent_elevation_deg, variant, observer, refractivity_model
        )
    finite = [value for value in values.values() if math.isfinite(value)]
    width = max(finite) - min(finite) if finite else math.nan
    return width, values


def compute_verdict(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    tolerance_arcsec: float,
    observer: Observer | None = None,
    members: Sequence[EnsembleMember] | None = None,
    refractivity_model: str = "ciddor_1996",
    spread_point: SpreadPoint | None = None,
    floor: ValidityFloor | None = None,
) -> ValidityVerdict:
    """Compute the verdict. Called before the correction it gates is reported.

    Args:
        apparent_elevation_deg: Apparent elevation, degrees.
        state: Atmospheric state at the observer.
        tolerance_arcsec: Declared pointing tolerance, arcseconds. Positive.
        observer: Observer.
        members: Ensemble. Defaults to
            :func:`refractpath.disagreement.default_ensemble`.
        refractivity_model: Formulation passed to the members that take one.
        spread_point: Pre-computed spread at this elevation, to avoid
            recomputing it.
        floor: Pre-computed validity floor, to avoid recomputing it.

    Returns:
        The verdict.

    Raises:
        ValueError: On a non-positive or non-finite tolerance.
    """
    if not math.isfinite(tolerance_arcsec) or tolerance_arcsec <= 0.0:
        raise ValueError(
            f"tolerance_arcsec must be positive and finite, got {tolerance_arcsec!r}"
        )
    ensemble = default_ensemble() if members is None else tuple(members)
    point = (
        ensemble_at(apparent_elevation_deg, state, observer, ensemble, refractivity_model)
        if spread_point is None
        else spread_point
    )
    measured_floor = (
        validity_floor(
            state,
            tolerance_arcsec,
            observer,
            ensemble,
            refractivity_model,
        )
        if floor is None
        else floor
    )
    envelope, _ = profile_envelope_arcsec(
        apparent_elevation_deg, state, observer, refractivity_model
    )

    inside_validity = {
        member.name: apparent_elevation_deg >= member.validity_floor_deg
        for member in ensemble
    }
    analytic_floor = plane_parallel_floor_deg(state, refractivity_model)
    oriani_floor = oriani_breakdown_elevation_deg(state, observer, refractivity_model)
    domains = state.inside_declared_domains()
    wavelength_ranges = {
        name: low <= state.wavelength_um <= high
        for name, (low, high, _) in REFRACTIVITY_VALIDITY_UM.items()
    }
    spread_of_n = refractivity_spread(
        state.wavelength_um,
        state.pressure_pa,
        state.temperature_c,
        state.vapour_pressure_pa,
        state.co2_ppm,
    )

    above_floor = bool(
        math.isnan(measured_floor.floor_deg)
        or apparent_elevation_deg > measured_floor.floor_deg
    )

    reasons: list[str] = []
    refuse = False
    if point.spread_arcsec >= tolerance_arcsec:
        refuse = True
        reasons.append(
            f"the ensemble of {len(point.values_arcsec)} independent standard methods "
            f"spans {point.spread_arcsec:.3f} arcsec at {apparent_elevation_deg:.4f} deg "
            f"apparent elevation, which is at or above the declared tolerance of "
            f"{tolerance_arcsec:.3f} arcsec; the widest pair is "
            f"{point.minimum_name} against {point.maximum_name}"
        )
    if not above_floor:
        refuse = True
        reasons.append(
            f"{apparent_elevation_deg:.4f} deg is at or below the measured validity "
            f"floor of {measured_floor.floor_deg:.4f} deg for a tolerance of "
            f"{tolerance_arcsec:.3f} arcsec"
        )
    if math.isfinite(envelope) and envelope >= tolerance_arcsec:
        refuse = True
        reasons.append(
            f"the unmeasured temperature profile above the observer moves the ray "
            f"trace by {envelope:.3f} arcsec across the three declared surface lapse "
            f"rates of equation (23), which is at or above the declared tolerance; "
            f"the dominant term in the answer is an input you did not supply"
        )

    degraded = False
    outside_validity = sorted(name for name, inside in inside_validity.items() if not inside)
    if outside_validity:
        degraded = True
        reasons.append(
            "below the declared validity floor of: "
            + ", ".join(
                f"{name} ({CLOSED_FORM_BY_NAME[name].validity_floor_deg:g} deg)"
                if name in CLOSED_FORM_BY_NAME
                else name
                for name in outside_validity
            )
        )
    if point.undefined_members:
        degraded = True
        reasons.append(
            "undefined at this elevation and excluded from equation (22): "
            + ", ".join(point.undefined_members)
        )
    outside_domains = sorted(name for name, inside in domains.items() if not inside)
    if outside_domains:
        degraded = True
        reasons.append(
            "atmospheric state outside this product's declared domain in: "
            + ", ".join(outside_domains)
            + f" (wavelength {WAVELENGTH_DOMAIN_UM[0]}-{WAVELENGTH_DOMAIN_UM[1]} um, "
            f"pressure {PRESSURE_DOMAIN_HPA[0]}-{PRESSURE_DOMAIN_HPA[1]} hPa, "
            f"temperature {TEMPERATURE_DOMAIN_C[0]}-{TEMPERATURE_DOMAIN_C[1]} C)"
        )
    outside_wavelength = sorted(
        name for name, inside in wavelength_ranges.items() if not inside
    )
    if outside_wavelength:
        degraded = True
        reasons.append(
            f"wavelength {state.wavelength_um:g} um is outside the published range of: "
            + ", ".join(outside_wavelength)
        )
    if math.isfinite(envelope) and not refuse and envelope >= 0.5 * tolerance_arcsec:
        degraded = True
        reasons.append(
            f"the profile envelope of equation (23) is {envelope:.3f} arcsec, at or "
            f"above half the declared tolerance"
        )

    if refuse:
        label = "refuse"
    elif degraded:
        label = "degraded"
    else:
        label = "usable"

    return ValidityVerdict(
        verdict=label,
        apparent_elevation_deg=float(apparent_elevation_deg),
        tolerance_arcsec=float(tolerance_arcsec),
        disagreement_arcsec=point.spread_arcsec,
        profile_envelope_arcsec=envelope,
        validity_floor_deg=measured_floor.floor_deg,
        above_validity_floor=above_floor,
        models_inside_published_validity=inside_validity,
        models_undefined=point.undefined_members,
        plane_parallel_analytic_floor_deg=analytic_floor,
        oriani_breakdown_elevation_deg=oriani_floor,
        state_inside_declared_domains=domains,
        refractivity_inside_published_wavelength_range=wavelength_ranges,
        refractivity_relative_spread=float(spread_of_n["relative_spread"]),
        undeterminable=UNDETERMINABLE,
        reasons=tuple(reasons),
    )
