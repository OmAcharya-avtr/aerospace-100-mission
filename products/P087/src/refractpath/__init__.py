"""refractpath: slant-path atmospheric refraction with the cross-model disagreement measured.

What this package computes, and in what order
---------------------------------------------
The top-level call is :func:`refraction`. It returns a :class:`RefractionResult`
whose **first** field is a :class:`refractpath.verdict.ValidityVerdict`: the
computed answer to whether the correction is worth using at the elevation,
atmospheric state and pointing tolerance you supplied. The verdict is computed
before the correction it gates, and it is a returned object, not documentation.

The correction itself is reported as an ensemble, never as a single number:
four independently published closed forms (:mod:`refractpath.models`), a ray
trace through a layered refractivity profile (:mod:`refractpath.raytrace`), and
the spread between them (:mod:`refractpath.disagreement`). The spread is the
output. A single number from any one of these methods, quoted without the
spread, is a number whose error you cannot bound.

What this package is not
------------------------
It is not an astrometry package and it should not be used as one. If you need a
refracted horizontal coordinate as part of a coordinate transformation, use
``astropy``, whose ``AltAz`` frame applies refraction, and which is the right
tool for astrometric work.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use. No correction computed here is suitable for
operational pointing below the measured validity floor reported in the verdict.

Equation numbering
------------------
Equations (1) to (4) are in :mod:`refractpath.atmosphere`, (5) to (12) in
:mod:`refractpath.refractivity`, (13) to (17) in :mod:`refractpath.models`,
(18) to (21) in :mod:`refractpath.raytrace`, (22) in
:mod:`refractpath.disagreement` and (23) in :mod:`refractpath.verdict`. Each
carries its source, units, assumptions and validity range in the module
docstring where it is defined.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .atmosphere import (
    EARTH_RADIUS_M,
    PRESSURE_DOMAIN_HPA,
    TEMPERATURE_DOMAIN_C,
    US1976_LAYERS,
    WAVELENGTH_DOMAIN_UM,
    AtmosphericState,
    LayeredProfile,
    Observer,
    build_profile,
    saturation_vapour_pressure_pa,
)
from .disagreement import (
    EnsembleMember,
    SpreadPoint,
    ValidityFloor,
    default_ensemble,
    default_floor_grid,
    ensemble_at,
    pairwise_disagreement,
    spread_curve,
    validity_floor,
)
from .models import (
    CLOSED_FORM_BY_NAME,
    CLOSED_FORM_MODELS,
    ClosedFormModel,
    bennett_1982,
    observer_refractivity,
    oriani_breakdown_elevation_deg,
    oriani_series,
    oriani_series_unguarded,
    oriani_truncation_ratio,
    plane_parallel,
    plane_parallel_floor_deg,
    saemundsson_1986,
    saemundsson_1986_from_true,
)
from .raytrace import RayTraceResult, raytrace_refraction, shell_heights, trace
from .reference_checks import (
    BENNETT_STATE,
    REFERENCE_CHECKS,
    STANDARD_STATE,
    bennett_departure_arcsec,
    constant_of_refraction_arcsec,
    run_reference_checks,
)
from .references import (
    PUBLISHED_BY_KEY,
    PUBLISHED_REFERENCES,
    PublishedReference,
    ReferenceComparison,
    compare,
)
from .refractivity import (
    REFRACTIVITY_MODELS,
    REFRACTIVITY_VALIDITY_UM,
    birch_downs_1994,
    ciddor_1996,
    ciddor_vapour_pressure_pa,
    edlen_1966,
    iau_explanatory_supplement,
    refractivity_spread,
)
from .report import format_spread_point, format_table, format_validity_floor, format_verdict
from .verdict import (
    PROFILE_ENVELOPE_LAPSE_RATES,
    UNDETERMINABLE,
    ValidityVerdict,
    compute_verdict,
    profile_envelope_arcsec,
)

__version__ = "0.1.0"

#: Default pointing tolerance, arcseconds. Chosen as a round figure in the
#: region where a mount's own pointing model matters; it is a *declared* input
#: and every number that depends on it says so.
DEFAULT_TOLERANCE_ARCSEC = 10.0


@dataclass(frozen=True)
class RefractionResult:
    """The result of a refraction query. The verdict comes first, deliberately.

    Attributes:
        verdict: The computed validity verdict. Read this before the numbers
            below it.
        apparent_elevation_deg: The elevation asked for, degrees.
        true_elevation_deg: ``apparent_elevation_deg`` minus the reference
            refraction, degrees.
        reference_refraction_arcsec: The ray trace of equation (19), which is
            the reference the closed forms are scored against. This is the
            single number to use **if** the verdict permits it.
        reference_method: Name of the reference method.
        methods_arcsec: Every ensemble member's refraction, arcseconds.
        undefined_methods: Members undefined at this elevation.
        disagreement_arcsec: Equation (22).
        widest_pair: The two members holding the extremes.
        profile_envelope_arcsec: Equation (23).
        profile_envelope_members: Per-lapse-rate ray traces behind it.
        refractivity_at_observer: ``n - 1`` from each formulation at the
            observer, plus the spread.
        floor: The measured validity floor and its evidence.
        state: The atmospheric state used.
        observer: The observer used.
    """

    verdict: ValidityVerdict
    apparent_elevation_deg: float
    true_elevation_deg: float
    reference_refraction_arcsec: float
    reference_method: str
    methods_arcsec: dict[str, float]
    undefined_methods: tuple[str, ...]
    disagreement_arcsec: float
    widest_pair: tuple[str, str]
    profile_envelope_arcsec: float
    profile_envelope_members: dict[str, float]
    refractivity_at_observer: dict[str, float]
    floor: ValidityFloor = field(repr=False)
    state: AtmosphericState = field(repr=False)
    observer: Observer = field(repr=False)

    @property
    def usable(self) -> bool:
        """Shorthand for ``self.verdict.usable``."""
        return self.verdict.usable


def refraction(
    apparent_elevation_deg: float,
    state: AtmosphericState | None = None,
    observer: Observer | None = None,
    tolerance_arcsec: float = DEFAULT_TOLERANCE_ARCSEC,
    members: Sequence[EnsembleMember] | None = None,
    refractivity_model: str = "ciddor_1996",
    floor_grid_deg: Sequence[float] | None = None,
) -> RefractionResult:
    """Refraction at one apparent elevation, with the verdict computed first.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees, in
            [-5, 90].
        state: Atmospheric state at the observer. Defaults to 1013.25 hPa,
            10 C, dry, 550 nm.
        observer: Observer. Defaults to sea level on the equator.
        tolerance_arcsec: Declared pointing tolerance, arcseconds. This is an
            input, not a property of the atmosphere, and the validity floor is
            a function of it.
        members: Disagreement ensemble. Defaults to
            :func:`refractpath.disagreement.default_ensemble`.
        refractivity_model: Which refractivity formulation the methods that
            take one should use. Equations (15) and (16) take none.
        floor_grid_deg: Elevation grid for the floor scan. Defaults to
            :func:`refractpath.disagreement.default_floor_grid`.

    Returns:
        The result, verdict first.

    Raises:
        TypeError: On a non-numeric elevation.
        ValueError: On an out-of-range elevation or a non-positive tolerance.
        KeyError: On an unknown refractivity formulation.
    """
    here_state = AtmosphericState() if state is None else state
    here_observer = Observer() if observer is None else observer
    ensemble = default_ensemble() if members is None else tuple(members)

    point = ensemble_at(
        apparent_elevation_deg, here_state, here_observer, ensemble, refractivity_model
    )
    floor = validity_floor(
        here_state,
        tolerance_arcsec,
        here_observer,
        ensemble,
        refractivity_model,
        grid_deg=floor_grid_deg,
    )
    envelope, envelope_members = profile_envelope_arcsec(
        apparent_elevation_deg, here_state, here_observer, refractivity_model
    )
    computed_verdict = compute_verdict(
        apparent_elevation_deg,
        here_state,
        tolerance_arcsec,
        here_observer,
        ensemble,
        refractivity_model,
        spread_point=point,
        floor=floor,
    )

    reference = raytrace_refraction(
        apparent_elevation_deg, here_state, here_observer, refractivity_model
    )
    true_elevation = (
        apparent_elevation_deg - reference / 3600.0 if math.isfinite(reference) else math.nan
    )
    return RefractionResult(
        verdict=computed_verdict,
        apparent_elevation_deg=float(apparent_elevation_deg),
        true_elevation_deg=true_elevation,
        reference_refraction_arcsec=reference,
        reference_method="raytrace",
        methods_arcsec=dict(point.values_arcsec),
        undefined_methods=point.undefined_members,
        disagreement_arcsec=point.spread_arcsec,
        widest_pair=(point.minimum_name, point.maximum_name),
        profile_envelope_arcsec=envelope,
        profile_envelope_members=envelope_members,
        refractivity_at_observer=refractivity_spread(
            here_state.wavelength_um,
            here_state.pressure_pa,
            here_state.temperature_c,
            here_state.vapour_pressure_pa,
            here_state.co2_ppm,
        ),
        floor=floor,
        state=here_state,
        observer=here_observer,
    )


__all__ = [
    "BENNETT_STATE",
    "CLOSED_FORM_BY_NAME",
    "CLOSED_FORM_MODELS",
    "DEFAULT_TOLERANCE_ARCSEC",
    "EARTH_RADIUS_M",
    "PRESSURE_DOMAIN_HPA",
    "PROFILE_ENVELOPE_LAPSE_RATES",
    "PUBLISHED_BY_KEY",
    "PUBLISHED_REFERENCES",
    "REFERENCE_CHECKS",
    "REFRACTIVITY_MODELS",
    "REFRACTIVITY_VALIDITY_UM",
    "STANDARD_STATE",
    "TEMPERATURE_DOMAIN_C",
    "UNDETERMINABLE",
    "US1976_LAYERS",
    "WAVELENGTH_DOMAIN_UM",
    "AtmosphericState",
    "ClosedFormModel",
    "EnsembleMember",
    "LayeredProfile",
    "Observer",
    "PublishedReference",
    "RayTraceResult",
    "ReferenceComparison",
    "RefractionResult",
    "SpreadPoint",
    "ValidityFloor",
    "ValidityVerdict",
    "__version__",
    "bennett_1982",
    "bennett_departure_arcsec",
    "birch_downs_1994",
    "build_profile",
    "ciddor_1996",
    "ciddor_vapour_pressure_pa",
    "compare",
    "compute_verdict",
    "constant_of_refraction_arcsec",
    "default_ensemble",
    "default_floor_grid",
    "edlen_1966",
    "ensemble_at",
    "format_spread_point",
    "format_table",
    "format_validity_floor",
    "format_verdict",
    "iau_explanatory_supplement",
    "observer_refractivity",
    "oriani_breakdown_elevation_deg",
    "oriani_series",
    "oriani_series_unguarded",
    "oriani_truncation_ratio",
    "pairwise_disagreement",
    "plane_parallel",
    "plane_parallel_floor_deg",
    "profile_envelope_arcsec",
    "raytrace_refraction",
    "refraction",
    "refractivity_spread",
    "run_reference_checks",
    "saemundsson_1986",
    "saemundsson_1986_from_true",
    "saturation_vapour_pressure_pa",
    "shell_heights",
    "spread_curve",
    "trace",
    "validity_floor",
]
