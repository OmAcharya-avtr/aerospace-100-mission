"""The closed-form refraction models: baseline 1, the things that are actually used.

Every model here is a function of one apparent (observed) elevation and one
atmospheric state, and every one returns the refraction in **arcseconds**,
defined throughout this package as

    R = h_apparent - h_true

so that R is positive: refraction raises a source. A model that is undefined at
the requested elevation returns ``nan`` rather than an extrapolated number.
That is the point of the module.

Equation (13), the plane-parallel model, exact form. Snell's law at a single
plane interface between vacuum and air of refractive index ``n`` at the
observer:

    sin(z_true) = n sin(z_app)
    R = z_true - z_app

Source: Snell's law; the astronomical application and the ``R = (n-1) tan z``
first-order form are in Smart, W. M. (1977), "Textbook on Spherical
Astronomy", 6th edition, Cambridge University Press, chapter 3. Units: angles
in radians internally, arcseconds on return. Assumptions: a plane-parallel
atmosphere, that is, no Earth curvature. Validity: this model has an **exact,
analytic** validity floor. The equation has no solution when
``n sin(z_app) > 1``, so it is undefined for

    h_app < 90 degrees - arcsin(1 / n)

which is about 1.36 degrees at 1010 hPa and 10 C. This floor is computed, not
asserted, by :func:`plane_parallel_floor_deg`.

Equation (14), Oriani's theorem, the two-term series in ``tan z``:

    R = kappa (1 - beta) tan(z_app) - kappa (beta - kappa / 2) tan^3(z_app)
    kappa = n - 1 at the observer
    beta  = H / a,   H = R_dry T_0 / g0,   a = Earth radius

Source: Oriani's theorem, as given in Smart (1977), chapter 3. Units: as
above. Assumptions: a spherically symmetric atmosphere in which the
refractivity falls off exponentially with a single scale height; the series is
truncated after ``tan^3``. Validity: the series is a large-``tan z``
divergence waiting to happen. It is conventionally used to about 70 to 75
degrees zenith distance, so this product declares a floor of 15 degrees
elevation, and ``validation/validate_model_grid.py`` measures where it
actually departs from the ray trace rather than taking that on trust. Below
about 2 degrees elevation the ``tan^3`` term overtakes the ``tan`` term and the
series returns a refraction smaller than at 45 degrees, which is not a small
error but a sign reversal of the trend.

Equation (14a), the truncation ratio of equation (14), and the breakdown
elevation computed from it:

    ratio(z) = |second term| / |first term|
             = ((beta - kappa / 2) / (1 - beta)) tan^2(z)
    ratio = 1  at  h = 90 degrees - arctan(sqrt((1 - beta) / (beta - kappa / 2)))

A two-term truncation of an asymptotic series means something only while the
second term is a *correction* to the first. Where ``ratio >= 1`` the truncation
has inverted and the expression is no longer an approximation to anything, so
:func:`oriani_series` returns ``nan`` there. That elevation is **computed from
the series itself**, not declared: at 1010 hPa and 10 C it is about 1.95
degrees. :func:`oriani_series_unguarded` returns the raw series anyway, because
exhibiting what the divergence does is one of this product's outputs and the
guarded form would hide it.

The ``z_app`` in equation (14) is the apparent zenith distance. Texts differ on
whether the theorem is stated for the apparent or the true zenith distance; the
difference enters at second order in ``kappa`` and its size is measured in
``validation/validate_model_grid.py`` rather than argued about here.

Equation (15), Bennett (1982):

    R_arcmin = cot(h_app + 7.31 / (h_app + 4.4))     [h_app in degrees]

Source: Bennett, G. G. (1982), "The calculation of astronomical refraction in
marine navigation", Journal of Navigation 35(2), 255-259. Reproduced with its
accuracy claim in Meeus, J. (1998), "Astronomical Algorithms", 2nd edition,
Willmann-Bell, chapter 16. Units: degrees in, arcminutes out, converted to
arcseconds on return here. Assumptions: standard conditions of 1010 hPa and
10 C, with equation (17) applied for other conditions. Validity: 0 to 90
degrees apparent elevation, as published, with a stated maximum error of 0.07
arcminutes against Garfinkel (1967). **That claim is checked in
``validation/validate_published_references.py`` and the result of the check is
reported there whether it passes or not.**

Equation (16), Saemundsson (1986):

    R_arcmin = 1.02 cot(h_true + 10.3 / (h_true + 5.11))   [h_true in degrees]

Source: Saemundsson, Th. (1986), Sky and Telescope 72, 70, as given in Meeus
(1998) chapter 16. Units: as equation (15). Note that this formula takes the
**true** elevation, so this module inverts it numerically to express it as a
function of the apparent elevation like every other model here; the inversion
is a Brent root find on ``h_true + R(h_true) = h_app`` and its residual is
reported.

Equation (17), the pressure and temperature scaling applied to equations (15)
and (16):

    R(p, t) = R_standard (p / 1010) (283 / (273 + t))

Source: Meeus (1998), chapter 16. Units: p in hPa, t in degrees Celsius. This
is a density ratio with the reference state of 1010 hPa and 10 C built in, and
it carries no wavelength dependence at all: equations (15) and (16) are
monochromatic formulae for an unstated wavelength. That is itself a finding and
it is reported in ``validation/validate_dispersion.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.optimize import brentq

from .atmosphere import EARTH_RADIUS_M, G0, R_DRY, AtmosphericState, Observer
from .refractivity import REFRACTIVITY_MODELS

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi
ARCSEC_PER_ARCMIN = 60.0

#: Reference state built into equations (15) and (16), as published.
BENNETT_REFERENCE_PRESSURE_HPA = 1010.0
BENNETT_REFERENCE_TEMPERATURE_C = 10.0


def observer_refractivity(
    state: AtmosphericState, refractivity_model: str = "ciddor_1996"
) -> float:
    """``n - 1`` at the observer from the named formulation.

    Args:
        state: Atmospheric state at the observer.
        refractivity_model: Key of :data:`refractpath.refractivity.REFRACTIVITY_MODELS`.

    Returns:
        ``n - 1``, dimensionless.

    Raises:
        KeyError: If ``refractivity_model`` is not a known formulation.
    """
    if refractivity_model not in REFRACTIVITY_MODELS:
        raise KeyError(
            f"unknown refractivity model {refractivity_model!r}; "
            f"choose from {sorted(REFRACTIVITY_MODELS)}"
        )
    function = REFRACTIVITY_MODELS[refractivity_model]
    if refractivity_model == "ciddor_1996":
        value = function(
            state.wavelength_um,
            state.pressure_pa,
            state.temperature_c,
            state.vapour_pressure_pa,
            state.co2_ppm,
        )
    else:
        value = function(
            state.wavelength_um,
            state.pressure_pa,
            state.temperature_c,
            state.vapour_pressure_pa,
        )
    return float(value)


def plane_parallel_floor_deg(
    state: AtmosphericState, refractivity_model: str = "ciddor_1996"
) -> float:
    """Apparent elevation below which equation (13) has no solution, degrees.

    This is an analytic floor, ``90 - arcsin(1 / n)``, and it is the one
    validity limit in this package that does not have to be measured.

    Args:
        state: Atmospheric state at the observer.
        refractivity_model: Which formulation supplies ``n``.

    Returns:
        The floor in degrees of apparent elevation.
    """
    n = 1.0 + observer_refractivity(state, refractivity_model)
    return 90.0 - math.degrees(math.asin(min(1.0 / n, 1.0)))


def _check_elevation(apparent_elevation_deg: float) -> float:
    if not isinstance(apparent_elevation_deg, (int, float)) or isinstance(
        apparent_elevation_deg, bool
    ):
        raise TypeError(
            "apparent_elevation_deg must be a real number, got "
            f"{type(apparent_elevation_deg).__name__}"
        )
    value = float(apparent_elevation_deg)
    if not math.isfinite(value):
        raise ValueError(f"apparent_elevation_deg must be finite, got {value!r}")
    if not -5.0 <= value <= 90.0:
        raise ValueError(
            f"apparent_elevation_deg must lie in [-5, 90], got {value!r}; "
            "no model here is published below the horizon and none is defined above "
            "the zenith"
        )
    return value


def plane_parallel(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (13), exact. Refraction in arcseconds, or ``nan`` below the floor.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Ignored; accepted so that every model has one signature.
        refractivity_model: Which formulation supplies ``n``.

    Returns:
        ``h_apparent - h_true`` in arcseconds, or ``nan`` where equation (13)
        has no solution.
    """
    del observer
    elevation = _check_elevation(apparent_elevation_deg)
    n = 1.0 + observer_refractivity(state, refractivity_model)
    z_app = math.radians(90.0 - elevation)
    argument = n * math.sin(z_app)
    if argument >= 1.0:
        return math.nan
    return (math.asin(argument) - z_app) * ARCSEC_PER_RADIAN


def _oriani_coefficients(
    state: AtmosphericState, observer: Observer | None, refractivity_model: str
) -> tuple[float, float]:
    """``kappa`` and ``beta`` of equation (14)."""
    kappa = observer_refractivity(state, refractivity_model)
    radius = EARTH_RADIUS_M if observer is None else observer.radius_m
    beta = (R_DRY * state.temperature_k / G0) / radius
    return kappa, beta


def oriani_breakdown_elevation_deg(
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Apparent elevation at which equation (14a)'s truncation ratio reaches 1, degrees.

    Computed in closed form from the series coefficients, not declared.

    Args:
        state: Atmospheric state at the observer.
        observer: Observer, for the Earth radius.
        refractivity_model: Which formulation supplies ``kappa``.

    Returns:
        The breakdown elevation in degrees. ``nan`` if ``beta <= kappa / 2``,
        in which case the ``tan^3`` coefficient does not change sign and the
        ratio never reaches 1.
    """
    kappa, beta = _oriani_coefficients(state, observer, refractivity_model)
    denominator = beta - kappa / 2.0
    if denominator <= 0.0:
        return math.nan
    return 90.0 - math.degrees(math.atan(math.sqrt((1.0 - beta) / denominator)))


def oriani_truncation_ratio(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (14a): the ``tan^3`` term over the ``tan`` term, dimensionless.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Observer, for the Earth radius.
        refractivity_model: Which formulation supplies ``kappa``.

    Returns:
        The ratio. ``inf`` at and below the horizontal, where ``tan z`` is
        infinite.
    """
    elevation = _check_elevation(apparent_elevation_deg)
    if elevation <= 0.0:
        return math.inf
    kappa, beta = _oriani_coefficients(state, observer, refractivity_model)
    tan_z = math.tan(math.radians(90.0 - elevation))
    return abs(beta - kappa / 2.0) * tan_z**2 / abs(1.0 - beta)


def oriani_series_unguarded(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (14) as published, with no breakdown guard. Arcseconds.

    Returns the raw two-term series at any elevation above the horizontal,
    including where it has diverged. This is the function the validation
    scripts and the figures use to exhibit the divergence; it is not in the
    disagreement ensemble.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Observer, used only for the Earth radius through its height.
        refractivity_model: Which formulation supplies ``kappa``.

    Returns:
        ``h_apparent - h_true`` in arcseconds, or ``nan`` at and below the
        horizontal where ``tan z`` is infinite in exact arithmetic.
    """
    elevation = _check_elevation(apparent_elevation_deg)
    if elevation <= 0.0:
        return math.nan
    kappa, beta = _oriani_coefficients(state, observer, refractivity_model)
    tan_z = math.tan(math.radians(90.0 - elevation))
    series = kappa * (1.0 - beta) * tan_z - kappa * (beta - kappa / 2.0) * tan_z**3
    return series * ARCSEC_PER_RADIAN


def oriani_series(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (14) with the equation (14a) breakdown guard. Arcseconds.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Observer, used only for the Earth radius through its height.
        refractivity_model: Which formulation supplies ``kappa``.

    Returns:
        ``h_apparent - h_true`` in arcseconds, or ``nan`` where the truncation
        ratio of equation (14a) has reached 1 and the series has stopped
        approximating anything.
    """
    if oriani_truncation_ratio(
        apparent_elevation_deg, state, observer, refractivity_model
    ) >= 1.0:
        return math.nan
    return oriani_series_unguarded(
        apparent_elevation_deg, state, observer, refractivity_model
    )


def _pressure_temperature_factor(state: AtmosphericState) -> float:
    """Equation (17)."""
    return (state.pressure_hpa / BENNETT_REFERENCE_PRESSURE_HPA) * (
        283.0 / (273.0 + state.temperature_c)
    )


def bennett_1982(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equations (15) and (17). Refraction in arcseconds.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer; only pressure and
            temperature are used, through equation (17).
        observer: Ignored.
        refractivity_model: Ignored. Equation (15) carries no refractivity
            model and no wavelength; the argument is accepted so that every
            model has one signature, and the fact that it is ignored is a
            property of the published formula and is reported.

    Returns:
        ``h_apparent - h_true`` in arcseconds. Negative near the zenith, which
        is a property of the published formula and is reported rather than
        clipped.
    """
    del observer, refractivity_model
    elevation = _check_elevation(apparent_elevation_deg)
    argument_deg = elevation + 7.31 / (elevation + 4.4)
    standard_arcmin = 1.0 / math.tan(math.radians(argument_deg))
    return standard_arcmin * ARCSEC_PER_ARCMIN * _pressure_temperature_factor(state)


def saemundsson_1986_from_true(
    true_elevation_deg: float, state: AtmosphericState
) -> float:
    """Equations (16) and (17) in their published direction. Arcseconds.

    Args:
        true_elevation_deg: True (geometric) elevation, degrees.
        state: Atmospheric state at the observer.

    Returns:
        ``h_apparent - h_true`` in arcseconds.

    Raises:
        ValueError: If ``true_elevation_deg`` is at or below the -5.11 degree
            pole of equation (16).
    """
    value = float(true_elevation_deg)
    if not math.isfinite(value):
        raise ValueError(f"true_elevation_deg must be finite, got {value!r}")
    if value <= -5.11 + 1e-9:
        raise ValueError(
            f"true_elevation_deg must be above the -5.11 degree pole of equation (16), "
            f"got {value!r}"
        )
    argument_deg = value + 10.3 / (value + 5.11)
    standard_arcmin = 1.02 / math.tan(math.radians(argument_deg))
    return standard_arcmin * ARCSEC_PER_ARCMIN * _pressure_temperature_factor(state)


def saemundsson_1986(
    apparent_elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    refractivity_model: str = "ciddor_1996",
) -> float:
    """Equation (16), inverted onto the apparent elevation. Arcseconds.

    The inversion solves ``h_true + R(h_true) / 3600 = h_app`` by Brent's
    method. The residual of that solve is below 1e-9 arcseconds by
    construction of the tolerance and is checked in the test suite; it is a
    numerical inversion of a published formula and introduces no physics.

    Args:
        apparent_elevation_deg: Apparent (observed) elevation, degrees.
        state: Atmospheric state at the observer.
        observer: Ignored.
        refractivity_model: Ignored, as for equation (15).

    Returns:
        ``h_apparent - h_true`` in arcseconds.
    """
    del observer, refractivity_model
    elevation = _check_elevation(apparent_elevation_deg)

    def residual(h_true: float) -> float:
        return h_true + saemundsson_1986_from_true(h_true, state) / 3600.0 - elevation

    low, high = elevation - 1.5, elevation + 0.05
    low = max(low, -5.11 + 1e-6)
    if residual(low) * residual(high) > 0.0:
        # Widen once, then give up rather than return a number from a bracket
        # that does not contain a sign change.
        low = max(elevation - 4.0, -5.11 + 1e-6)
        if residual(low) * residual(high) > 0.0:
            return math.nan
    h_true = brentq(residual, low, high, xtol=1e-13, rtol=1e-15, maxiter=200)
    return saemundsson_1986_from_true(h_true, state)


@dataclass(frozen=True)
class ClosedFormModel:
    """One closed-form refraction model with its published provenance.

    Attributes:
        name: Key used everywhere in this package.
        function: Callable taking ``(apparent_elevation_deg, state, observer,
            refractivity_model)`` and returning arcseconds.
        equation: Equation number in this package's docstrings.
        citation: The published source.
        validity_floor_deg: Apparent elevation below which the source, or this
            product where the source states none, declares the model unusable.
        validity_source: Where ``validity_floor_deg`` comes from.
        uses_refractivity_model: Whether the model consumes a refractivity
            formulation at all. Equations (15) and (16) do not, and therefore
            carry no wavelength dependence.
    """

    name: str
    function: Callable[..., float]
    equation: str
    citation: str
    validity_floor_deg: float
    validity_source: str
    uses_refractivity_model: bool


#: Baseline 1, in full. Iterated over by every disagreement figure.
CLOSED_FORM_MODELS: tuple[ClosedFormModel, ...] = (
    ClosedFormModel(
        name="plane_parallel",
        function=plane_parallel,
        equation="(13)",
        citation="Snell's law; Smart (1977) ch. 3",
        validity_floor_deg=30.0,
        validity_source=(
            "declared by this product at 30 deg as the elevation above which the "
            "neglect of curvature is conventionally tolerated; the model also has "
            "an exact analytic floor at 90 - arcsin(1/n), about 1.36 deg, computed "
            "by plane_parallel_floor_deg"
        ),
        uses_refractivity_model=True,
    ),
    ClosedFormModel(
        name="oriani_series",
        function=oriani_series,
        equation="(14)",
        citation="Oriani's theorem; Smart (1977) ch. 3",
        validity_floor_deg=15.0,
        validity_source=(
            "declared by this product at 15 deg, from the conventional statement "
            "that the two-term series is used to about 75 deg zenith distance; the "
            "measured departure is in validation/validate_model_grid.py. Below that "
            "the series is still returned until the equation (14a) truncation ratio "
            "reaches 1, a breakdown elevation computed by "
            "oriani_breakdown_elevation_deg"
        ),
        uses_refractivity_model=True,
    ),
    ClosedFormModel(
        name="bennett_1982",
        function=bennett_1982,
        equation="(15)",
        citation="Bennett (1982), J. Navigation 35(2), 255-259",
        validity_floor_deg=0.0,
        validity_source="published: 0 to 90 deg, max error 0.07 arcmin vs Garfinkel (1967)",
        uses_refractivity_model=False,
    ),
    ClosedFormModel(
        name="saemundsson_1986",
        function=saemundsson_1986,
        equation="(16)",
        citation="Saemundsson (1986), Sky and Telescope 72, 70, via Meeus (1998) ch. 16",
        validity_floor_deg=0.0,
        validity_source="published: non-negative true elevation",
        uses_refractivity_model=False,
    ),
)

CLOSED_FORM_BY_NAME: dict[str, ClosedFormModel] = {
    model.name: model for model in CLOSED_FORM_MODELS
}
