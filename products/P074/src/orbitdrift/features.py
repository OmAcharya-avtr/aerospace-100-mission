"""Orbit and object features, and what each one is a proxy for.

The feature set is the one the specification names -- altitude, eccentricity,
inclination, ``B*`` and epoch age -- plus two derived quantities that make the
physics available to a linear model without making it available to only the
learned one. Both predictors in :mod:`orbitdrift.predictors` see exactly the
same columns, which is the only way the benchmark between them means anything.

``log10_perigee_density``
    Base-ten logarithm of the modelled atmospheric density at perigee
    altitude. Drag scales with density, density varies by five orders of
    magnitude over the altitude range of interest, and a model given altitude
    alone has to learn that exponential relation from data. Giving it the
    logarithm of the density instead is not a trick: it is the physically
    correct variable, and withholding it in order to make the learned model
    look better would be the trick.

``log10_drag_parameter``
    ``log10(C_D A/m * rho_perigee)``, the combination that actually appears in
    the drag acceleration. ``B*`` alone cannot express it because the same
    ``B*`` means a very different acceleration at 400 km and at 1400 km.

``B*`` is signed and can be negative, so it is carried as a signed value and
as ``log10`` of its magnitude with the sign in a separate column, rather than
as a logarithm that would be undefined. ``tests/test_features.py`` covers the
negative and the exactly-zero cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sgp4.api import Satrec

from .constants import SECONDS_PER_DAY, WGS72
from .forces import atmospheric_density_kg_m3, cd_area_over_mass_m2_kg

FEATURE_NAMES: tuple[str, ...] = (
    "perigee_altitude_km",
    "apogee_altitude_km",
    "eccentricity",
    "inclination_deg",
    "bstar",
    "log10_abs_bstar",
    "bstar_sign",
    "period_min",
    "log10_perigee_density",
    "log10_drag_parameter",
)
"""Static per-object features, in the column order every model here uses."""

TIME_FEATURE_NAME = "epoch_age_days"
"""The one feature that varies along a divergence curve."""

ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + (TIME_FEATURE_NAME,)

_LOG_FLOOR = 1e-30


@dataclass(frozen=True)
class ObjectFeatures:
    """Static features of one object. Units are in the field names."""

    perigee_altitude_km: float
    apogee_altitude_km: float
    eccentricity: float
    inclination_deg: float
    bstar: float
    log10_abs_bstar: float
    bstar_sign: float
    period_min: float
    log10_perigee_density: float
    log10_drag_parameter: float

    def as_array(self) -> np.ndarray:
        """Features in :data:`FEATURE_NAMES` order, shape ``(10,)``."""
        values = asdict(self)
        return np.array([values[name] for name in FEATURE_NAMES], dtype=float)


def features_from_satrec(satrec: Satrec) -> ObjectFeatures:
    """Extract :class:`ObjectFeatures` from an initialised ``Satrec``.

    ``satrec.alta`` and ``satrec.altp`` are apogee and perigee height in Earth
    radii, as SGP4 computes them from the mean elements; they are multiplied by
    the WGS72 radius because that is the radius SGP4 used to form them.
    """
    perigee_km = float(satrec.altp) * WGS72.radius_km
    apogee_km = float(satrec.alta) * WGS72.radius_km
    bstar = float(satrec.bstar)
    coefficient = float(cd_area_over_mass_m2_kg(bstar))
    density = float(atmospheric_density_kg_m3(max(perigee_km, 0.0)))
    drag_parameter = abs(coefficient) * density
    mean_motion_rad_min = float(satrec.no_kozai)
    period_min = 2.0 * np.pi / mean_motion_rad_min if mean_motion_rad_min > 0.0 else np.inf
    return ObjectFeatures(
        perigee_altitude_km=perigee_km,
        apogee_altitude_km=apogee_km,
        eccentricity=float(satrec.ecco),
        inclination_deg=float(np.rad2deg(satrec.inclo)),
        bstar=bstar,
        log10_abs_bstar=float(np.log10(max(abs(bstar), _LOG_FLOOR))),
        bstar_sign=float(np.sign(bstar)),
        period_min=period_min,
        log10_perigee_density=float(np.log10(max(density, _LOG_FLOOR))),
        log10_drag_parameter=float(np.log10(max(drag_parameter, _LOG_FLOOR))),
    )


def design_matrix(
    static: np.ndarray, ages_days: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Broadcast static features against epoch ages into a design matrix.

    Parameters
    ----------
    static
        Shape ``(n, len(FEATURE_NAMES))``.
    ages_days
        Shape ``(n,)``, one age per row -- already flattened, so an object
        contributing many ages appears as many rows.

    Returns
    -------
    (matrix, names)
        ``matrix`` has shape ``(n, len(ALL_FEATURE_NAMES))``.
    """
    s = np.atleast_2d(np.asarray(static, dtype=float))
    t = np.asarray(ages_days, dtype=float).reshape(-1, 1)
    if s.shape[0] != t.shape[0]:
        raise ValueError(
            f"static has {s.shape[0]} rows but ages_days has {t.shape[0]}; "
            "they must be flattened to the same length"
        )
    if s.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"static must have {len(FEATURE_NAMES)} columns ({FEATURE_NAMES}); "
            f"got {s.shape[1]}"
        )
    return np.hstack([s, t]), ALL_FEATURE_NAMES


def period_s_from_satrec(satrec: Satrec) -> float:
    """Keplerian period implied by the TLE mean motion, seconds."""
    mean_motion_rad_min = float(satrec.no_kozai)
    if mean_motion_rad_min <= 0.0:
        raise ValueError(f"mean motion {mean_motion_rad_min} rad/min is not positive")
    return 60.0 * 2.0 * np.pi / mean_motion_rad_min


def epoch_jd(satrec: Satrec) -> float:
    """TLE epoch as a single Julian date.

    ``sgp4`` splits the epoch into a whole part and a fraction specifically so
    that the sum is not formed; forming it here costs about 1e-8 day of
    resolution, which is 1 ms, and that is recorded rather than ignored: it is
    below the 15 s integration step by four orders of magnitude.
    """
    return float(satrec.jdsatepoch) + float(satrec.jdsatepochF)


def seconds_to_days(seconds: np.ndarray | float) -> np.ndarray:
    """Convert seconds to days."""
    return np.asarray(seconds, dtype=float) / SECONDS_PER_DAY
