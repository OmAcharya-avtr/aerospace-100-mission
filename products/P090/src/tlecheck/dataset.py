"""Seeded synthetic element catalogue with complete ground truth.

**This package ships no catalogue, and that is a deliberate limitation, not an
oversight.** The authoritative source of two-line element sets is an operator's
catalogue — Space-Track, or a mission's own orbit determination — and it is not
redistributable, not reproducible from a repository, and not something a
screening tool should bundle a stale copy of. Every element set in this
repository's tests, validation and examples is therefore constructed here from
declared element values, which has one large advantage for a validation
exercise: the corruption injected into an element set is known exactly rather
than inferred, so precision and recall are measured against truth and not
against another screen's opinion.

What the generator is, stated plainly
-------------------------------------
It builds each object's element history **with the SGP4 propagator itself**.
One base element set is drawn per object; each history entry is then produced by
running SGP4 backwards from that base to the entry's epoch, reading the mean
elements SGP4 computed there, converting the mean motion back to the Kozai form
the text format carries, adding seeded orbit-determination jitter, and writing a
conforming two-line element set. The result is an element history that is
*dynamically consistent under the propagator the screens are screening for*,
which matters because the continuity screen of :mod:`tlecheck.continuity`
measures exactly that consistency.

Two earlier versions of this generator were replaced, and both are reinstated
and re-measured alongside the current one by
``validation/validate_baseline_history.py`` so that the figures below are
reproducible rather than remembered. Median propagated disagreement between
consecutive element sets of an uncorrupted object, low-Earth and sun-synchronous:

- version 1, the first-order J2 secular rates of equations (16) and (17) with a
  mean anomaly advanced at ``360 n dt``: **21.03 km** and **49.33 km**;
- version 2, the propagator run backwards with ``Satrec.nm`` written straight
  into the Kozai field: **112.09 km** and **603.64 km**;
- version 3, the construction below: **0.0564 km** and **0.0231 km**.

A continuity screen measuring either of the first two is measuring the
generator.

It is still not a real catalogue, and the consequence of that is stated rather
than implied.

The consequence of that is important and is stated rather than implied: the
precision and recall numbers in this repository are measured against **this
generator's** corruptions, and the *consequence* numbers in kilometres are
measured by **the real SGP4 propagator** operating on these element sets. The
second set of numbers is therefore a true measurement of what SGP4 does with a
corrupted element set; the first set is a measurement of screen behaviour on a
synthetic corruption distribution, and a real corruption distribution would
give different numbers.

Secular rates, kept as an independent analytic check
-----------------------------------------------------
Equations (16) and (17) below are **not** used to build the histories any more.
They are retained as a first-order analytic statement of the same secular
theory the propagator implements, and they are compared against the propagator's
own ``nodedot`` and ``argpdot`` in ``validation/validate_bounds.py``. Two
independent implementations of the same theory agreeing is a different check
from one implementation agreeing with itself.

(16) Nodal regression, first-order J2:

        dOmega/dt = -1.5 n J2 (R_e / p)^2 cos i,   p = a (1 - e^2)

(17) Apsidal precession, first-order J2:

        domega/dt = 0.75 n J2 (R_e / p)^2 (4 - 5 sin^2 i)

    Both with ``n`` in radians per second, giving radians per second. Source:
    Vallado, D. A. (2013), *Fundamentals of Astrodynamics and Applications*,
    4th edition, equations (9-38) and (9-39); the first-order secular rates of
    Brouwer's theory. ``J2 = 0.001082616`` is the WGS-72 value the SGP4
    reference implementation carries.

    **Validity range.** First order in J2, valid for small eccentricity and
    away from the critical inclination of 63.435 degrees, where equation (17)
    vanishes. One generated family sits deliberately at 63.4 degrees so that a
    near-zero apsidal rate is in the corpus.

(18) Un-Kozai transformation and its inverse. The mean motion written in a
     two-line element set is a Kozai mean motion; the mean motion SGP4 works
     with internally, and the one it reports in ``Satrec.nm`` after a
     propagation, is the Brouwer value. The forward transformation is the one
     the reference implementation performs in its ``initl`` step:

         a_k   = (k_e / n_K)^(2/3)
         d_1   = 0.75 J2 (3 cos^2 i - 1) / ((1 - e^2)^(3/2))
         d     = d_1 / a_k^2
         a_del = a_k (1 - d^2 - d (1/3 + 134 d^2 / 81))
         n_B   = n_K / (1 + d_1 / a_del^2)

     Source: Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006),
     *Revisiting Spacetrack Report #3*, AIAA 2006-6753, the ``initl``
     subroutine; after Kozai, Y. (1959), *The motion of a close earth
     satellite*, Astronomical Journal 64, 367, and Brouwer, D. (1959),
     *Solution of the problem of artificial satellite theory without drag*,
     Astronomical Journal 64, 378.

     The inverse is needed to write a history element set and has no closed
     form; :func:`kozai_mean_motion` solves it by the fixed-point iteration
     ``n <- n n_B / n_B(n)``, which converges because ``d`` is of order 1e-3.
     The round trip is exact to the last bit for every element set in the
     corpus, which is asserted in ``tests/test_dataset.py``.

     **Why it matters by 100 km.** Writing the Brouwer mean motion into the
     Kozai field biases the semi-major axis by about half a kilometre, which is
     a period error, which is an along-track drift. Measured by
     ``validation/validate_baseline_history.py``, which reinstates the omission
     and runs both: the propagated disagreement between consecutive uncorrupted
     element sets is a median of **112.09 km** for a low-Earth object and
     **603.64 km** for a sun-synchronous one without the inverse, and
     **0.0564 km** and **0.0231 km** with it -- a factor of 1987 and 26175 from
     one line.

The corruption families
-----------------------
Seven, each with an exact ground-truth label.

``none``
    No corruption.

``checksum_digit``
    The checksum column alone is changed, data intact. The checksum screen
    fires; the propagated position error is zero. This family exists because
    it separates *the screen fired* from *the element set is wrong*, and
    those are different statements.

``data_digit``
    One digit of one numeric field is changed and the stated checksum is left
    as it was, so the checksum mismatches. This is the corruption a checksum
    is for.

``field_shift``
    One character is deleted from the line body, everything to its right
    shifts one column left, a space pads column 68, and **the checksum is
    recomputed**. A ground system that formats and then checksums produces
    exactly this, and the resulting line is internally self-consistent, so no
    checksum screen can see it.

``sign_flip``
    The sign of the B* drag term is flipped and the checksum recomputed. The
    element set is inside every physical bound; the object gains energy
    instead of losing it.

``implausible_eccentricity``
    The eccentricity is replaced with a value far outside the object's own
    regime and the checksum recomputed.

``stale_epoch``
    The candidate is a genuine, uncorrupted element set from earlier in the
    object's own history. Nothing about it is wrong except its age.

``cross_source_divergence``
    A second source publishes a slightly different element set for the same
    object and the candidate is that second source. Both pass format,
    checksum and bounds.

Units
-----
Kilometres, days, degrees, revolutions per day, inverse Earth radii.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bounds import EARTH_RADIUS_KM, semi_major_axis_km
from .elements import ElementSet, apply_checksum, format_tle, parse_tle
from .epoch import epoch_julian_date, julian_date
from .propagate import make_satrec

J2 = 0.001082616
"""WGS-72 second zonal harmonic, dimensionless, as SGP4 carries it."""

XKE = 0.07436691613317342
"""WGS-72 ``sqrt(mu)`` in Earth radii to the 3/2 per minute, as SGP4 carries it.

Read from ``sgp4.api.Satrec.xke`` rather than recomputed, and asserted equal to
it in ``tests/test_dataset.py``, so equation (18) is performed with the same
constant the propagator uses.
"""

CRITICAL_INCLINATION_DEG = 63.435

CORRUPTION_FAMILIES: tuple[str, ...] = (
    "none",
    "checksum_digit",
    "data_digit",
    "field_shift",
    "sign_flip",
    "implausible_eccentricity",
    "stale_epoch",
    "cross_source_divergence",
)


@dataclass(frozen=True)
class Family:
    """One orbit archetype the generator draws objects from.

    Element centres are representative of the regime named; they are not taken
    from any specific object and no numerical value here is quoted from a
    catalogue.
    """

    name: str
    mean_motion_rev_per_day: float
    inclination_deg: float
    eccentricity: float
    bstar: float
    mean_motion_spread: float
    inclination_spread: float
    eccentricity_spread: float


FAMILIES: tuple[Family, ...] = (
    Family("leo_sun_synchronous", 14.60, 98.20, 0.0012, 1.1e-4, 0.08, 0.30, 0.0006),
    Family("leo_low_inclination", 15.50, 51.60, 0.0008, 4.0e-5, 0.10, 0.40, 0.0005),
    Family("leo_decaying", 15.92, 52.00, 0.0021, 1.0e-3, 0.09, 0.50, 0.0009),
    Family("meo_navigation", 2.0056, 55.00, 0.0019, 1.0e-6, 0.0040, 0.60, 0.0010),
    Family("geo_stationary", 1.00272, 0.0600, 0.00020, -2.0e-5, 0.00050, 0.0400, 0.00012),
    Family("heo_critical_inclination", 2.0060, 63.40, 0.7200, 1.2e-4, 0.0050, 0.30, 0.0150),
)


def nodal_regression_deg_per_day(
    mean_motion_rev_per_day: float, eccentricity: float, inclination_deg: float
) -> float:
    """Equation (16) in degrees per day."""
    a = semi_major_axis_km(mean_motion_rev_per_day)
    if not math.isfinite(a):
        return float("nan")
    p = a * (1.0 - eccentricity**2)
    n_rad_s = 2.0 * math.pi * mean_motion_rev_per_day / 86400.0
    cos_i = math.cos(math.radians(inclination_deg))
    rate = -1.5 * n_rad_s * J2 * (EARTH_RADIUS_KM / p) ** 2 * cos_i
    return math.degrees(rate) * 86400.0


def apsidal_precession_deg_per_day(
    mean_motion_rev_per_day: float, eccentricity: float, inclination_deg: float
) -> float:
    """Equation (17) in degrees per day."""
    a = semi_major_axis_km(mean_motion_rev_per_day)
    if not math.isfinite(a):
        return float("nan")
    p = a * (1.0 - eccentricity**2)
    n_rad_s = 2.0 * math.pi * mean_motion_rev_per_day / 86400.0
    sin_i = math.sin(math.radians(inclination_deg))
    rate = 0.75 * n_rad_s * J2 * (EARTH_RADIUS_KM / p) ** 2 * (4.0 - 5.0 * sin_i**2)
    return math.degrees(rate) * 86400.0


@dataclass(frozen=True)
class Record:
    """One screening case with complete ground truth.

    Attributes
    ----------
    satnum:
        Object number, unique within a split.
    family:
        Archetype name.
    corruption:
        Ground-truth corruption family, one of :data:`CORRUPTION_FAMILIES`.
    candidate:
        The element set to be screened.
    reference:
        The uncorrupted element set at the candidate's intended epoch. Used
        only to measure consequence; never given to a screen or to the model.
    history:
        Prior element sets for the object, oldest first.
    other_sources:
        Element sets for the same object from other catalogues, possibly
        empty.
    use_jd:
        Julian date the ephemeris is wanted at, days.
    """

    satnum: int
    family: str
    corruption: str
    candidate: ElementSet
    reference: ElementSet
    history: tuple[ElementSet, ...]
    other_sources: tuple[ElementSet, ...]
    use_jd: float


def _fields(
    satnum: int,
    epoch_year: int,
    epoch_day: float,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
    mean_motion: float,
    bstar: float,
    ndot2: float,
    element_number: int,
    revolution_number: int,
) -> dict[str, object]:
    return {
        "satnum": satnum,
        "epoch_year": epoch_year,
        "epoch_day": epoch_day,
        "inclination_deg": inclination_deg,
        "raan_deg": raan_deg % 360.0,
        "eccentricity": eccentricity,
        "arg_perigee_deg": arg_perigee_deg % 360.0,
        "mean_anomaly_deg": mean_anomaly_deg % 360.0,
        "mean_motion_rev_per_day": mean_motion,
        "bstar": bstar,
        "mean_motion_dot_rev_per_day2": ndot2,
        "designator": f"{epoch_year % 100:02d}{satnum % 1000:03d}A",
        "element_number": element_number,
        "revolution_number": revolution_number,
    }


def unkozai_mean_motion(
    kozai_rad_per_min: float, eccentricity: float, inclination_rad: float
) -> float:
    """Equation (18) forward: Kozai to Brouwer mean motion, radians per minute.

    Raises
    ------
    ValueError
        If ``kozai_rad_per_min`` is not positive, or ``eccentricity`` is not in
        [0, 1).
    """
    if not math.isfinite(kozai_rad_per_min) or kozai_rad_per_min <= 0.0:
        raise ValueError(
            f"kozai_rad_per_min must be positive and finite, got {kozai_rad_per_min!r}"
        )
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity!r}")
    omeosq = 1.0 - eccentricity * eccentricity
    rteosq = math.sqrt(omeosq)
    cosio2 = math.cos(inclination_rad) ** 2
    ak = (XKE / kozai_rad_per_min) ** (2.0 / 3.0)
    d1 = 0.75 * J2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    d = d1 / (ak * ak)
    adel = ak * (1.0 - d * d - d * (1.0 / 3.0 + 134.0 * d * d / 81.0))
    return kozai_rad_per_min / (1.0 + d1 / (adel * adel))


def kozai_mean_motion(
    brouwer_rad_per_min: float,
    eccentricity: float,
    inclination_rad: float,
    iterations: int = 8,
) -> float:
    """Equation (18) inverse, by fixed-point iteration. Radians per minute.

    Raises
    ------
    ValueError
        If ``brouwer_rad_per_min`` is not positive, or ``iterations`` is below 1.
    """
    if not math.isfinite(brouwer_rad_per_min) or brouwer_rad_per_min <= 0.0:
        raise ValueError(
            f"brouwer_rad_per_min must be positive and finite, got {brouwer_rad_per_min!r}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    estimate = brouwer_rad_per_min
    for _ in range(iterations):
        estimate = (
            estimate
            * brouwer_rad_per_min
            / unkozai_mean_motion(estimate, eccentricity, inclination_rad)
        )
    return estimate


JITTER = {
    "inclination_deg": 2.0e-4,
    "raan_deg": 2.0e-3,
    "arg_perigee_deg": 4.0e-3,
    "mean_anomaly_deg": 4.0e-3,
    "eccentricity": 2.0e-6,
    "mean_motion_relative": 1.5e-8,
    "bstar_relative": 0.03,
}
"""Seeded orbit-determination jitter added to every generated element set.

Each entry is a standard deviation in the element's own unit, or a relative
standard deviation where the name says so. They are chosen to put the generated
element-to-element continuity of a low-Earth object in the range a real
catalogue achieves, tens to hundreds of metres, rather than to produce a
particular screening result. The resulting continuity is measured, by family,
in ``validation/validate_rule_screens_output.txt``.
"""


def rewound(
    base: ElementSet,
    offset_days: float,
    source: str = "rewound",
    rng: np.random.Generator | None = None,
    element_number: int = 100,
    revolution_number: int = 10000,
) -> ElementSet | None:
    """The same orbit expressed at a different epoch, via the propagator.

    SGP4 is run from ``base`` to ``base``'s epoch plus ``offset_days``, its own
    secularly updated mean elements are read, the mean motion is converted back
    to the Kozai form the text format carries by equation (18), and a conforming
    element set is written.

    **This is the only honest way to construct an older element set for the same
    object, and the reason is worth stating.** Stamping an earlier epoch onto an
    element set without moving its angles does not produce an older element set;
    it produces a different orbit. The first version of the valid-but-stale
    control in ``validation/validate_consequence.py`` was built that way and the
    number it produced -- 9752.6918 km of position error at an epoch change of
    one and a half days -- was meaningless. That figure was measured during the
    build against code that is no longer here, and the control as it now stands
    is in ``validation/validate_consequence_output.txt`` section 5 with the
    propagator-based construction behind it.

    Parameters
    ----------
    base:
        The element set to move.
    offset_days:
        Days from ``base``'s epoch. Negative moves backwards.
    source:
        Label for the returned element set.
    rng:
        When given, seeded orbit-determination jitter from :data:`JITTER` is
        added to every element. ``None`` adds none, which is what a caller
        constructing a control wants.
    element_number, revolution_number:
        Written into the counter fields.

    Returns
    -------
    ElementSet | None
        ``None`` when the propagator reported an error, because an element set
        that cannot be produced is not quietly replaced by a different one.
    """
    satrec, _refusal = make_satrec(base)
    if satrec is None:
        return None
    error, _position, _velocity = satrec.sgp4_tsince(offset_days * 1440.0)
    if error != 0:
        return None
    eccentricity = max(0.0, min(0.99, float(satrec.em)))
    try:
        kozai = kozai_mean_motion(float(satrec.nm), eccentricity, float(satrec.im))
    except ValueError:
        return None
    mean_motion = kozai * 1440.0 / (2.0 * math.pi)

    def noise(key: str) -> float:
        return JITTER[key] * rng.standard_normal() if rng is not None else 0.0

    fields = _fields(
        satnum=base.satnum,
        epoch_year=base.epoch_full_year,
        epoch_day=base.epoch_day + offset_days,
        inclination_deg=math.degrees(float(satrec.im)) % 180.0
        + noise("inclination_deg"),
        raan_deg=math.degrees(float(satrec.Om)) + noise("raan_deg"),
        eccentricity=max(0.0, eccentricity + noise("eccentricity")),
        arg_perigee_deg=math.degrees(float(satrec.om)) + noise("arg_perigee_deg"),
        mean_anomaly_deg=math.degrees(float(satrec.mm)) + noise("mean_anomaly_deg"),
        mean_motion=mean_motion
        * (
            1.0 + JITTER["mean_motion_relative"] * rng.standard_normal()
            if rng is not None
            else 1.0
        ),
        bstar=base.bstar
        * (
            1.0 + JITTER["bstar_relative"] * rng.standard_normal()
            if rng is not None
            else 1.0
        ),
        ndot2=base.mean_motion_dot_rev_per_day2,
        element_number=element_number,
        revolution_number=revolution_number,
    )
    line1, line2 = format_tle(**fields)  # type: ignore[arg-type]
    return parse_tle(line1, line2, source=source)


def _rewind(
    base: ElementSet,
    offset_days: float,
    rng: np.random.Generator,
    source: str,
    element_number: int,
    revolution_number: int,
) -> ElementSet | None:
    """:func:`rewound` with seeded jitter, used by the generator."""
    return rewound(
        base,
        offset_days,
        source=source,
        rng=rng,
        element_number=element_number,
        revolution_number=revolution_number,
    )


def _series(
    rng: np.random.Generator,
    family: Family,
    satnum: int,
    base_year: int,
    base_day: float,
    history_length: int,
    cadence_days: float,
    source: str,
) -> list[ElementSet]:
    """Generate up to ``history_length + 1`` element sets, oldest first.

    The last entry is the current element set and the earlier ones are its
    history. Entries the propagator could not produce are omitted, so a short
    history is a real outcome and the screens must handle it; that is what
    their determinability verdicts are for.
    """
    mean_motion = (
        family.mean_motion_rev_per_day + family.mean_motion_spread * rng.standard_normal()
    )
    inclination = family.inclination_deg + family.inclination_spread * rng.standard_normal()
    inclination = min(max(inclination, 0.01), 179.99)
    eccentricity = abs(
        family.eccentricity + family.eccentricity_spread * rng.standard_normal()
    )
    eccentricity = min(eccentricity, 0.85)
    bstar = family.bstar * (1.0 + 0.25 * rng.standard_normal())
    ndot2 = max(0.0, bstar) * 0.45

    base_fields = _fields(
        satnum=satnum,
        epoch_year=base_year,
        epoch_day=base_day,
        inclination_deg=inclination,
        raan_deg=float(rng.uniform(0.0, 360.0)),
        eccentricity=eccentricity,
        arg_perigee_deg=float(rng.uniform(0.0, 360.0)),
        mean_anomaly_deg=float(rng.uniform(0.0, 360.0)),
        mean_motion=mean_motion,
        bstar=bstar,
        ndot2=ndot2,
        element_number=100,
        revolution_number=10000,
    )
    base = parse_tle(*format_tle(**base_fields), source=source)  # type: ignore[arg-type]

    sets: list[ElementSet] = []
    for index in range(history_length + 1):
        offset = cadence_days * (index - history_length)
        element_set = _rewind(
            base,
            offset,
            rng,
            source,
            element_number=100 + index,
            revolution_number=10000 + index * 30,
        )
        if element_set is not None:
            sets.append(element_set)
    return sets


ELEMENT_SPANS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((19, 32), (34, 43), (45, 52), (54, 61)),
    2: ((9, 16), (18, 25), (27, 33), (35, 42), (44, 51), (53, 63)),
}
"""One-based column spans of the numeric element fields, by line number.

A ``data_digit`` corruption is confined to these spans. Perturbing a digit of
the element set number or the revolution counter would be a corruption with no
propagated consequence by construction, and a corruption family whose measured
consequence is zero for a reason that has nothing to do with screening would
make the benign class meaningless.
"""


def _digit_columns(line: str, line_number: int) -> list[int]:
    """Zero-based indices of digit columns inside the numeric element fields."""
    columns: list[int] = []
    for low, high in ELEMENT_SPANS[line_number]:
        columns.extend(
            index for index in range(low - 1, high) if line[index].isdigit()
        )
    return columns


def corrupt_checksum_digit(element_set: ElementSet, rng: np.random.Generator) -> ElementSet:
    """Change the checksum column alone, leaving the data untouched."""
    which = int(rng.integers(0, 2))
    lines = [element_set.line1, element_set.line2]
    body = lines[which][:68]
    stated = int(lines[which][68]) if lines[which][68].isdigit() else 0
    replacement = (stated + 1 + int(rng.integers(0, 9))) % 10
    lines[which] = body + str(replacement)
    return parse_tle(lines[0], lines[1], source=element_set.source)


def corrupt_data_digit(element_set: ElementSet, rng: np.random.Generator) -> ElementSet:
    """Change one digit of one numeric field, leaving the stated checksum alone."""
    which = int(rng.integers(0, 2))
    lines = [element_set.line1, element_set.line2]
    line = lines[which]
    columns = _digit_columns(line, which + 1)
    column = int(rng.choice(columns))
    digit = int(line[column])
    replacement = (digit + 1 + int(rng.integers(0, 9))) % 10
    lines[which] = line[:column] + str(replacement) + line[column + 1 :]
    return parse_tle(lines[0], lines[1], source=element_set.source)


def corrupt_field_shift(element_set: ElementSet, rng: np.random.Generator) -> ElementSet:
    """Delete one body character, shift left, pad, and recompute the checksum."""
    which = int(rng.integers(0, 2))
    lines = [element_set.line1, element_set.line2]
    line = lines[which]
    column = int(rng.integers(9, 56))
    body = line[:column] + line[column + 1 : 68] + " "
    lines[which] = apply_checksum(body)
    return parse_tle(lines[0], lines[1], source=element_set.source)


def corrupt_sign_flip(element_set: ElementSet, _rng: np.random.Generator) -> ElementSet:
    """Flip the sign of the B* drag term and recompute the checksum."""
    body = element_set.line1[:68]
    sign = body[53]
    flipped = "-" if sign != "-" else " "
    line1 = apply_checksum(body[:53] + flipped + body[54:])
    return parse_tle(line1, element_set.line2, source=element_set.source)


def corrupt_eccentricity(element_set: ElementSet, rng: np.random.Generator) -> ElementSet:
    """Replace the eccentricity with a value outside the object's regime."""
    target = float(rng.uniform(0.30, 0.97))
    if abs(target - element_set.eccentricity) < 0.2:
        target = min(0.97, element_set.eccentricity + 0.4)
    digits = f"{round(target * 1.0e7):07d}"[:7]
    body = element_set.line2[:68]
    line2 = apply_checksum(body[:26] + digits + body[33:])
    return parse_tle(element_set.line1, line2, source=element_set.source)


AGREEING_SOURCE_SCALE = (0.002, 0.060)
"""Perturbation scale of a second catalogue that agrees, dimensionless.

Two catalogues for the same object usually agree to well under a kilometre.
This range reproduces that, with a tail that crosses the default one-kilometre
divergence threshold, which is what makes the symmetric-pair demonstration of
``validation/validate_rule_screens.py`` possible at all.
"""

DIVERGING_SOURCE_SCALE = (0.5, 3.0)
"""Perturbation scale of the ``cross_source_divergence`` family."""


def _perturbed_source(
    element_set: ElementSet,
    rng: np.random.Generator,
    source: str,
    scale_range: tuple[float, float] = DIVERGING_SOURCE_SCALE,
) -> ElementSet:
    """A second catalogue's element set for the same object, slightly different.

    ``scale_range`` multiplies every element perturbation, so the same code
    produces an agreeing source and a diverging one and the two can never
    drift apart in their construction.
    """
    scale = float(rng.uniform(*scale_range))
    mean_motion = element_set.mean_motion_rev_per_day * (
        1.0 + scale * 1.2e-6 * rng.standard_normal()
    )
    fields = _fields(
        satnum=element_set.satnum,
        epoch_year=element_set.epoch_full_year,
        epoch_day=element_set.epoch_day,
        inclination_deg=element_set.inclination_deg + scale * 0.0015 * rng.standard_normal(),
        raan_deg=element_set.raan_deg + scale * 0.0030 * rng.standard_normal(),
        eccentricity=max(0.0, element_set.eccentricity + scale * 2.0e-6 * rng.standard_normal()),
        arg_perigee_deg=element_set.arg_perigee_deg + scale * 0.0060 * rng.standard_normal(),
        mean_anomaly_deg=element_set.mean_anomaly_deg + scale * 0.0060 * rng.standard_normal(),
        mean_motion=mean_motion,
        bstar=element_set.bstar * (1.0 + scale * 0.05 * rng.standard_normal()),
        ndot2=element_set.mean_motion_dot_rev_per_day2,
        element_number=element_set.element_number,
        revolution_number=element_set.revolution_number,
    )
    line1, line2 = format_tle(**fields)  # type: ignore[arg-type]
    return parse_tle(line1, line2, source=source)


CORRUPTORS = {
    "checksum_digit": corrupt_checksum_digit,
    "data_digit": corrupt_data_digit,
    "field_shift": corrupt_field_shift,
    "sign_flip": corrupt_sign_flip,
    "implausible_eccentricity": corrupt_eccentricity,
}


def generate(
    objects: int = 240,
    seed: int = 20261029,
    corruption_rate: float = 0.5,
    history_length: int = 7,
    cadence_days: float = 1.5,
    use_offset_days: float = 1.0,
    satnum_base: int = 10000,
    base_year: int = 2026,
    base_day: float = 200.5,
) -> list[Record]:
    """Generate ``objects`` screening cases with complete ground truth.

    Parameters
    ----------
    objects:
        Number of objects, divided as evenly as possible across
        :data:`FAMILIES`.
    seed:
        Seed for ``numpy.random.default_rng``. Regeneration is deterministic:
        the same seed gives byte-identical element set text.
    corruption_rate:
        Probability that an object's candidate element set is corrupted at
        all. The corrupted ones are drawn uniformly from the seven corruption
        families.
    history_length:
        Prior element sets per object.
    cadence_days:
        Spacing of the element history, days.
    use_offset_days:
        How far after the current epoch the ephemeris is wanted, days.
    satnum_base:
        First object number. Splits use disjoint blocks.
    base_year, base_day:
        Epoch of the current element set.

    Raises
    ------
    ValueError
        If ``objects`` is below 1, ``corruption_rate`` is outside [0, 1],
        ``history_length`` is below 0, or ``cadence_days`` is not positive.
    """
    if objects < 1:
        raise ValueError(f"objects must be at least 1, got {objects}")
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate!r}")
    if history_length < 0:
        raise ValueError(f"history_length must be non-negative, got {history_length}")
    if not math.isfinite(cadence_days) or cadence_days <= 0.0:
        raise ValueError(f"cadence_days must be positive and finite, got {cadence_days!r}")

    rng = np.random.default_rng(seed)
    corrupt_families = [name for name in CORRUPTION_FAMILIES if name != "none"]
    records: list[Record] = []
    for index in range(objects):
        family = FAMILIES[index % len(FAMILIES)]
        satnum = satnum_base + index
        sets = _series(
            rng, family, satnum, base_year, base_day, history_length, cadence_days, "source_a"
        )
        if len(sets) < 2:
            continue
        history = tuple(sets[:-1])
        current = sets[-1]
        use_jd = epoch_julian_date(current) + use_offset_days

        corruption = "none"
        if float(rng.random()) < corruption_rate:
            corruption = str(rng.choice(corrupt_families))

        candidate = current
        reference = current
        other_sources: tuple[ElementSet, ...] = ()

        if corruption in CORRUPTORS:
            candidate = CORRUPTORS[corruption](current, rng)
        elif corruption == "stale_epoch":
            back = int(rng.integers(2, max(3, history_length) + 1))
            candidate = history[-min(back, len(history))]
            candidate = parse_tle(candidate.line1, candidate.line2, source="source_a")
        elif corruption == "cross_source_divergence":
            candidate = _perturbed_source(current, rng, "source_b", DIVERGING_SOURCE_SCALE)
            other_sources = (current,)
        elif corruption == "none" and float(rng.random()) < 0.5:
            other_sources = (
                _perturbed_source(current, rng, "source_b", AGREEING_SOURCE_SCALE),
            )

        records.append(
            Record(
                satnum=satnum,
                family=family.name,
                corruption=corruption,
                candidate=candidate,
                reference=reference,
                history=history,
                other_sources=other_sources,
                use_jd=use_jd,
            )
        )
    return records


def split(
    name: str,
    objects: int = 240,
    corruption_rate: float = 0.5,
    **overrides: object,
) -> list[Record]:
    """One of the three declared splits, each with a disjoint object block.

    ``"train"``, ``"test"`` and ``"calibration"`` use different seeds and
    different object number blocks, so no object appears in two splits and the
    element text of one split is not reachable from another.

    Raises
    ------
    ValueError
        If ``name`` is not a declared split.
    """
    table = {
        "train": (20261029, 10000),
        "test": (20261030, 40000),
        "calibration": (20261031, 70000),
    }
    if name not in table:
        raise ValueError(f"unknown split {name!r}; expected one of {sorted(table)}")
    seed, base = table[name]
    return generate(
        objects=objects,
        seed=seed,
        corruption_rate=corruption_rate,
        satnum_base=base,
        **overrides,  # type: ignore[arg-type]
    )


def reference_epoch_jd(base_year: int = 2026, base_day: float = 200.5) -> float:
    """Julian date of the generator's nominal current epoch, days."""
    return julian_date(base_year, 1, 1.0) + (base_day - 1.0)
