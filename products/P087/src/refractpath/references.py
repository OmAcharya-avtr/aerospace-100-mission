"""Published reference values this package did not generate, and the checks against them.

Read this first, because the honesty of everything downstream rests on it.

A product that scores its own implementations against its own other
implementation has verified consistency, not correctness. This module holds the
external material: numbers and accuracy claims that were published by other
people, before this package existed, at conditions those publications state.
Each entry records what kind of reference it is, because the three kinds are
not equally strong and collapsing them would be the dishonest move:

``published_value``
    A number stated in a published source at stated conditions. The strongest
    kind available here.

``published_claim``
    An accuracy claim made by the author of a formula about that formula,
    relative to a computation this package does not implement. Checking it
    tests the claim against *this* package's reference, which is not the
    reference the claim was made against; a miss therefore does not prove the
    claim wrong, and the check says so in its own note.

``convention``
    A value adopted by a published authority as a working convention rather
    than measured. The Astronomical Almanac's horizon refraction is of this
    kind: it is a rounded figure chosen to make a rise-and-set calculation
    repeatable, not a measurement, and the true horizon refraction varies by
    far more than the figure's own precision.

What this module does **not** contain is a transcription of a printed
refraction table. No such table was available to this build, and inventing
plausible table entries would be worse than having none. That gap is stated in
``validation/VALIDATION.md`` and in the README, and it is the main weakness of
this product's external verification. The checks below are what there is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PublishedReference:
    """One external reference value or claim.

    Attributes:
        key: Identifier used in the validation output.
        quantity: What is being compared, with its units.
        value: The published number, in the units named in ``quantity``.
        tolerance: Absolute tolerance for the comparison, same units. Chosen
            from the published precision of the value where the source states
            one and declared here where it does not; which of the two it is, is
            stated in ``tolerance_source``.
        tolerance_source: Where the tolerance comes from.
        conditions: The conditions at which the published value applies.
        citation: The published source.
        kind: ``"published_value"``, ``"published_claim"`` or ``"convention"``.
        note: What a miss would and would not prove.
    """

    key: str
    quantity: str
    value: float
    tolerance: float
    tolerance_source: str
    conditions: str
    citation: str
    kind: str
    note: str


#: Refractivity of standard air at 633 nm, from the Birch and Downs (1994)
#: dispersion equation. This is the single most widely quoted refractive-index
#: value for air and it is quoted to five significant figures as 2.7653e-4.
REFRACTIVITY_633NM = PublishedReference(
    key="refractivity_standard_air_633nm",
    quantity="n - 1 of standard air at 633 nm, dimensionless",
    value=2.7653e-4,
    tolerance=5e-9,
    tolerance_source=(
        "half a unit in the last quoted digit of the published five-figure value"
    ),
    conditions="15 C, 101325 Pa, dry, 450 ppm CO2, vacuum wavelength 633 nm",
    citation=(
        "Birch, K. P. and Downs, M. J. (1993), Metrologia 30(3), 155-162, with the "
        "1994 correction, Metrologia 31(4), 315-316"
    ),
    kind="published_value",
    note=(
        "This checks the transcription and evaluation of equation (6) against the "
        "number its own authors publish. A miss here is a defect in this package."
    ),
)

#: Density of standard dry air. ISO 2533 / US Standard Atmosphere 1976 give
#: 1.2250 kg/m^3 at 15 C and 101325 Pa for the ideal-gas model; the Ciddor
#: (1996) real-gas density of equation (11) sits above it by the
#: compressibility correction, which is itself about 4.3e-4 of the value.
DENSITY_STANDARD_AIR = PublishedReference(
    key="density_standard_dry_air",
    quantity="density of standard dry air, kg/m^3",
    value=1.2250,
    tolerance=0.0010,
    tolerance_source=(
        "declared by this product, set to cover the real-gas compressibility "
        "correction of about 4.3e-4 relative that separates the ideal-gas standard "
        "value from the BIPM-1991 value of equation (11)"
    ),
    conditions="15 C, 101325 Pa, dry air",
    citation="US Standard Atmosphere 1976, NOAA/NASA/USAF, table 1; ISO 2533:1975",
    kind="published_value",
    note=(
        "A cross-check on equation (11), which no other formulation in this package "
        "uses, through a published quantity that is not a refractivity."
    ),
)

#: Saturation vapour pressure of water at 20 C. 2339 Pa to four figures is a
#: standard tabulated value and both equation (4) and equation (12) must
#: reproduce it.
SATURATION_VAPOUR_PRESSURE_20C = PublishedReference(
    key="saturation_vapour_pressure_20c",
    quantity="saturation vapour pressure over liquid water at 20 C, Pa",
    value=2339.0,
    tolerance=2.0,
    tolerance_source=(
        "declared by this product at the size of the disagreement between the "
        "tabulated figure's own rounding and the two formulations compared"
    ),
    conditions="20 C, over liquid water",
    citation=(
        "Buck, A. L. (1981), J. Applied Meteorology 20, 1527-1532, equation (8); the "
        "IAPWS form quoted by Ciddor, P. E. (1996), Applied Optics 35(9), 1566-1573"
    ),
    kind="published_value",
    note=(
        "Checks equations (4) and (12) independently against a value neither was "
        "fitted to in this package."
    ),
)

#: The constant of refraction: the coefficient of tan z in equation (14). The
#: figure quoted in spherical-astronomy texts for the visible band at 10 C and
#: about 1013 hPa is 58.2 to 58.3 arcsec, and the interval, not the midpoint,
#: is what is published.
CONSTANT_OF_REFRACTION = PublishedReference(
    key="constant_of_refraction",
    quantity="constant of refraction, the coefficient of tan z in equation (14), arcsec",
    value=58.25,
    tolerance=0.25,
    tolerance_source=(
        "the width of the published interval 58.2 to 58.3 arcsec, taken as a "
        "midpoint and a half-width rather than narrowed"
    ),
    conditions="10 C, 1013.25 hPa, dry, visible band taken at 550 nm",
    citation="Smart, W. M. (1977), Textbook on Spherical Astronomy, 6th ed., ch. 3",
    kind="published_value",
    note=(
        "The published figure is given for the visible band without a stated "
        "wavelength; 550 nm is this product's choice and the sensitivity of the "
        "measured value to that choice is reported beside the check."
    ),
)

#: The Astronomical Almanac's adopted horizon refraction. A convention: 34
#: arcmin of refraction plus 16 arcmin of solar semidiameter is the 50 arcmin
#: standard depression used for sunrise and sunset.
HORIZON_REFRACTION_CONVENTION = PublishedReference(
    key="horizon_refraction_convention",
    quantity="refraction at zero apparent elevation, arcsec",
    value=34.0 * 60.0,
    tolerance=1.0 * 60.0,
    tolerance_source=(
        "declared by this product at one arcminute, which is the precision at which "
        "the convention itself is stated; the real quantity varies by several "
        "arcminutes with the surface lapse rate, as equation (23) measures"
    ),
    conditions="standard conditions, apparent elevation zero",
    citation=(
        "the 34 arcmin refraction adopted with 16 arcmin of semidiameter to give the "
        "50 arcmin standard depression for rise and set; Explanatory Supplement to "
        "the Astronomical Almanac, 3rd ed., Urban and Seidelmann (eds.), 2013"
    ),
    kind="convention",
    note=(
        "Agreement here is weak evidence and disagreement is weaker still: the "
        "adopted figure is a rounded convention and the quantity it stands for is "
        "the most profile-sensitive number this package computes."
    ),
)

#: Bennett's own accuracy claim for equation (15).
BENNETT_ACCURACY_CLAIM = PublishedReference(
    key="bennett_accuracy_claim",
    quantity="maximum error of equation (15) over 0 to 90 deg, arcsec",
    value=0.07 * 60.0,
    tolerance=0.0,
    tolerance_source=(
        "none: the published figure is itself the tolerance, so the check is whether "
        "the measured maximum departure is at or below it"
    ),
    conditions="1010 hPa, 10 C, apparent elevation 0 to 90 deg",
    citation=(
        "Bennett, G. G. (1982), J. Navigation 35(2), 255-259, relative to Garfinkel, "
        "B. (1967), Astronomical Journal 72, 235-254; quoted in Meeus, J. (1998), "
        "Astronomical Algorithms, 2nd ed., ch. 16"
    ),
    kind="published_claim",
    note=(
        "The claim was made against Garfinkel (1967), which this package does not "
        "implement. The check measures equation (15) against this package's ray "
        "trace instead. A miss does not show Bennett wrong; it shows that the two "
        "reference atmospheres differ by more than 0.07 arcmin wherever it misses, "
        "and equation (23) says how much freedom there is in that choice."
    ),
)

#: Every reference, in the order the validation script reports them.
PUBLISHED_REFERENCES: tuple[PublishedReference, ...] = (
    REFRACTIVITY_633NM,
    DENSITY_STANDARD_AIR,
    SATURATION_VAPOUR_PRESSURE_20C,
    CONSTANT_OF_REFRACTION,
    HORIZON_REFRACTION_CONVENTION,
    BENNETT_ACCURACY_CLAIM,
)

PUBLISHED_BY_KEY: dict[str, PublishedReference] = {
    reference.key: reference for reference in PUBLISHED_REFERENCES
}


@dataclass(frozen=True)
class ReferenceComparison:
    """The result of comparing a measured value against a published one.

    Attributes:
        reference: The published reference.
        measured: What this package computed.
        difference: ``measured - reference.value``.
        within_tolerance: Whether ``abs(difference) <= reference.tolerance``,
            or for a ``published_claim`` whose tolerance is zero, whether
            ``measured <= reference.value``.
        detail: One line of context, including anything that would change the
            comparison.
    """

    reference: PublishedReference
    measured: float
    difference: float
    within_tolerance: bool
    detail: str = ""


def compare(
    reference: PublishedReference, measured: float, detail: str = ""
) -> ReferenceComparison:
    """Compare a measured value against a published one.

    For a ``published_claim`` with a zero tolerance the published number is an
    upper bound on an error, so the test is one-sided: the measured maximum
    departure must not exceed it.

    Args:
        reference: The published reference.
        measured: The measured value, in the reference's units.
        detail: Context to carry into the report.

    Returns:
        The comparison. Nothing is raised on a miss; a miss is a result.
    """
    difference = measured - reference.value
    if reference.kind == "published_claim" and reference.tolerance == 0.0:
        within = bool(math.isfinite(measured) and measured <= reference.value)
    else:
        within = bool(math.isfinite(measured) and abs(difference) <= reference.tolerance)
    return ReferenceComparison(
        reference=reference,
        measured=float(measured),
        difference=float(difference),
        within_tolerance=within,
        detail=detail,
    )
