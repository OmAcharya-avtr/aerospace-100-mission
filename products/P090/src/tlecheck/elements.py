"""Two-line element set parsing, formatting and checksum, by fixed column span.

The two-line element (TLE) format is a fixed-column text record. This module
parses it by column span rather than by whitespace splitting, because a
whitespace split cannot see the fault class this package exists to screen for:
a field whose characters have moved within their own span still splits into
plausible tokens.

Format reference
----------------
Column assignments and the checksum rule are those of the NORAD two-line
element set as documented in:

- Hoots, F. R. and Roehrich, R. L. (1980), *Models for Propagation of NORAD
  Element Sets*, Spacetrack Report No. 3, U.S. Air Force Aerospace Defense
  Command. The element definitions and the mean-element interpretation.
- Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006),
  *Revisiting Spacetrack Report #3*, AIAA 2006-6753. Section 2 for the
  column layout of the distributed text form and for the implied-decimal
  exponential fields.
- Vallado, D. A. (2013), *Fundamentals of Astrodynamics and Applications*,
  4th edition, Microcosm Press, section 9.7 and appendix B.

The checksum is the sum of all digits in columns 1 to 68 with every minus sign
counted as 1 and every other non-digit counted as 0, reduced modulo 10. This
implementation was verified against a published element set for object 25544
(epoch 2019-343.69339541), whose stated checksums are 1 and 2 and whose
computed checksums are 1 and 2; that element set is the known-answer fixture in
``tests/test_elements.py``.

Units
-----
``inclination_deg``, ``raan_deg``, ``arg_perigee_deg``, ``mean_anomaly_deg``
are degrees. ``mean_motion_rev_per_day`` is revolutions per day.
``mean_motion_dot_rev_per_day2`` is revolutions per day squared and is the
field stored as n-dot/2. ``mean_motion_ddot_rev_per_day3`` is revolutions per
day cubed and is the field stored as n-double-dot/6. ``bstar`` is in units of
inverse Earth radii. ``eccentricity`` is dimensionless. ``epoch_day`` is the
fractional day of year, where 1.0 is the start of 1 January.

Validity range
--------------
Nothing here validates physics; it reports what the characters say. Two-digit
epoch years are resolved by the conventional rule of Vallado et al. (2006)
section 2: 57 to 99 means 1957 to 1999, 00 to 56 means 2000 to 2056. That
rule has a 2057 cliff and it is stated as a determinability limit rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LINE_LENGTH = 69
"""Columns in a conforming line, including the checksum."""

CHECKSUM_COLUMN = 69
"""One-based column of the checksum digit."""

EPOCH_YEAR_PIVOT = 57
"""Two-digit epoch years at or above this value are nineteen-hundreds."""


def checksum(line: str) -> int:
    """Modulo-10 checksum of columns 1 to 68 of ``line``.

    Every digit contributes its value, every minus sign contributes 1, every
    other character contributes 0. Columns beyond 68 are ignored, so this can
    be called on a full 69-column line or on the 68-column body.

    Parameters
    ----------
    line:
        Element set line, any length. Only the first 68 characters are read.

    Returns
    -------
    int
        The checksum digit, 0 to 9.
    """
    total = 0
    for character in line[: CHECKSUM_COLUMN - 1]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def apply_checksum(body: str) -> str:
    """Return ``body`` truncated or padded to 68 columns with its checksum appended.

    This is what a ground system does after formatting a line: the checksum is
    computed from the characters that were actually written. It is used
    deliberately in :mod:`tlecheck.dataset` to construct corruptions whose
    checksum is *valid*, because those are the corruptions a checksum screen
    cannot see.
    """
    text = body[: CHECKSUM_COLUMN - 1].ljust(CHECKSUM_COLUMN - 1)
    return text + str(checksum(text))


def _implied_decimal(text: str) -> tuple[float, str | None]:
    """Decode an implied-decimal exponential field such as ``' 38792-4'``.

    The field holds a sign, a mantissa whose decimal point is implied before
    its first digit, then a signed single-digit exponent. ``' 38792-4'`` is
    ``+0.38792e-4``. Returns the value and a fault string, one of which is
    always ``None``.
    """
    raw = text.strip()
    if raw in {"", "0", "+0", "-0"}:
        return 0.0, None
    sign = 1.0
    if raw[0] in "+-":
        sign = -1.0 if raw[0] == "-" else 1.0
        raw = raw[1:]
    if len(raw) < 2 or raw[-2] not in "+-":
        mantissa_text, exponent_text = raw, "0"
    else:
        mantissa_text, exponent_text = raw[:-2], raw[-2:]
    if not mantissa_text.isdigit():
        return 0.0, f"implied-decimal mantissa is not numeric: {text!r}"
    try:
        exponent = int(exponent_text)
    except ValueError:
        return 0.0, f"implied-decimal exponent is not numeric: {text!r}"
    mantissa = float("0." + mantissa_text)
    return sign * mantissa * 10.0**exponent, None


def format_implied_decimal(value: float) -> str:
    """Encode ``value`` into the 8-column implied-decimal exponential field.

    Inverse of :func:`_implied_decimal` to five mantissa digits, which is the
    precision the format carries. A zero is written as the conventional
    ``' 00000-0'``.
    """
    if value == 0.0 or not _finite(value):
        return " 00000-0"
    sign = "-" if value < 0 else " "
    magnitude = abs(value)
    exponent = 0
    while magnitude >= 1.0:
        magnitude /= 10.0
        exponent += 1
    while magnitude < 0.1:
        magnitude *= 10.0
        exponent -= 1
    mantissa = int(round(magnitude * 100000.0))
    if mantissa >= 100000:
        mantissa //= 10
        exponent += 1
    exponent_sign = "-" if exponent < 0 else "+"
    return f"{sign}{mantissa:05d}{exponent_sign}{abs(exponent) % 10:d}"


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _float_span(line: str, lo: int, hi: int, name: str, faults: list[str]) -> float:
    """Read a plain decimal field from one-based columns ``lo`` to ``hi``."""
    text = line[lo - 1 : hi]
    try:
        return float(text)
    except ValueError:
        faults.append(f"{name}: columns {lo}-{hi} are not a number: {text!r}")
        return float("nan")


@dataclass(frozen=True)
class ElementSet:
    """One parsed two-line element set, with the faults found while parsing.

    Attributes carry the units given in the module docstring. ``faults`` is
    empty for a conforming record; it is never a verdict, only a list of things
    the characters could not supply.
    """

    satnum: int
    classification: str
    designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_dot_rev_per_day2: float
    mean_motion_ddot_rev_per_day3: float
    bstar: float
    ephemeris_type: int
    element_number: int
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float
    revolution_number: int
    line1: str
    line2: str
    source: str = "unnamed"
    faults: tuple[str, ...] = field(default_factory=tuple)

    @property
    def epoch_full_year(self) -> int:
        """Four-digit epoch year under the 1957-2056 window convention."""
        if self.epoch_year >= EPOCH_YEAR_PIVOT:
            return 1900 + self.epoch_year
        return 2000 + self.epoch_year

    @property
    def checksum_line1_stated(self) -> int | None:
        """Checksum digit as written on line 1, or ``None`` if not a digit."""
        return _stated_checksum(self.line1)

    @property
    def checksum_line2_stated(self) -> int | None:
        """Checksum digit as written on line 2, or ``None`` if not a digit."""
        return _stated_checksum(self.line2)

    @property
    def checksum_line1_computed(self) -> int:
        """Checksum of line 1 columns 1 to 68."""
        return checksum(self.line1)

    @property
    def checksum_line2_computed(self) -> int:
        """Checksum of line 2 columns 1 to 68."""
        return checksum(self.line2)

    @property
    def parsable(self) -> bool:
        """``True`` when every numeric field yielded a finite number."""
        return not self.faults


def _stated_checksum(line: str) -> int | None:
    if len(line) < CHECKSUM_COLUMN:
        return None
    digit = line[CHECKSUM_COLUMN - 1]
    return int(digit) if digit.isdigit() else None


def parse_tle(line1: str, line2: str, source: str = "unnamed") -> ElementSet:
    """Parse two element set lines by column span, never raising.

    A malformed line produces an :class:`ElementSet` whose ``faults`` describe
    what could not be read and whose unreadable numeric fields are ``nan``.
    Every fault string begins with ``"line 1: "`` or ``"line 2: "`` so that
    :func:`tlecheck.screens.format_screen` can attribute it without guessing.
    Raising would be the wrong contract for a screening tool: the caller asked
    whether the record is usable, and an exception answers a different
    question.

    Parameters
    ----------
    line1, line2:
        The two lines, with or without trailing newline. Lines shorter than 69
        columns are right-padded with spaces and the shortfall is recorded as a
        fault.
    source:
        Free-text label for the catalogue or file the lines came from. Used by
        the cross-source screen to say which source disagreed.

    Returns
    -------
    ElementSet
    """
    faults: list[str] = []
    one = _normalise(line1, "line 1", faults)
    two = _normalise(line2, "line 2", faults)
    if one[:1] != "1":
        faults.append(f"line 1: column 1 is {one[:1]!r}, expected '1'")
    if two[:1] != "2":
        faults.append(f"line 2: column 1 is {two[:1]!r}, expected '2'")

    satnum1 = _int_span(one, 3, 7, "line 1: satellite number", faults)
    satnum2 = _int_span(two, 3, 7, "line 2: satellite number", faults)
    if satnum1 is not None and satnum2 is not None and satnum1 != satnum2:
        faults.append(f"satellite number differs between lines: {satnum1} and {satnum2}")

    classification = one[7]
    designator = one[9:17].strip()
    epoch_year = _int_span(one, 19, 20, "line 1: epoch year", faults)
    epoch_day = _float_span(one, 21, 32, "line 1: epoch day", faults)
    ndot = _float_span(one, 34, 43, "line 1: mean motion first derivative", faults)
    nddot, nddot_fault = _implied_decimal(one[44:52])
    if nddot_fault:
        faults.append("line 1: mean motion second derivative: " + nddot_fault)
    bstar, bstar_fault = _implied_decimal(one[53:61])
    if bstar_fault:
        faults.append("line 1: bstar: " + bstar_fault)
    ephemeris_type = _int_span(one, 63, 63, "line 1: ephemeris type", faults)
    element_number = _int_span(one, 65, 68, "line 1: element set number", faults)

    inclination = _float_span(two, 9, 16, "line 2: inclination", faults)
    raan = _float_span(two, 18, 25, "line 2: right ascension of ascending node", faults)
    eccentricity_text = two[26:33]
    if eccentricity_text.strip().isdigit():
        eccentricity = float("0." + eccentricity_text.strip().rjust(7, "0"))
    else:
        faults.append(
            f"line 2: eccentricity: columns 27-33 are not digits: {eccentricity_text!r}"
        )
        eccentricity = float("nan")
    arg_perigee = _float_span(two, 35, 42, "line 2: argument of perigee", faults)
    mean_anomaly = _float_span(two, 44, 51, "line 2: mean anomaly", faults)
    mean_motion = _float_span(two, 53, 63, "line 2: mean motion", faults)
    revolution_number = _int_span(two, 64, 68, "line 2: revolution number", faults)

    return ElementSet(
        satnum=satnum1 if satnum1 is not None else -1,
        classification=classification,
        designator=designator,
        epoch_year=epoch_year if epoch_year is not None else -1,
        epoch_day=epoch_day,
        mean_motion_dot_rev_per_day2=ndot,
        mean_motion_ddot_rev_per_day3=nddot,
        bstar=bstar,
        ephemeris_type=ephemeris_type if ephemeris_type is not None else -1,
        element_number=element_number if element_number is not None else -1,
        inclination_deg=inclination,
        raan_deg=raan,
        eccentricity=eccentricity,
        arg_perigee_deg=arg_perigee,
        mean_anomaly_deg=mean_anomaly,
        mean_motion_rev_per_day=mean_motion,
        revolution_number=revolution_number if revolution_number is not None else -1,
        line1=one,
        line2=two,
        source=source,
        faults=tuple(faults),
    )


def _normalise(line: str, label: str, faults: list[str]) -> str:
    text = line.rstrip("\r\n")
    if len(text) < LINE_LENGTH:
        faults.append(f"{label}: {len(text)} columns, expected {LINE_LENGTH}")
    elif len(text) > LINE_LENGTH:
        faults.append(f"{label}: {len(text)} columns, expected {LINE_LENGTH}")
    return text.ljust(LINE_LENGTH)[:LINE_LENGTH]


def _int_span(line: str, lo: int, hi: int, name: str, faults: list[str]) -> int | None:
    text = line[lo - 1 : hi].strip()
    if not text:
        faults.append(f"{name}: columns {lo}-{hi} are blank")
        return None
    if not text.lstrip("+-").isdigit():
        faults.append(f"{name}: columns {lo}-{hi} are not an integer: {text!r}")
        return None
    return int(text)


def format_tle(
    *,
    satnum: int,
    epoch_year: int,
    epoch_day: float,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
    mean_motion_rev_per_day: float,
    bstar: float = 0.0,
    mean_motion_dot_rev_per_day2: float = 0.0,
    mean_motion_ddot_rev_per_day3: float = 0.0,
    classification: str = "U",
    designator: str = "98067A",
    ephemeris_type: int = 0,
    element_number: int = 999,
    revolution_number: int = 10000,
) -> tuple[str, str]:
    """Write two conforming element set lines, checksums included.

    This exists because this package ships no catalogue. Every element set in
    the test suite, the validation scripts and the examples is written here
    from declared element values, so the corruption injected into it is known
    exactly rather than inferred.

    The round trip ``parse_tle(*format_tle(**fields))`` reproduces every field
    to the precision the format carries, and ``format_tle`` applied to the
    fields of a published element set for object 25544 reproduces that element
    set's two lines character for character. Both are asserted in
    ``tests/test_elements.py``.
    """
    designator_text = designator.ljust(6)[:6]
    body1 = (
        "1 "
        + f"{satnum:05d}"[:5]
        + classification[:1]
        + " "
        + designator_text[:2]
        + designator_text[2:5]
        + designator_text[5:6].ljust(3)
        + " "
        + f"{epoch_year % 100:02d}"
        + f"{epoch_day:12.8f}"
        + " "
        + _signed_decimal_point(mean_motion_dot_rev_per_day2)
        + " "
        + format_implied_decimal(mean_motion_ddot_rev_per_day3)
        + " "
        + format_implied_decimal(bstar)
        + " "
        + f"{ephemeris_type:1d}"
        + " "
        + f"{element_number:>4d}"
    )
    body2 = (
        "2 "
        + f"{satnum:05d}"[:5]
        + " "
        + f"{inclination_deg:8.4f}"
        + " "
        + f"{raan_deg:8.4f}"
        + " "
        + f"{round(eccentricity * 1.0e7):07d}"[:7]
        + " "
        + f"{arg_perigee_deg:8.4f}"
        + " "
        + f"{mean_anomaly_deg:8.4f}"
        + " "
        + f"{mean_motion_rev_per_day:11.8f}"
        + f"{revolution_number % 100000:5d}"
    )
    return apply_checksum(body1), apply_checksum(body2)


def _signed_decimal_point(value: float) -> str:
    """Encode the n-dot/2 field: a sign column then ``.`` and eight digits."""
    sign = "-" if value < 0 else " "
    text = f"{abs(value):.8f}"
    if text.startswith("0."):
        text = text[1:]
    return (sign + text)[:10].ljust(10)
