"""Construction and parsing of two-line element sets.

Why this module exists
----------------------
This package needs spacecraft for its examples, tests and validation. It does
**not** ship real catalogue elements. Every TLE used anywhere in this repository
is *synthesised* here from orbital elements that are written down in a committed
script, formatted into the standard two-line layout with correct checksums, and
then parsed back by ``sgp4`` so the round trip is verified. Nothing here is a
real object, no epoch corresponds to a real observation, and no number in this
repository is a statement about any catalogued spacecraft.

That choice is deliberate. A product that quotes catalogue elements it cannot
re-derive is quoting data it cannot verify, and a stale TLE would silently
change every measured number in the repository on the day it was refreshed.
Synthesising the elements makes the orbit a declared input.

Format
------
The layout is the NORAD two-line element set as described in Hoots & Roehrich
(1980), *Spacetrack Report No. 3*, and restated in Vallado et al. (2006),
*Revisiting Spacetrack Report #3*, AIAA 2006-6753. Fixed 69-character lines; the
eccentricity field carries an implied leading decimal point; ``bstar`` uses the
exponent-suffix form; the final character of each line is the modulo-10 sum of
the digits with each minus sign counted as one.

Units
-----
Angles in degrees, mean motion in revolutions per day, ``bstar`` in inverse
Earth radii, epoch as a two-digit year plus fractional day of year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TLE_LINE_LENGTH = 69
"""Characters in one TLE line, excluding the newline."""


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum of the first 68 characters of a TLE line.

    Digits add their value, a minus sign adds one, every other character adds
    nothing. Hoots & Roehrich (1980).
    """
    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def _exponent_field(value: float) -> str:
    """Format ``value`` in the eight-character exponent-suffix TLE form.

    ``0.00011606`` becomes ``" 11606-3"``; a negative value takes a leading
    minus. Zero is the canonical ``" 00000-0"``.
    """
    if value == 0.0:
        return " 00000-0"
    sign = "-" if value < 0.0 else " "
    magnitude = abs(float(value))
    exponent = int(math.floor(math.log10(magnitude))) + 1
    mantissa = int(round(magnitude / (10.0**exponent) * 1e5))
    if mantissa >= 100000:
        mantissa //= 10
        exponent += 1
    if not -9 <= exponent <= 9:
        raise ValueError(f"value {value} does not fit the TLE exponent field")
    return f"{sign}{mantissa:05d}{exponent:+d}"


@dataclass(frozen=True)
class OrbitElements:
    """Mean Keplerian elements in the form a TLE carries them.

    Attributes
    ----------
    satnum : int
        Catalogue number placeholder, 1 to 99999. Not a real catalogue number.
    epoch_year : int
        Two-digit year, 0 to 99, interpreted by ``sgp4`` as 2000-2056 for
        values below 57.
    epoch_day : float
        Fractional day of year, 1.0 to 367.0.
    inclination_deg : float
        Degrees, 0 to 180.
    raan_deg : float
        Right ascension of the ascending node, degrees.
    eccentricity : float
        Dimensionless, 0 to below 1. The TLE field holds seven digits, so the
        stored value is quantised to 1e-7.
    argp_deg : float
        Argument of perigee, degrees.
    mean_anomaly_deg : float
        Degrees.
    mean_motion_rev_day : float
        Revolutions per solar day, above 0.
    bstar : float
        SGP4 drag term, inverse Earth radii.
    name : str
        Label used in reports. Not part of the two-line format.
    """

    satnum: int
    epoch_year: int
    epoch_day: float
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argp_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float
    bstar: float = 0.0
    name: str = "SYNTHETIC"

    def __post_init__(self) -> None:
        if not 1 <= int(self.satnum) <= 99999:
            raise ValueError(f"satnum must be within [1, 99999], got {self.satnum}")
        if not 0 <= int(self.epoch_year) <= 99:
            raise ValueError(f"epoch_year must be a two-digit year, got {self.epoch_year}")
        if not 1.0 <= float(self.epoch_day) < 367.0:
            raise ValueError(f"epoch_day must be within [1, 367), got {self.epoch_day}")
        if not 0.0 <= float(self.inclination_deg) <= 180.0:
            raise ValueError(
                f"inclination must be within [0, 180] deg, got {self.inclination_deg}"
            )
        if not 0.0 <= float(self.eccentricity) < 1.0:
            raise ValueError(f"eccentricity must be within [0, 1), got {self.eccentricity}")
        if float(self.mean_motion_rev_day) <= 0.0:
            raise ValueError(
                f"mean motion must be positive rev/day, got {self.mean_motion_rev_day}"
            )
        if float(self.mean_motion_rev_day) >= 100.0:
            raise ValueError(
                "mean motion must be below 100 rev/day to fit the TLE field, got "
                f"{self.mean_motion_rev_day}"
            )

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis implied by the mean motion, km.

        Kepler's third law with the WGS-72 gravitational parameter SGP4 uses:
        ``a = (mu / n**2)**(1/3)`` with ``n`` in rad/s. This is the osculating
        value of the *mean* mean motion and is not the SGP4 internal semi-major
        axis, which includes a secular correction. It is used only for the loose
        pass-duration bound in :mod:`contactsched.geometry`.
        """
        from .geometry import MU_EARTH_KM3_S2

        n_rad_s = float(self.mean_motion_rev_day) * 2.0 * math.pi / 86400.0
        return (MU_EARTH_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)

    def to_lines(self, element_number: int = 999, rev_number: int = 1) -> tuple[str, str]:
        """Render the elements as a checksummed two-line element set.

        Returns
        -------
        tuple of str
            Two strings of exactly 69 characters each.
        """
        line1 = (
            "1 "
            f"{int(self.satnum):05d}"
            "U "
            "26001A   "
            f"{int(self.epoch_year):02d}"
            f"{float(self.epoch_day):012.8f}"
            " "
            " .00000000"
            " "
            " 00000-0"
            " "
            f"{_exponent_field(float(self.bstar))}"
            " 0 "
            f"{int(element_number):04d}"
        )
        line2 = (
            "2 "
            f"{int(self.satnum):05d}"
            " "
            f"{float(self.inclination_deg):8.4f}"
            " "
            f"{float(self.raan_deg) % 360.0:8.4f}"
            " "
            f"{int(round(float(self.eccentricity) * 1e7)):07d}"
            " "
            f"{float(self.argp_deg) % 360.0:8.4f}"
            " "
            f"{float(self.mean_anomaly_deg) % 360.0:8.4f}"
            " "
            f"{float(self.mean_motion_rev_day):11.8f}"
            f"{int(rev_number):05d}"
        )
        if len(line1) != 68:
            raise ValueError(f"line 1 body is {len(line1)} characters, expected 68: {line1!r}")
        if len(line2) != 68:
            raise ValueError(f"line 2 body is {len(line2)} characters, expected 68: {line2!r}")
        return line1 + str(tle_checksum(line1)), line2 + str(tle_checksum(line2))


def parse_lines(line1: str, line2: str) -> dict[str, float]:
    """Read the numeric fields back out of a two-line element set.

    This is a deliberately independent reader. It exists so that
    :meth:`OrbitElements.to_lines` can be round-tripped without using ``sgp4``
    as both the writer's and the reader's reference, and so that the checksum
    can be verified by something other than the function that wrote it.

    Returns
    -------
    dict
        Keys ``satnum``, ``epoch_year``, ``epoch_day``, ``bstar``,
        ``inclination_deg``, ``raan_deg``, ``eccentricity``, ``argp_deg``,
        ``mean_anomaly_deg``, ``mean_motion_rev_day``.

    Raises
    ------
    ValueError
        If either line is not 69 characters, does not start with its line
        number, or fails its checksum.
    """
    for index, line in ((1, line1), (2, line2)):
        if len(line) != TLE_LINE_LENGTH:
            raise ValueError(
                f"TLE line {index} must be {TLE_LINE_LENGTH} characters, got {len(line)}"
            )
        if line[0] != str(index):
            raise ValueError(f"TLE line {index} must begin with '{index}', got {line[0]!r}")
        expected = tle_checksum(line)
        if int(line[68]) != expected:
            raise ValueError(
                f"TLE line {index} checksum is {line[68]}, computed {expected}"
            )
    mantissa = line1[53:59]
    exponent = line1[59:61]
    bstar = float(f"{mantissa[0]}.{mantissa[1:]}") * 10.0 ** int(exponent)
    return {
        "satnum": float(int(line1[2:7])),
        "epoch_year": float(int(line1[18:20])),
        "epoch_day": float(line1[20:32]),
        "bstar": bstar,
        "inclination_deg": float(line2[8:16]),
        "raan_deg": float(line2[17:25]),
        "eccentricity": float("0." + line2[26:33]),
        "argp_deg": float(line2[34:42]),
        "mean_anomaly_deg": float(line2[43:51]),
        "mean_motion_rev_day": float(line2[52:63]),
    }
