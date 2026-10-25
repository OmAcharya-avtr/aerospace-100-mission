"""The published SGP4 verification test vectors, parsed independently.

Where these come from
---------------------
Both files ship **inside the installed ``sgp4`` distribution** (Brandon
Rhodes, version 2.27) as package data:

``SGP4-VER.TLE``
    The 33 verification two-line element sets, each line 2 extended past
    column 69 with a propagation start, stop and step in minutes.
``tcppver.out``
    The expected position and velocity at each of those times, 700 lines.

They are the verification set published with Vallado, Crawford, Hujsak and
Kelso (2006), "Revisiting Spacetrack Report #3", AIAA 2006-6753, and the
accompanying ``testcpp.cpp`` reference program; the sgp4 distribution carries
them unchanged so that its own test suite can regenerate the second from the
first. ``validation/validate_sgp4_vectors.py`` records the exact installed
distribution name and version it read them from.

What this module does **not** do
--------------------------------
It does not import ``sgp4.tests``. That module regenerates ``tcppver.out`` as
*text* with the same format strings the C++ program used and compares the
strings, so a bug shared between the generator and the formatter would be
invisible to it. The parser here reads the published numbers as numbers and
compares them numerically to a propagation this package drives itself, which
is a different check on the same data. The two are not substitutes: the string
comparison also checks the derived orbital elements and the calendar date,
which this module ignores.

Rows for which the propagator returns a non-zero error code are **reported and
skipped, not silently dropped**. Seven of the 33 cases are deliberate failure
cases -- decayed orbits, negative eccentricity after the Lyddane correction,
a negative semi-latus rectum -- and ``tcppver.out`` handles one of them by
repeating the previous satellite's last line, a quirk of the reference
program. A parser that compared that row would report a spurious mismatch of
thousands of kilometres. :func:`compare_to_published` counts those rows
separately and names them.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pkgutil import get_data

import numpy as np
from sgp4.api import Satrec

from .errors import InvalidTleError

TLE_FILE = "SGP4-VER.TLE"
EXPECTED_FILE = "tcppver.out"


def sgp4_distribution_version() -> str:
    """Installed version of the ``sgp4`` distribution, or ``"unknown"``."""
    try:
        return version("sgp4")
    except PackageNotFoundError:  # pragma: no cover - sgp4 is a hard dependency
        return "unknown"


def _read(name: str) -> str:
    raw = get_data("sgp4", name)
    if raw is None:  # pragma: no cover - would mean a broken sgp4 install
        raise InvalidTleError(
            f"the installed sgp4 distribution does not carry {name}; "
            "the published verification vectors cannot be read"
        )
    return raw.decode("ascii").replace("\r", "")


@dataclass(frozen=True)
class PublishedCase:
    """One verification satellite and the rows published for it.

    Attributes
    ----------
    satnum
        NORAD catalogue number as the file gives it.
    line1, line2
        The two element-set lines, truncated to 69 columns.
    schedule_min
        ``(start, stop, step)`` in minutes from epoch, from columns 70 onward
        of line 2.
    times_min
        Published times since epoch, minutes, shape ``(m,)``.
    positions_km, velocities_km_s
        Published TEME states, shapes ``(m, 3)``.
    """

    satnum: int
    line1: str
    line2: str
    schedule_min: tuple[float, float, float]
    times_min: np.ndarray
    positions_km: np.ndarray
    velocities_km_s: np.ndarray


def published_cases() -> list[PublishedCase]:
    """Parse ``SGP4-VER.TLE`` and ``tcppver.out`` into comparable cases."""
    tle_lines = [
        line for line in _read(TLE_FILE).splitlines() if line[:2] in ("1 ", "2 ")
    ]
    if len(tle_lines) % 2 != 0:
        raise InvalidTleError(
            f"{TLE_FILE} holds {len(tle_lines)} element lines, which is not a whole "
            "number of two-line sets"
        )

    # tcppver.out is grouped into blocks headed "<satnum> xx". The blocks are
    # matched to the element sets **positionally**, not by catalogue number,
    # because SGP4-VER.TLE deliberately lists some catalogue numbers twice --
    # the same object with a different propagation schedule. Keying on the
    # catalogue number merges those blocks and silently compares a satellite
    # against another test case's rows; it inflated the row count of this
    # parser from 667 to 763 before it was found.
    blocks: list[tuple[int, list[list[float]]]] = []
    for line in _read(EXPECTED_FILE).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith("xx"):
            blocks.append((int(stripped.split()[0]), []))
            continue
        if not blocks:
            raise InvalidTleError(f"{EXPECTED_FILE} has a data row before any header")
        fields = stripped.split()
        if len(fields) < 7:
            continue
        blocks[-1][1].append([float(value) for value in fields[:7]])

    if len(blocks) != len(tle_lines) // 2:
        raise InvalidTleError(
            f"{EXPECTED_FILE} holds {len(blocks)} blocks but {TLE_FILE} holds "
            f"{len(tle_lines) // 2} element sets; the two files are not the pair "
            "this parser expects"
        )

    cases: list[PublishedCase] = []
    for index in range(0, len(tle_lines), 2):
        line1, line2 = tle_lines[index], tle_lines[index + 1]
        satnum = int(line1[2:7].strip())
        schedule_text = line2[69:].split()
        if len(schedule_text) != 3:
            raise InvalidTleError(
                f"satellite {satnum} in {TLE_FILE} does not carry a three-field "
                f"start/stop/step after column 69; found {schedule_text!r}"
            )
        schedule = (
            float(schedule_text[0]),
            float(schedule_text[1]),
            float(schedule_text[2]),
        )
        block_satnum, block_rows = blocks[index // 2]
        if block_satnum != satnum:
            raise InvalidTleError(
                f"block {index // 2} of {EXPECTED_FILE} is headed {block_satnum} but "
                f"the matching element set is {satnum}; the files are out of order"
            )
        rows = np.asarray(block_rows, dtype=float).reshape(-1, 7)
        cases.append(
            PublishedCase(
                satnum=satnum,
                line1=line1[:69],
                line2=line2[:69],
                schedule_min=schedule,
                times_min=rows[:, 0],
                positions_km=rows[:, 1:4],
                velocities_km_s=rows[:, 4:7],
            )
        )
    return cases


@dataclass(frozen=True)
class VectorComparison:
    """Result of comparing one published case against a live propagation."""

    satnum: int
    rows_compared: int
    rows_skipped_error_code: int
    error_codes: tuple[int, ...]
    max_position_error_km: float
    max_velocity_error_km_s: float


def compare_to_published(case: PublishedCase) -> VectorComparison:
    """Propagate ``case`` with ``sgp4`` and compare against its published rows.

    The published file prints position to 8 decimal places of a kilometre and
    velocity to 9 decimal places of a km/s, so the comparison cannot be
    tighter than about 5e-9 km and 5e-10 km/s from rounding of the file alone.
    The returned maxima are absolute differences in km and km/s.

    ``Satrec.sgp4_tsince`` is used rather than a Julian-date call, because the
    published times are minutes since epoch and converting them to a Julian
    date and back introduces a rounding error of its own that would be
    mistaken for propagator disagreement.
    """
    satrec = Satrec.twoline2rv(case.line1, case.line2)
    worst_r = 0.0
    worst_v = 0.0
    compared = 0
    skipped = 0
    codes: list[int] = []
    for index, tsince in enumerate(case.times_min):
        code, r, v = satrec.sgp4_tsince(float(tsince))
        if code != 0 or not np.all(np.isfinite(r)):
            skipped += 1
            codes.append(int(code))
            continue
        worst_r = max(worst_r, float(np.max(np.abs(np.asarray(r) - case.positions_km[index]))))
        worst_v = max(
            worst_v, float(np.max(np.abs(np.asarray(v) - case.velocities_km_s[index])))
        )
        compared += 1
    return VectorComparison(
        satnum=case.satnum,
        rows_compared=compared,
        rows_skipped_error_code=skipped,
        error_codes=tuple(codes),
        max_position_error_km=worst_r,
        max_velocity_error_km_s=worst_v,
    )
