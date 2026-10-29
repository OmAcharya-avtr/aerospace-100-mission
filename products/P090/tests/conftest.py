"""Shared fixtures. Generated corpora are session-scoped because they cost seconds.

The published element set in :data:`REFERENCE_LINE1` and
:data:`REFERENCE_LINE2` is the one known-answer fixture in this suite that did
not come out of this repository's own generator. It is a two-line element set
for object 25544 at epoch 2019-343.69339541 as distributed in the ``sgp4``
project's own documentation, and it is used for exactly three things: the
checksum known-answer test, the character-for-character formatting round trip,
and the epoch decoding cross-check against ``sgp4``. No screening or
benchmarking number in this repository is derived from it.
"""

from __future__ import annotations

import pytest

from tlecheck import benchmark, dataset

REFERENCE_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
REFERENCE_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

REFERENCE_FIELDS = {
    "satnum": 25544,
    "epoch_year": 2019,
    "epoch_day": 343.69339541,
    "inclination_deg": 51.6439,
    "raan_deg": 211.2001,
    "eccentricity": 0.0007417,
    "arg_perigee_deg": 17.6667,
    "mean_anomaly_deg": 85.6398,
    "mean_motion_rev_per_day": 15.50103472,
    "bstar": 3.8792e-05,
    "mean_motion_dot_rev_per_day2": 1.764e-05,
    "designator": "98067A",
    "element_number": 999,
    "revolution_number": 20248,
}

SMALL_OBJECTS = 48
BENCH_OBJECTS = 180


@pytest.fixture(scope="session")
def reference_lines() -> tuple[str, str]:
    """The published element set lines, unmodified."""
    return REFERENCE_LINE1, REFERENCE_LINE2


@pytest.fixture(scope="session")
def reference_set():
    """The published element set, parsed."""
    from tlecheck.elements import parse_tle

    return parse_tle(REFERENCE_LINE1, REFERENCE_LINE2, source="published-reference")


@pytest.fixture(scope="session")
def small_records() -> list[dataset.Record]:
    """A small generated corpus, every corruption family represented."""
    return dataset.split("test", objects=SMALL_OBJECTS, history_length=12)


@pytest.fixture(scope="session")
def clean_record(small_records) -> dataset.Record:
    """The first uncorrupted record of the small corpus."""
    for record in small_records:
        if record.corruption == "none":
            return record
    raise AssertionError("the small corpus contains no uncorrupted record")


@pytest.fixture(scope="session")
def prepared_train():
    """The training split, screened, measured and featurised."""
    return benchmark.prepare(
        dataset.split("train", objects=BENCH_OBJECTS, history_length=12)
    )


@pytest.fixture(scope="session")
def prepared_test():
    """The test split, screened, measured and featurised."""
    return benchmark.prepare(
        dataset.split("test", objects=BENCH_OBJECTS, history_length=12)
    )


def record_of(records: list[dataset.Record], corruption: str) -> dataset.Record:
    """First record of ``corruption``, or fail the test saying which is missing."""
    for record in records:
        if record.corruption == corruption:
            return record
    raise AssertionError(f"no record with corruption {corruption!r} in the corpus")
