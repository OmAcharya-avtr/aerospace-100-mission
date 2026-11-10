"""Shared fixtures. Every seed, record and network here is declared.

The ``short`` record -- two declared years rather than eight -- is used wherever
the claim under test does not depend on record length, because the whole suite
is run twice over by ``validation/validate_approx_audit.py`` and four more times
by ``validation/probe_working_directories.py``. Claims that DO depend on record
length use the reference record and say so in their own docstrings.

Hypothesis is DERANDOMISED here and its example database is disabled. Two
reasons, both of them defects this mission has shipped. First,
``validation/validate_approx_audit.py`` reports the NAMES of the tests whose
assertions rest on an absolute tolerance, and a predecessor shipped an audit
whose membership changed from run to run because Hypothesis drew different
examples each time, which makes the report worthless as evidence. Second,
``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone: every counterexample found during this build is pinned as an
explicit named test in ``tests/test_regressions.py`` with its literal inputs,
and the database is disabled so that a stale local database cannot make a
failing suite pass.
"""

from __future__ import annotations

import os
import tempfile

# Hypothesis writes its storage -- an example database, a constants cache and a
# Unicode table -- into `.hypothesis` in the CURRENT WORKING DIRECTORY, which
# for a suite run from the product root is the repository itself. The release
# gate runs pytest from there with `-p no:cacheprovider`, which suppresses
# `.pytest_cache` and says nothing about this one. Pointing the storage at a
# temporary directory outside the repository keeps the tree clean whoever runs
# the suite. `setdefault` so that an explicit setting from
# `validation/_bootstrap.subprocess_environment` wins.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY",
    tempfile.mkdtemp(prefix="sitediverse-hypothesis-"),
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hypothesis import HealthCheck, settings  # noqa: E402

from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.geometry import SITES, upper_triangle_pairs  # noqa: E402
from sitediverse.splits import make_split  # noqa: E402

settings.register_profile(
    "sitediverse",
    max_examples=30,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("sitediverse")

#: Declared station set used by the pair-level tests: the closest pair of the
#: first regional group, separation 106.2 km.
NEAR_PAIR: tuple[int, int] = (0, 1)

#: Declared station set used by the far-pair tests, separation 2557.5 km.
FAR_PAIR: tuple[int, int] = (0, 12)

#: Declared four-station network: the whole first regional group.
CLUSTER: tuple[int, ...] = (0, 1, 2, 3)


@pytest.fixture(scope="session")
def short_record():
    """The declared two-year record. Independent of record length by design."""
    return generate_record(RECORD_SPECS["short"])


@pytest.fixture(scope="session")
def reference_record():
    """The declared eight-year anisotropic record: the reference."""
    return generate_record(RECORD_SPECS["anisotropic"])


@pytest.fixture(scope="session")
def isotropic_record():
    """The declared eight-year isotropic control record."""
    return generate_record(RECORD_SPECS["isotropic"])


@pytest.fixture(scope="session")
def short_climatology(short_record):
    """The climatology estimated from the short record."""
    return estimate_climatology(short_record.cloud)


@pytest.fixture(scope="session")
def reference_climatology(reference_record):
    """The climatology estimated from the reference record."""
    return estimate_climatology(reference_record.cloud)


@pytest.fixture(scope="session")
def short_split(short_record):
    """The declared split of the short record."""
    return make_split(short_record.epochs, short_record.site_count)


@pytest.fixture(scope="session")
def all_pairs():
    """Every station pair of the declared network."""
    return upper_triangle_pairs(len(SITES))


def relative_error(measured: float, expected: float) -> float:
    """``|measured - expected| / |expected|``. Expected must be non-zero."""
    if expected == 0.0:
        raise ValueError(
            "relative_error has no meaning against an expected value of zero; "
            "compare with an absolute tolerance instead"
        )
    return float(abs(measured - expected) / abs(expected))


def max_relative_error(measured: np.ndarray, expected: np.ndarray) -> float:
    """Largest relative error between two arrays. Expected must be non-zero."""
    left = np.asarray(measured, dtype=float)
    right = np.asarray(expected, dtype=float)
    if np.any(right == 0.0):
        raise ValueError(
            "max_relative_error has no meaning where the expected array "
            "contains a zero"
        )
    return float(np.max(np.abs((left - right) / right)))
