"""Shared fixtures. Every campaign here is SYNTHETIC and is labelled so.

The Hypothesis example database is disabled rather than redirected. The
mission's build guide records that ``.hypothesis/`` is gitignored, so a
falsifying example found by one run is lost on a fresh clone; the response is to
pin every counterexample as an explicit regression test with its literal inputs,
which ``tests/test_regressions.py`` does. With the database off, no cache
directory is written into the tree by the test run at all.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Hypothesis 6.170 writes a `.hypothesis/constants` cache into the CURRENT
# WORKING DIRECTORY on import, regardless of whether the example database is
# disabled. Running pytest from the product root therefore leaves a cache
# directory in the tree. It is gitignored and so never reaches a clone, but the
# tree is cleaner if it is never created, and the only way to stop it is to
# point the storage directory elsewhere BEFORE hypothesis is imported. Hence the
# environment variable here and the E402 noqa below.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY",
    str(Path(tempfile.gettempdir()) / "thermalbudget-hypothesis"),
)

import pytest  # noqa: E402
from hypothesis import HealthCheck, settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import synthetic_trace  # noqa: E402

#: Examples per property. The default keeps the suite inside the release gate's
#: budget; a deeper hunt is run by setting the environment variable, which is
#: what ``validation/validate_property_hunt.py`` does so the deeper count is a
#: number someone executed rather than a claim.
MAX_EXAMPLES = int(os.environ.get("THERMALBUDGET_HYPOTHESIS_EXAMPLES", "60"))

settings.register_profile(
    "thermalbudget",
    database=None,
    deadline=None,
    max_examples=MAX_EXAMPLES,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.load_profile("thermalbudget")


@pytest.fixture(scope="session")
def no_throttle_campaign(tmp_path_factory):
    """A SYNTHETIC campaign whose clock never leaves its maximum.

    This is the regime the MEASURED campaign turned out to be in: across the
    three duty-1.0 soaks the die rose 10.94 degC and the mean CPU clock never
    left 1728000 kHz. It exists so that every degenerate code path -- no
    identifiable clock-versus-temperature relation, no identifiable
    clock-to-throughput exponent, baseline 2 reducing to baseline 1 -- is
    exercised by the suite.

    When this fixture was written the only evidence for the regime was the five
    rows of ``burst_1.00_a`` quoted in ``docs/SPEC.md``, and the docstring said
    so. It is now backed by the whole campaign
    (``validation/validate_clock_confound_output.txt``).
    """
    directory = tmp_path_factory.mktemp("no_throttle")
    return synthetic_trace.load_synthetic_campaign(directory, regime="no_throttle")


@pytest.fixture(scope="session")
def throttled_campaign(tmp_path_factory):
    """A SYNTHETIC campaign with a declared clock roll-off above a knee."""
    directory = tmp_path_factory.mktemp("throttled")
    return synthetic_trace.load_synthetic_campaign(directory, regime="throttled")


@pytest.fixture(scope="session")
def parameters():
    """The parameters the synthetic fixtures were generated with."""
    return synthetic_trace.SyntheticParameters(
        regime="throttled", seed=synthetic_trace.SEED
    )


@pytest.fixture
def short_run(tmp_path):
    """A SYNTHETIC loaded run far too short to have equilibrated.

    Used to exercise the ``NOT_EQUILIBRATED`` branch deterministically. A
    borderline run would make the assertion a coin toss on the noise draw.
    """
    from thermalbudget.provenance import Origin
    from thermalbudget.trace import load_trace

    path = synthetic_trace.write_run(
        tmp_path, "soak_1.00_short", 1.0, 150, regime="no_throttle"
    )
    return load_trace(path, origin=Origin.SYNTHETIC)
