"""Shared fixtures. Every seed, campaign and amplitude here is declared.

The ``short`` campaign is used wherever the claim under test does not depend on
record length, because the suite is run twice over by
``validation/validate_approx_audit.py`` and once more, four times, by
``validation/probe_working_directories.py``. Claims that DO depend on record
length are tested on the ``full`` campaign and say so in their own docstrings.
"""

from __future__ import annotations

import os
import tempfile

# Hypothesis writes its own storage -- an example database, a constants cache
# and a Unicode table -- into `.hypothesis` in the CURRENT WORKING DIRECTORY,
# which for a suite run from the product root is the repository itself. The
# release gate runs pytest from there with `-p no:cacheprovider`, which
# suppresses `.pytest_cache` and says nothing about this one. Pointing the
# storage at a temporary directory outside the repository keeps the tree clean
# whoever runs the suite. `setdefault` so that an explicit setting from
# `validation/_bootstrap.subprocess_environment` wins.
#
# This is set before `hypothesis` is imported, because the location is read at
# import time. If a future pytest loads the hypothesis plugin before this
# conftest -- plugin entry points are loaded first in general -- the directory
# reappears in the tree; it is gitignored with an anchored pattern and
# `validation/validate_cli.py` reports it as inherited rather than failing on
# it, so that case costs nothing but is worth knowing about.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY",
    tempfile.mkdtemp(prefix="twindiverge-hypothesis-"),
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hypothesis import HealthCheck, settings  # noqa: E402

from twindiverge.identifiability import analyse
from twindiverge.plant import (
    CAMPAIGNS,
    NOMINAL,
    Campaign,
    WheelParameters,
    simulate_with_sensitivities,
)
from twindiverge.telemetry import TELEMETRY_SETS, TelemetrySet

# Hypothesis is DERANDOMISED here and its example database is disabled.
#
# Two reasons, both of them defects this mission has shipped. First,
# `validation/validate_approx_audit.py` runs the whole suite twice and reports
# the NAMES of the tests whose assertions rest on an absolute tolerance; a
# predecessor shipped an audit whose reported membership changed from run to run
# because Hypothesis generated different examples each time, which makes the
# report worthless as evidence. Second, `.hypothesis/` is gitignored, so a
# falsifying example found by one run is lost on a fresh clone -- every
# counterexample Hypothesis finds here is pinned as an explicit named test in
# tests/test_regressions.py with its literal inputs, and the database is
# disabled so that a stale local database cannot make a failing suite pass.
settings.register_profile(
    "twindiverge",
    max_examples=30,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("twindiverge")

#: A declared parameter set with a time constant of about one second, used only
#: to measure the integrator's convergence order. Equation (1) integrated with
#: the declared NOMINAL parameters already agrees with the closed form of
#: twindiverge.reference to 6.6e-15 relative at a 0.1 s step, so its truncation
#: error is below rounding and an order cannot be measured from it. This set is
#: a numerical exercise and not a claim about any wheel: a viscous coefficient
#: of 1.3e-2 N m s / rad and a torque constant of 1.3025 N m / A are both
#: implausible by three orders of magnitude.
ORDER_PARAMETERS = WheelParameters(1.30e-2, 1.3025, 2.50e-3, 1.30e-2, 2.00e-8)

#: Initial rate of the convergence-order exercise, rad/s.
ORDER_INITIAL_RATE = 150.0

#: Duration of the convergence-order exercise, s.
ORDER_DURATION = 2.0


@pytest.fixture(scope="session")
def nominal() -> WheelParameters:
    """The declared nominal parameters."""
    return NOMINAL


@pytest.fixture(scope="session")
def short_set() -> TelemetrySet:
    """The 20 s rate-only telemetry set."""
    return TELEMETRY_SETS["rate_short"]


@pytest.fixture(scope="session")
def full_set() -> TelemetrySet:
    """The 200 s rate-only telemetry set: the reference set."""
    return TELEMETRY_SETS["rate_full"]


@pytest.fixture(scope="session")
def torque_set() -> TelemetrySet:
    """The 20 s two-channel telemetry set, which is structurally full rank."""
    return TELEMETRY_SETS["rate_torque_short"]


@pytest.fixture(scope="session")
def coast_set() -> TelemetrySet:
    """The undriven coast-down set, whose null space is two-dimensional."""
    return TELEMETRY_SETS["rate_coast"]


@pytest.fixture(scope="session")
def short_twin():
    """The nominal trajectory with sensitivities over the ``short`` campaign."""
    return simulate_with_sensitivities(NOMINAL, CAMPAIGNS["short"])


@pytest.fixture(scope="session")
def short_report(short_set: TelemetrySet):
    """Baseline 1 on the ``rate_short`` set."""
    return analyse(short_set, NOMINAL)


@pytest.fixture(scope="session")
def torque_report(torque_set: TelemetrySet):
    """Baseline 1 on the ``rate_torque_short`` set."""
    return analyse(torque_set, NOMINAL)


def order_campaign(step_s: float) -> Campaign:
    """A driven campaign of the declared order-exercise duration at ``step_s``."""
    samples = int(round(ORDER_DURATION / step_s)) + 1
    return Campaign(
        f"order{step_s}",
        step_s,
        samples,
        1.0,
        ORDER_DURATION,
        initial_rate_rad_s=ORDER_INITIAL_RATE,
    )


def relative_error(measured: np.ndarray, expected: np.ndarray) -> float:
    """Largest relative error between two arrays. Both must be non-zero."""
    left = np.asarray(measured, dtype=float)
    right = np.asarray(expected, dtype=float)
    return float(np.max(np.abs((left - right) / right)))
