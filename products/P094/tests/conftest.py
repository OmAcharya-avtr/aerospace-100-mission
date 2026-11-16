"""Shared fixtures.

Two kinds of data reach this suite and the split is deliberate.

**The committed hardware campaign** in ``data/traces/`` is read by the tests
that assert measured properties: the gate reachability of SPEC §9, the pinned
row counts, the pre-registration verdicts. It is committed, so it is present in
a cold clone and those tests are not skipped. If it is ever absent, they are
skipped with a stated reason rather than failed, because a missing measurement
is not a code defect, and the validation scripts under ``validation/`` refuse
loudly in that case.

**Simulation mode** is measured live, here, by the committed instrument. The
session fixture runs all nine cells at the board's own 1500-frame stack and one
pass, which takes about four seconds. The pass count is the one dimension that
was measured to be safe to shorten -- at 1, 2, 4 and 12 passes the held-out
transfer error measured 0.1601, 0.1554, 0.1622 and 0.1599 -- while the frame
stack is NOT: at 150 or 300 frames the figure crosses SPEC §7's 0.5 threshold.
``short_stack_simulation`` deliberately uses a 150-frame stack so that the
sensitivity is asserted rather than described.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

from hiltimingdiff import (
    baseline_1,
    baseline_2,
    baseline_3,
    calibrate_ns_per_op,
    confounding,
    divergence,
    drift_comparison,
    fit_drift,
    frequency_covariation,
    gate_census,
    load_campaign,
    measure_simulation,
    pass_units,
)
from hiltimingdiff.traces import TraceDataAbsentError

# Hypothesis writes a example database into .hypothesis/ by default, which is a
# build artifact inside the product tree. The profile below turns the database
# off: a falsifying example found by one run would be LOST on a fresh clone
# anyway, so every counterexample this suite has found is pinned as an explicit
# regression test with literal inputs in tests/test_properties.py instead.
settings.register_profile(
    "hiltimingdiff",
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("hiltimingdiff")

#: Passes for the session fixture. One pass over all nine cells at the board's
#: own stack length; measured safe to shorten, see the module docstring.
FIXTURE_PASSES = 1

#: Cells and stack length of the SHORT-STACK fixture, which exists to assert
#: the frame-stack sensitivity rather than to produce a reportable figure.
SHORT_STACK_CELLS = ("w48_sparse", "w64_sparse", "w96_sparse")
SHORT_STACK_PASSES = 2
SHORT_STACK_FRAMES = 150

#: Bootstrap resamples in the suite. Lower than the package default so the
#: suite stays fast; every test that asserts an interval asserts a property of
#: the interval (ordering, containment, recorded unit), never a pinned endpoint,
#: because an endpoint at 300 resamples is not the published endpoint.
FIXTURE_RESAMPLES = 300


@pytest.fixture(scope="session")
def campaign():
    """The committed hardware campaign, loaded once."""
    try:
        return load_campaign()
    except TraceDataAbsentError as error:  # pragma: no cover - committed in-tree
        pytest.skip(f"measured campaign absent: {error}")


@pytest.fixture(scope="session")
def units(campaign):
    """Per-pass units: 540 ``(cell, pass)`` rows x 6 stages."""
    return pass_units(campaign)


@pytest.fixture(scope="session")
def simulation(campaign):
    """A live simulation-mode measurement at the declared stack length."""
    return measure_simulation(campaign, passes=FIXTURE_PASSES)


@pytest.fixture(scope="session")
def short_stack_simulation(campaign):
    """A deliberately short-stack measurement, for the sensitivity assertions.

    Its figures are NOT reportable: the frame stack does not match the board's
    and the result says so. It exists so the suite asserts the sensitivity.
    """
    return measure_simulation(
        campaign,
        passes=SHORT_STACK_PASSES,
        frames=SHORT_STACK_FRAMES,
        cells=SHORT_STACK_CELLS,
    )


@pytest.fixture(scope="session")
def short_stack_baseline2(units, short_stack_simulation):
    """Baseline 2 fitted on the short-stack measurement."""
    return baseline_2(units, short_stack_simulation, resamples=FIXTURE_RESAMPLES)


@pytest.fixture(scope="session")
def ns_per_op(simulation):
    """The single global scalar of SPEC §3, from the fixture measurement."""
    return calibrate_ns_per_op(simulation)


@pytest.fixture(scope="session")
def divergence_result(units, ns_per_op):
    """The SPEC §5 outputs at the fixture's ``ns_per_op``."""
    return divergence(units, ns_per_op, resamples=FIXTURE_RESAMPLES)


@pytest.fixture(scope="session")
def baseline1(units, ns_per_op):
    """Baseline 1, the declared timing model."""
    return baseline_1(units, ns_per_op)


@pytest.fixture(scope="session")
def baseline2(units, simulation):
    """Baseline 2, single-scalar transfer to held-out stages."""
    return baseline_2(units, simulation, resamples=FIXTURE_RESAMPLES)


@pytest.fixture(scope="session")
def baseline3(units):
    """Baseline 3, ground truth with pass-unit intervals."""
    return baseline_3(units, resamples=FIXTURE_RESAMPLES)


@pytest.fixture(scope="session")
def drift(campaign):
    """The fitted board drift over the eleven calibration blocks."""
    return fit_drift(campaign)


@pytest.fixture(scope="session")
def comparison(campaign, units, ns_per_op):
    """Raw measured divergence beside the drift-adjusted variant."""
    return drift_comparison(campaign, units, ns_per_op, resamples=FIXTURE_RESAMPLES)


@pytest.fixture(scope="session")
def confound(campaign, units):
    """The confounding check on randomised cell order."""
    return confounding(campaign, units)


@pytest.fixture(scope="session")
def covariation(campaign, units):
    """Measured frequency co-variation, SPEC §6."""
    return frequency_covariation(campaign, units)


@pytest.fixture(scope="session")
def census(campaign):
    """The gate branch census, SPEC §9."""
    return gate_census(campaign)
