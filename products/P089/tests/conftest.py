"""Shared fixtures. Every record used by the suite is generated here, once per module.

Fixtures are module-scoped rather than function-scoped because generating a
2000-sample record and assessing it costs a few milliseconds and the suite does
it hundreds of times. Nothing in the package mutates a record or a report --
both are frozen dataclasses holding arrays that are never written to -- so a
shared instance is safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import dataset, observability

#: Tolerance used wherever two floating-point statistics are compared and no
#: tighter bound is justified. Absolute, in the unit of the quantity compared.
#: Never used on a boolean produced by a threshold comparison.
LOOSE = 1.0e-9


@pytest.fixture(scope="module")
def rich():
    """A 2000-sample three-axis record: every degree of freedom observable."""
    return dataset.generate(profile="three_axis", samples=2000, seed=20261029)


@pytest.fixture(scope="module")
def single_axis():
    """A 2000-sample single-axis record: one degree of freedom unobservable."""
    return dataset.generate(profile="single_axis", samples=2000, seed=20261029)


@pytest.fixture(scope="module")
def near_single_axis():
    """A 2000-sample nearly-single-axis record: one degree of freedom marginal."""
    return dataset.generate(profile="near_single_axis", samples=2000, seed=20261029)


@pytest.fixture(scope="module")
def multi():
    """A record with three secondaries, each with its own mounting and misalignment."""
    return dataset.generate(profile="three_axis", samples=1500, n_secondaries=3, seed=777)


@pytest.fixture(scope="module")
def rich_verdict(rich):
    return observability.assess(rich.record)


@pytest.fixture(scope="module")
def single_verdict(single_axis):
    return observability.assess(single_axis.record)


@pytest.fixture(scope="module")
def noiseless_single():
    """A single-axis record whose Gramian is rank deficient in exact arithmetic.

    Built by hand rather than by the generator, because the generator always
    adds noise and the point of this fixture is the noiseless limit of equation
    (10). The reference and secondary rates are the same array, so the residual
    is identically zero and only the Gramian's structure is under test.
    """
    from aligncal.record import AlignmentRecord, SensorSpec

    times = np.arange(500, dtype=float) / 50.0
    axis = np.array([1.0, 2.0, -2.0]) / 3.0
    rates = (0.04 * np.sin(2.0 * np.pi * 0.3 * times))[:, None] * axis
    return AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 1.0e-4),
        reference_rates=rates,
        secondaries={"secondary_1": SensorSpec("secondary_1", 1.0e-4)},
        secondary_rates={"secondary_1": rates},
        provenance="tests/conftest.py noiseless_single",
    )
