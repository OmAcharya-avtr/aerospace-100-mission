"""Shared fixtures.

The robust invariant set costs about three seconds to compute (one linear
programme per facet per iteration), so it is session-scoped. Everything derived
from it is session-scoped too, except the guard, which carries a mutable
minimum-dwell counter and is therefore built fresh per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    SimplexGuard,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
)


@pytest.fixture(scope="session")
def plant():
    """The shipped illustrative single-axis attitude plant."""
    return reference_plant()


@pytest.fixture(scope="session")
def controllers(plant):
    """``(baseline, performance)`` for the reference plant."""
    return reference_controllers(plant)


@pytest.fixture(scope="session")
def invariant_result(plant, controllers):
    """The computed robust invariant set, with its iteration record."""
    return robust_invariant_set(plant, controllers[0])


@pytest.fixture(scope="session")
def invariant_set(invariant_result):
    """``S`` alone."""
    return invariant_result.polytope


@pytest.fixture
def guard(plant, controllers, invariant_set):
    """A fresh guard with no hysteresis."""
    return SimplexGuard(plant, controllers[0], invariant_set)


@pytest.fixture(scope="session")
def small_plant():
    """A one-dimensional plant whose robust invariant set has a closed form.

    ``x_{k+1} = a x_k + u_k + w_k`` with ``|x| <= xmax``, ``|u| <= umax`` and
    ``|w| <= wmax``. Under the baseline ``u = -k x`` the closed loop is
    ``x_{k+1} = (a - k) x_k + w_k``, and for ``|a - k| < 1`` the maximal robust
    invariant set inside ``|x| <= xmax`` is

        |x| <= min(xmax, wmax / (1 - |a - k|)),

    which is the standard geometric-series bound and is the known answer the
    tests in ``tests/test_invariant.py`` check against by hand.
    """
    from simplexguard import Plant, box

    a = np.array([[1.2]])
    b = np.array([[1.0]])
    return Plant(
        A=a,
        B=b,
        disturbance=box([0.05]),
        state_constraints=box([1.0]),
        input_constraints=box([5.0]),
        dt=1.0,
    )


@pytest.fixture(scope="session")
def rng():
    """A fixed generator, so every test that samples is reproducible."""
    return np.random.default_rng(51051)
