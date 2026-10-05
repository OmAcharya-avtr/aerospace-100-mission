"""Shared fixtures. The trained reference model is session-scoped because
fitting it costs about two seconds on the single contended core this suite runs
on, and sixty tests do not need sixty fits."""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.bitlayout import float_layout
from bitflipsim.datasets import make_problem, reference_parameters


@pytest.fixture(scope="session")
def problem():
    return make_problem(n_train=400, n_calibration=120, n_evaluation=120)


@pytest.fixture(scope="session")
def params(problem):
    return reference_parameters(problem)


@pytest.fixture(scope="session")
def f32():
    return float_layout("float32")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
