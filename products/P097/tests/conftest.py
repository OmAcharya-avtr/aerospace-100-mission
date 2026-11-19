"""Shared fixtures. Every number here comes from the committed history.

No test in this suite reads a clock. The product is about intermittency and a
product about intermittency whose own suite is intermittent has refuted itself;
the mission this product was built in lost a whole product to checks that
compared two wall-clock readings on a shared two-core runner. So every duration
any test asserts on is an integer number of milliseconds read from
``data/run_history.json``, which ``data/generate_history.py`` wrote once from a
fixed seed and which is committed with its SHA-256 digest.

The history fixture is session-scoped: loading and validating it costs about a
tenth of a second and the suite loads it once.
"""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.history import RunHistory, history_from_mapping, load_history

from _support import HISTORY_PATH, SEED, tiny_history


@pytest.fixture(scope="session")
def history() -> RunHistory:
    """The committed recorded run history."""
    return load_history(HISTORY_PATH)


@pytest.fixture(scope="session")
def summaries(history: RunHistory) -> tuple:
    """Per-test summaries over the whole committed history."""
    return history.summaries()


@pytest.fixture
def tiny() -> RunHistory:
    """The hand-checkable two-test, two-session history."""
    return history_from_mapping(tiny_history(), source="tiny")


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator at the declared seed, for tests that need draws."""
    return np.random.default_rng(SEED)
