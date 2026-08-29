"""Shared fixtures.

The trained model is expensive relative to a unit test, so it is built once
per session with a deliberately small configuration (180 scenarios x 2
realisations instead of the 900 x 3 production default). Statistical
assertions (interval coverage) use correspondingly wide tolerances; the
production-size numbers live in ``validation/benchmark_results.md``.
"""

from __future__ import annotations

import pytest

from turbscope.model import TurbScopeModel, train_default_model

SMALL_N_SCENARIOS = 180
SMALL_N_REALISATIONS = 2


@pytest.fixture(scope="session")
def small_model() -> tuple[TurbScopeModel, dict]:
    """A small, seeded, conformally calibrated model plus its data splits."""
    return train_default_model(
        n_scenarios=SMALL_N_SCENARIOS,
        n_realisations=SMALL_N_REALISATIONS,
        random_state=11,
    )
